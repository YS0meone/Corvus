import asyncio
from langchain.tools import tool, ToolRuntime
from typing import List, Optional
from app.services.qdrant import QdrantService
from app.services.s2_client import S2Client
from app.core.config import settings
from pydantic import BaseModel, Field
from langgraph.types import Command
from app.core.schema import S2Paper
from langchain_tavily import TavilySearch
from app.agent.utils import get_paper_info_text
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from pydantic import ConfigDict
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage
from typing import Literal
from app.agent.utils import get_paper_abstract, remove_duplicated_evidence

filter_model = init_chat_model(model=settings.PF_FILTER_MODEL_NAME)

async def llm_document_filter_batch(evds: List[Document], query: str, abstracts: str, batch_size: int = 3) -> List[int]:

    llm_document_filter_system = """
    You are an expert in document filtering for academic paper QA.
    You are given a list of documents, a user query and the paper abstracts.
    You need to filter the documents based on the user query and the paper abstracts.
    If the document is relevant to answer the user question or even helpful to understand the user question, you should keep it
    If the document is not relevant to answer the user question or not helpful to understand the user question, you should filter it
    You should output a list of index of the documents to keep.
    """

    results = []
    for i in range(0, len(evds), batch_size):
        batch_evds = evds[i:i+batch_size]
        batch_evidence_text = "\n".join([f"Documents {i}:\n{evd.page_content}" for i, evd in enumerate(batch_evds)])
        llm_document_filter_prompt = f"""
        User query: {query}
        Paper abstracts: {abstracts}
        Documents: {batch_evidence_text}
        """
        class DocumentFilterResult(BaseModel):
            decisions: List[Literal["0", "1", "2"]] = Field(description="The index of the documents to keep")

        structured_model = filter_model.with_structured_output(DocumentFilterResult)
        llm_document_filter_response = await structured_model.ainvoke([
            SystemMessage(content=llm_document_filter_system),
            HumanMessage(content=llm_document_filter_prompt)
        ])
        index_to_keep = [int(decision) + i for decision in llm_document_filter_response.decisions]
        results.extend(index_to_keep)
    return results


@tool
async def retrieve_evidence_from_selected_papers(
    runtime: ToolRuntime,
    reasoning: str,
    query: str,
    limit: int = 10,
    score_threshold: Optional[float] = None,
) -> List[Document]:
    """
    Retrieve relevant evidence from the selected papers based on the query and automatically merge the evidence with the existing evidence.

    **When to use this tool:**
    - When you want to retrieve relevant evidence from the selected papers based on the query
    - When the evidence is not sufficient to answer the user question
    - When you need to retrieve more information from the selected papers to better understand the user question

    Args:
        reasoning: The reasoning for why you choose these arguments to retrieve evidence
        query: The search query string used to retrieve relevant evidence from the selected papers   
        limit: Maximum number of results to return (default: 10, max: 30), set larger number if you need more evidence or comprehensive coverage.
        score_threshold: Optional minimum similarity threshold (0.0-1.0) to filter results. Set higher value if you need more relevant evidence and lower for more comprehensive coverage.
    """
    tool_call_id = runtime.tool_call_id
    state = runtime.state if runtime else {}
    ids = state.get("selected_paper_ids", [])
    papers = state.get("papers", [])
    abstracts = get_paper_abstract(papers, ids)
    results = []
    existing_evds = state.get("evidences", [])

    try:
        # QdrantService.__init__ makes a blocking socket call (collection_exists),
        # so instantiation must happen inside the thread, not on the event loop.
        def _search():
            qdrant_service = QdrantService(settings.qdrant_config)
            return qdrant_service.search_selected_ids(
                ids=ids,
                query=query,
                k=min(limit, 30),
                score_threshold=score_threshold,
            )
        results = await asyncio.to_thread(_search)
        results = remove_duplicated_evidence(existing_evds, results)
        index_to_keep = await llm_document_filter_batch(results, query, abstracts, batch_size=3)
        results = [results[i] for i in index_to_keep if 0 <= i < len(results)]
    except Exception as e:
        return Command(
            update={"messages": [ToolMessage(
                content=f"Error during retrieve evidence from selected papers: {str(e)}",
                tool_call_id=tool_call_id
            )]}
        )
    # using reducer to merge the result and avoid racing condition
    return Command[tuple[()]](
        update={
            "messages": [ToolMessage(
                content=f"I found {len(results)} relevant evidence for your query.",
                tool_call_id=tool_call_id
            )],
            "evidences": results
        }
    )


class S2SearchPapersRequest(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    runtime: ToolRuntime
    reasoning: str = Field(..., description="Explain your search strategy: what are you looking for and why this query formulation?")
    query: str = Field(
        ...,
        description="""Search query for Semantic Scholar. Follow these guidelines for effective queries:

QUERY LENGTH: Keep queries SHORT (2-5 keywords). The API matches against title and abstract.
- BAD: "What are the methods for improving efficiency of large language models through quantization and pruning techniques?"
- GOOD: "language model quantization pruning"

PHRASE MATCHING: Use quotes for exact multi-word phrases.
- "graph neural network" (matches exact phrase)
- "large language model" efficiency (phrase + keyword)

BOOLEAN OPERATORS: Use + (AND) and | (OR) to combine terms.
- transformer + attention (both required)
- BERT | RoBERTa | GPT (any of these)
- "neural network" + (pruning | quantization) (phrase AND either term)

KEYWORD STRATEGIES:
- Use technical terms, not natural language questions
- Include acronyms AND full names: "LLM" | "large language model"
- Prefer nouns over verbs: "image classification" not "classifying images"
- Omit common words: the, a, an, of, for, in, on

EXAMPLES:
- Finding a specific paper: Use match_title=True with the paper title
- Methodology papers: "transformer attention mechanism"
- Survey papers: "survey" + "deep learning"
- Recent techniques: Use year filter instead of adding "recent" to query"""
    )
    year: str = Field(
        None,
        description="Filter by publication year range. Examples: '2023' (single year), '2020-2024' (range). Use this instead of adding year terms to query."
    )
    venue: List[str] = Field(
        None,
        description="Filter by venue names (conferences/journals). Examples: ['NeurIPS', 'ICML', 'ACL']. Case-insensitive."
    )
    fields_of_study: List[str] = Field(
        None,
        description="Filter by field. Options: 'Computer Science', 'Mathematics', 'Physics', 'Biology', 'Medicine', etc. Use for broad domain filtering."
    )
    publication_date_or_year: str = Field(
        None,
        description="Filter by date range in format 'YYYY-MM-DD:YYYY-MM-DD' or 'YYYY:YYYY'. Example: '2023-01-01:2024-12-31'"
    )
    min_citation_count: int = Field(
        None,
        description="Minimum citations required. Use to find influential papers (e.g., 100 for well-cited, 1000 for seminal works)."
    )
    match_title: bool = Field(
        False,
        description="Set True to find ONE paper by exact title match. Use when you know the specific paper title."
    )

@tool(
    args_schema=S2SearchPapersRequest
    )
async def s2_search_papers(
    runtime: ToolRuntime,
    reasoning: str,
    query: str,
    year: str = None,
    venue: List[str] = None,
    fields_of_study: List[str] = None,
    publication_date_or_year: str = None,
    min_citation_count: int = None,
    match_title: bool = False,
):
    """
    Search academic papers using Semantic Scholar's database of 200M+ papers.

    WHEN TO USE:
    - Finding papers on a specific research topic
    - Locating papers by methodology or technique
    - Finding influential/highly-cited papers in a field
    - Looking up a specific paper by title (use match_title=True)

    QUERY BEST PRACTICES:
    - Keep queries SHORT: 2-5 keywords, not full sentences
    - Use quotes for phrases: "graph neural network"
    - Use + for AND: transformer + efficiency
    - Use | for OR: BERT | GPT | T5
    - Combine with parentheses: "attention" + (vision | image)
    - Use FILTERS (year, venue, min_citation_count) instead of adding filter words to query

    EXAMPLES OF GOOD QUERIES:
    - "knowledge distillation" + transformer
    - "reinforcement learning" + robotics
    - BERT | RoBERTa (to find papers on either)
    - "graph neural network" + (node | link) + prediction

    EXAMPLES OF BAD QUERIES (too long/natural language):
    - "What are the recent advances in transformer architectures for NLP tasks?"
    - "Papers about how to make neural networks more efficient using pruning"

    Returns: List of papers with title, abstract, authors, year, and citation count.
    """
    # Get state from runtime (if available)
    state = runtime.state if runtime else {}
    tool_call_id = runtime.tool_call_id
    # Get new papers from S2
    try:
        s2_client = S2Client()
        new_results = await s2_client.search_papers(
            query=query,
            year=year,
            venue=venue,
            fields_of_study=fields_of_study,
            publication_date_or_year=publication_date_or_year,
            min_citation_count=min_citation_count,
            match_title=match_title
        )
    except Exception as e:
        error_msg = str(e)
        if "Title match not found" in error_msg or "ObjectNotFoundException" in str(type(e)):
            # Handle title match failure gracefully
            return Command(
                update={"messages": [ToolMessage(content=f"No exact title match found for query: '{query}'. Try searching without title matching or rephrase the query.", tool_call_id=tool_call_id)]}
            )
        else:
            return Command(
                update={"messages": [ToolMessage(content=f"Error searching for papers: {error_msg}", tool_call_id=tool_call_id)]}
            )
    
    return Command(
        update={"papers": new_results, "messages": [
            ToolMessage(
                content=f"I found {len(new_results)} new papers for your query.",
                tool_call_id=tool_call_id
                )]}
    )

class TavilySearchRequest(BaseModel):
    reasoning: str = Field(..., description="Explain why you need to understand this research topic better and what you hope to learn")
    query: str = Field(..., description="A natural language query about the research topic or field you want to understand")


@tool(args_schema=TavilySearchRequest)
def tavily_research_overview(reasoning: str, query: str) -> str:
    """
    Search for general information about research topics using Tavily web search.
    
    **When to use this tool:**
    - When the research topic is GENERAL or UNFAMILIAR to you
    - When you need to understand what a research field or topic is about
    - When you want to identify the most FAMOUS/SEMINAL papers in a field
    - BEFORE searching academic databases, to avoid missing important foundational work
    - When you need context about trending research areas or breakthrough papers
    
    **This tool helps you:**
    - Understand the landscape of a research topic
    - Find names of influential papers, authors, and concepts
    - Identify key terms and methodologies to use in academic searches
    - Discover landmark papers that should not be missed
    
    **Example use cases:**
    - "What are the most important papers in transformer architectures?"
    - "What is the current state of research in quantum machine learning?"
    - "Who are the key researchers and what are the foundational papers in reinforcement learning?"
    
    Args:
        reasoning: Your reasoning for why you need this overview
        query: Natural language question about the research topic
        
    Returns:
        Web search results with information about the research topic, including
        mentions of famous papers, key researchers, and important concepts.
    """
    try:
        tavily = TavilySearch(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
            include_images=False
        )
        
        results = tavily.invoke({"query": query})
        
        # Format results for better readability
        if isinstance(results, list):
            formatted_output = []
            formatted_output.append(f"Research Topic Overview for: {query}\n")
            formatted_output.append("=" * 80 + "\n")
            
            for i, result in enumerate(results, 1):
                if isinstance(result, dict):
                    title = result.get('title', 'No title')
                    content = result.get('content', 'No content')
                    url = result.get('url', '')
                    
                    formatted_output.append(f"\n{i}. {title}")
                    formatted_output.append(f"   URL: {url}")
                    formatted_output.append(f"   Content: {content}\n")
            
            return "\n".join(formatted_output)
        else:
            return str(results)
            
    except Exception as e:
        return f"Error searching with Tavily: {str(e)}. You may proceed with direct academic database search."
    
@tool
def get_paper_details(
    runtime: ToolRuntime
) -> str:
    """
    Get the metadata of the papers in the paper list. Call it if you need to know the details of the papers in the paper list.

    **When to use this tool:**
    - When you want to check if certain paper in the paper list
    - When you want to know what papers you already have to decide what to do next
    
    Returns:
    The metadata of all papers in the paper list.
    """

    state = runtime.state if runtime else {}
    papers = state.get("papers", [])
    if len(papers) == 0:
        return "No paper in the paper list."

    paper_info_text = get_paper_info_text(papers)
    return paper_info_text


class BackwardSnowballRequest(BaseModel):
    """Request schema for backward snowball search."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    runtime: ToolRuntime
    reasoning: str = Field(
        ...,
        description="Explain why you want to trace back the foundational papers that these seed papers cite"
    )
    seed_paper_ids: List[str] = Field(
        ...,
        description="List of Semantic Scholar paper IDs to use as seed papers. These are the papers whose references (bibliography) you want to explore."
    )
    top_k: int = Field(
        10,
        description="Number of top referenced papers to return (default: 10, max: 50)"
    )


@tool(args_schema=BackwardSnowballRequest)
async def backward_snowball(
    runtime: ToolRuntime,
    reasoning: str,
    seed_paper_ids: List[str],
    top_k: int = 10
):
    """
    Backward Snowball: Trace foundational papers by following what seed papers CITE (their references/bibliography).

    **What this does:**
    - Takes a list of paper IDs as seed papers
    - Fetches all papers that these seed papers reference/cite
    - Scores candidates based on citation count and how many seeds cite them
    - Returns the top-k most relevant referenced papers

    **When to use:**
    - You want to explore the foundational papers that influenced specific papers
    - You're tracing back to the original/parent works
    - You want to understand what prior work specific papers build upon

    **Example:**
    For papers about "GPT-3" and "BERT", this will find papers like
    "Attention is All You Need" (Transformer paper) that they cite.

    Args:
        reasoning: Why you want to find referenced papers
        seed_paper_ids: List of Semantic Scholar paper IDs (e.g., ["paperId1", "paperId2"])
        top_k: Number of top papers to return (default: 10, max: 50)

    Returns:
        Top-k most relevant papers cited by the seed papers
    """
    tool_call_id = runtime.tool_call_id
    
    if not seed_paper_ids:
        return Command(
            update={"messages": [ToolMessage(
                content="No seed paper IDs provided. Please specify which papers to use for citation chasing.",
                tool_call_id=tool_call_id
            )]}
        )
    
    top_k = min(max(1, top_k), 50)  # Clamp between 1 and 50
    
    try:
        s2_client = S2Client()
        
        # Fetch references for all seed papers
        all_references = {}  # corpus_id -> {paper_data, num_seeds_citing: int}
        
        for paper_id in seed_paper_ids:
            try:
                references = await s2_client.get_paper_references(
                    paper_id=paper_id,
                    fields=["paperId", "corpusId", "title", "abstract", "authors",
                           "year", "citationCount", "influentialCitationCount"]
                )
                
                if not references:
                    continue
                
                for ref in references:
                    corpus_id = ref.get("corpusId")
                    if not corpus_id:
                        continue
                    
                    if corpus_id not in all_references:
                        all_references[corpus_id] = {
                            "paper": ref,
                            "num_seeds_citing": 0
                        }
                    
                    all_references[corpus_id]["num_seeds_citing"] += 1
                        
            except Exception as e:
                continue
        
        if not all_references:
            return Command(
                update={"messages": [ToolMessage(
                    content=f"Could not find references for the {len(seed_paper_ids)} seed papers.",
                    tool_call_id=tool_call_id
                )]}
            )
        
        # Score and sort candidates
        scored_candidates = []
        for corpus_id, data in all_references.items():
            paper = data["paper"]
            num_seeds = data["num_seeds_citing"]
            citation_count = paper.get("citationCount", 0)
            
            # Simple scoring: prioritize papers cited by multiple seeds and highly cited
            score = num_seeds * 10.0 + min(citation_count / 100, 10.0)
            
            scored_candidates.append({"score": score, "paper": paper})
        
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = scored_candidates[:top_k]
        
        # Convert to S2Paper objects
        result_papers = []
        for candidate in top_candidates:
            paper_data = candidate["paper"]
            try:
                s2_paper = S2Paper(
                    paperId=paper_data.get("paperId", ""),
                    corpusId=paper_data.get("corpusId"),
                    title=paper_data.get("title"),
                    abstract=paper_data.get("abstract"),
                    authors=paper_data.get("authors"),
                    year=paper_data.get("year"),
                    citationCount=paper_data.get("citationCount"),
                    influentialCitationCount=paper_data.get("influentialCitationCount"),
                    url=paper_data.get("url", f"https://www.semanticscholar.org/paper/{paper_data.get('paperId', '')}")
                )
                result_papers.append(s2_paper)
            except Exception:
                continue
        
        return Command(
            update={
                "papers": result_papers,
                "messages": [ToolMessage(
                    content=f"Found {len(result_papers)} papers cited by the {len(seed_paper_ids)} seed papers.",
                    tool_call_id=tool_call_id
                )]
            }
        )
        
    except Exception as e:
        return Command(
            update={"messages": [ToolMessage(
                content=f"Error during backward snowball: {str(e)}",
                tool_call_id=tool_call_id
            )]}
        )


class ForwardSnowballRequest(BaseModel):
    """Request schema for forward snowball search."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    runtime: ToolRuntime
    reasoning: str = Field(
        ...,
        description="Explain why you want to find newer papers that cite these seed papers"
    )
    seed_paper_ids: List[str] = Field(
        ...,
        description="List of Semantic Scholar paper IDs to use as seed papers. These are the papers whose citations (papers that cite them) you want to explore."
    )
    top_k: int = Field(
        10,
        description="Number of top citing papers to return (default: 10, max: 50)"
    )


@tool(args_schema=ForwardSnowballRequest)
async def forward_snowball(
    runtime: ToolRuntime,
    reasoning: str,
    seed_paper_ids: List[str],
    top_k: int = 10
):
    """
    Forward Snowball: Find newer work by following who CITES the seed papers.

    **What this does:**
    - Takes a list of paper IDs as seed papers
    - Fetches all papers that cite these seed papers
    - Scores candidates based on citation count, recency, and how many seeds they cite
    - Returns the top-k most relevant citing papers

    **When to use:**
    - You want to find recent work building on specific foundational papers
    - You're looking for "child" papers that extend or apply seed papers
    - You want to discover how specific papers have been used or cited

    **Example:**
    For the "Attention is All You Need" paper, this will find papers like
    "BERT", "GPT-3", and other transformer-based models that cite it.

    Args:
        reasoning: Why you want to find citing papers
        seed_paper_ids: List of Semantic Scholar paper IDs (e.g., ["paperId1", "paperId2"])
        top_k: Number of top papers to return (default: 10, max: 50)

    Returns:
        Top-k most relevant papers that cite the seed papers
    """
    tool_call_id = runtime.tool_call_id
    
    if not seed_paper_ids:
        return Command(
            update={"messages": [ToolMessage(
                content="No seed paper IDs provided. Please specify which papers to use for citation chasing.",
                tool_call_id=tool_call_id
            )]}
        )
    
    top_k = min(max(1, top_k), 50)  # Clamp between 1 and 50
    
    try:
        s2_client = S2Client()
        
        # Fetch citations for all seed papers
        all_citations = {}  # corpus_id -> {paper_data, num_seeds_cited: int}
        
        for paper_id in seed_paper_ids:
            try:
                citations = await s2_client.get_paper_citations(
                    paper_id=paper_id,
                    fields=["paperId", "corpusId", "title", "abstract", "authors",
                           "year", "citationCount", "influentialCitationCount"]
                )
                
                if not citations:
                    continue
                
                for cite in citations:
                    corpus_id = cite.get("corpusId")
                    if not corpus_id:
                        continue
                    
                    if corpus_id not in all_citations:
                        all_citations[corpus_id] = {
                            "paper": cite,
                            "num_seeds_cited": 0
                        }
                    
                    all_citations[corpus_id]["num_seeds_cited"] += 1
                        
            except Exception as e:
                continue
        
        if not all_citations:
            return Command(
                update={"messages": [ToolMessage(
                    content=f"Could not find citations for the {len(seed_paper_ids)} seed papers.",
                    tool_call_id=tool_call_id
                )]}
            )
        
        # Score and sort candidates
        scored_candidates = []
        current_year = 2026
        
        for corpus_id, data in all_citations.items():
            paper = data["paper"]
            num_seeds = data["num_seeds_cited"]
            citation_count = paper.get("citationCount", 0)
            year = paper.get("year", 2000)
            
            # Recency bonus for papers from last 3 years
            recency_bonus = max(0, (year - (current_year - 3)) / 3.0) * 2.0
            
            # Simple scoring: prioritize papers citing multiple seeds, highly cited, and recent
            score = num_seeds * 10.0 + min(citation_count / 100, 10.0) + recency_bonus
            
            scored_candidates.append({"score": score, "paper": paper})
        
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = scored_candidates[:top_k]
        
        # Convert to S2Paper objects
        result_papers = []
        for candidate in top_candidates:
            paper_data = candidate["paper"]
            try:
                s2_paper = S2Paper(
                    paperId=paper_data.get("paperId", ""),
                    corpusId=paper_data.get("corpusId"),
                    title=paper_data.get("title"),
                    abstract=paper_data.get("abstract"),
                    authors=paper_data.get("authors"),
                    year=paper_data.get("year"),
                    citationCount=paper_data.get("citationCount"),
                    influentialCitationCount=paper_data.get("influentialCitationCount"),
                    url=paper_data.get("url", f"https://www.semanticscholar.org/paper/{paper_data.get('paperId', '')}")
                )
                result_papers.append(s2_paper)
            except Exception:
                continue
        
        return Command(
            update={
                "papers": result_papers,
                "messages": [ToolMessage(
                    content=f"Found {len(result_papers)} papers that cite the {len(seed_paper_ids)} seed papers.",
                    tool_call_id=tool_call_id
                )]
            }
        )

    except Exception as e:
        return Command(
            update={"messages": [ToolMessage(
                content=f"Error during forward snowball: {str(e)}",
                tool_call_id=tool_call_id
            )]}
        )

