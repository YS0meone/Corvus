import asyncio
import logging
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from app.tools.search import s2_search_papers, tavily_research_overview, forward_snowball, backward_snowball
from app.core.config import settings
from app.agent.utils import get_paper_info_text
from rerankers import Reranker, Document
from app.core.schema import S2Paper
from typing import List, Annotated, Union
from langchain.agents import AgentState
from langgraph.prebuilt import tools_condition

logger = logging.getLogger(__name__)


MAX_PAPER_LIST_LENGTH = 20

model = init_chat_model(model=settings.PF_AGENT_MODEL_NAME)

tools = [tavily_research_overview, s2_search_papers, forward_snowball, backward_snowball]
search_agent_model = model.bind_tools(tools)

# Initialize Cohere reranker
if not settings.COHERE_API_KEY:
    logger.warning("COHERE_API_KEY not set. Reranking will be skipped.")
    ranker = None
else:
    try:
        ranker = Reranker("cohere", api_key=settings.COHERE_API_KEY)
        logger.info("Cohere reranker initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize Cohere reranker: %s", e)
        ranker = None


class Replace:
    def __init__(self, value: list):
        self.value = value

def flexible_reducer(current: list, update: Union[list, "Replace"]) -> list:
    if isinstance(update, Replace):
        return update.value
    seen = {p.paperId for p in current}
    return current + [p for p in update if p.paperId not in seen]

class SearchAgentState(AgentState):
    search_task: str
    rerank_query: str
    papers: Annotated[List[S2Paper], flexible_reducer]
    iter: int



async def search_agent_node(state: SearchAgentState):
    paper_info_text = get_paper_info_text(state.get("papers", []))
    search_query_prompt = f"""
    You are a senior research assistant who helps finding academic papers based on a user query.

    Your goal is to utilize the provided tools to help user find the most relevant papers to the user query.

    You have access to multiple search methods:
    1. General web search (tavily_research_overview): Use this when the research topic is general or unfamiliar.
       This helps you understand the research landscape and identify famous/seminal papers you shouldn't miss.

    2. Academic database search (s2_search_papers): Search Semantic Scholar's database of 200M+ papers.
       Use keyword queries, filters by year, venue, citation count, etc. to find relevant papers.

    3. Citation chasing tools:
       - forward_snowball: Find papers that your seed papers CITE (their references/foundations)
       - backward_snowball: Find papers that CITE your seed papers (recent work building on them)
       Use these when you've found good papers and want to explore their citation network.

    Strategy tips:
    - If the user query is about a specific paper, use the academic database search with the title/author filters to find the paper and quickly finish the task.
    - If the user query is more general, you should follow the following strategy:
        - Start with web search if topic is unfamiliar to get context
        - Use academic database for targeted searches with filters
        - Use citation chasing to expand from good seed papers you've found

    Current papers in your list:
    {paper_info_text}

    Reflect on past actions and completed steps to decide what to do next.
    If you have sufficient results, stop and provide a concise summary of what you found.
    """

    response = await search_agent_model.ainvoke([
        SystemMessage(content=search_query_prompt),
        *state.get("messages", [])
    ])
    return {"messages": [response]}

search_tool_node = ToolNode(tools)

async def rerank_node(state: SearchAgentState):
    papers = state.get("papers", [])
    iter = state.get("iter", 0) + 1

    if not papers:
        return {"iter": iter}

    user_query = state.get("rerank_query", "")

    if not user_query or not user_query.strip():
        logger.warning("Skipping rerank: no query provided")
        return {"papers": Replace(papers[:MAX_PAPER_LIST_LENGTH]), "iter": iter}

    if ranker is None:
        logger.warning("Skipping rerank: COHERE_API_KEY not set")
        return {"papers": Replace(papers[:MAX_PAPER_LIST_LENGTH]), "iter": iter}

    try:
        docs = [
            Document(
                text=f"Title: {p.title or 'No title'}\nAbstract: {p.abstract or 'No abstract'}\nAuthors: {p.authors or []}",
                doc_id=str(p.paperId),
                metadata=p.model_dump(),
            )
            for p in papers
        ]
        logger.debug("Reranking %d papers with query: %s...", len(docs), user_query[:50])
        reranked = await asyncio.to_thread(ranker.rank, query=user_query, docs=docs)
        final_papers = [S2Paper.model_validate(m.document.metadata) for m in reranked.top_k(k=MAX_PAPER_LIST_LENGTH)]
        logger.info("Reranking successful: %d papers", len(final_papers))
    except Exception as e:
        logger.error("Reranking failed: %s: %s", type(e).__name__, e)
        final_papers = papers[:MAX_PAPER_LIST_LENGTH]

    return {"papers": Replace(final_papers), "iter": iter}

def my_tools_condition(state: SearchAgentState):
    if state.get("iter", 0) > 3:
        return "__end__"
    return tools_condition(state)

paper_finder_fast_graph = StateGraph(SearchAgentState)
paper_finder_fast_graph.add_node("search_agent", search_agent_node)
paper_finder_fast_graph.add_node("search_tool", search_tool_node)
paper_finder_fast_graph.add_node("rerank", rerank_node)

paper_finder_fast_graph.add_edge(START, "search_agent")
paper_finder_fast_graph.add_conditional_edges("search_agent", my_tools_condition, {"tools": "search_tool", "__end__": END})
paper_finder_fast_graph.add_edge("search_tool", "rerank")
paper_finder_fast_graph.add_edge("rerank", "search_agent")
paper_finder_fast_graph = paper_finder_fast_graph.compile()
