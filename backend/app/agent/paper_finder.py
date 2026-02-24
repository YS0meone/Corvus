import asyncio
import uuid
import logging
from langgraph.graph.ui import push_ui_message
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from app.agent.states import PaperFinderState
from langgraph.prebuilt import ToolNode
from app.tools.search import s2_search_papers, tavily_research_overview, forward_snowball, backward_snowball
from app.core.config import settings
from app.agent.utils import get_paper_info_text
from rerankers import Reranker, Document
from app.core.schema import S2Paper
from pydantic import BaseModel, Field
from typing import List, Tuple, Annotated
from langchain.agents import AgentState
from langgraph.prebuilt import tools_condition

logger = logging.getLogger(__name__)


if not settings.COHERE_API_KEY:
    logger.warning("COHERE_API_KEY not set. Reranking will be skipped.")
    ranker = None
else:
    try:
        ranker = Reranker("cohere", api_key=settings.COHERE_API_KEY)
    except Exception as e:
        logger.error("Failed to initialize Cohere reranker: %s", e)
        ranker = None

tools = [tavily_research_overview, s2_search_papers, forward_snowball, backward_snowball]

MAX_ITER = 3
MAX_PAPER_LIST_LENGTH = 35

model = init_chat_model(model=settings.PF_AGENT_MODEL_NAME)
search_agent_model = model.bind_tools(tools)

async def planner(state: PaperFinderState):
    tracking_id = str(uuid.uuid4())
    push_ui_message("finder_status", {
        "label": "Planning search strategy",
        "status": "running",
    }, id=tracking_id)

    system_prompt = """
    You are a senior researcher. The goal is to create a plan for your research assistant to find the most relevant papers to the user query.
    You are provided with a user query and potentially a list of papers known to the research assistant.
    You need to plan the best way to find the most relevant papers to the user query.
    
    Your assistant has access to multiple search methods:
    1. General web search: Understand context and find famous/seminal papers
    2. Academic database search: Find papers with keyword queries and filters (year, venue, citations, etc.)
    3. Citation chasing:
       - Forward snowball: Find papers that CITE seed papers (recent work building on them)
       - Backward snowball: Find papers that seed papers CITE (their foundations/references)

    Guidelines:
    - Every step must be a concrete search action (web search, database search, or citation chasing).
      Do NOT include steps like "review results", "filter papers", "select seed papers", or "compare papers" —
      result filtering and ranking is handled automatically after each step.
    - Think like a real researcher who would give different plans based on different scenarios:
        Examples:
        - General topic: Start with web search for context, then academic database search
        - Specific paper: Use academic database with title/author filters
    - Never use citation chasing (snowball) unless the user explicitly asks for related/citing/cited papers.
    - The granularity of each step should be adequate for the assistant to finish within one execution
    - Keep each step concise and to the point
    - Try to minimize the steps of plan as much as possible
    """

    paper_info_text = get_paper_info_text(state.get("papers", []), include_abstract=False)

    user_prompt = f"""
    Task: {state['search_task']}
    Papers information:
    {paper_info_text}
    """

    class Plan(BaseModel):
        plan_reasoning: str = Field(description="The reasoning for the plan you generated")
        plan_steps: List[str] = Field(description="The steps of the plan")

    structured_model = model.with_structured_output(Plan)

    try:
        response = await structured_model.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        n = len(response.plan_steps)
        push_ui_message("finder_status", {
            "label": "Planning search strategy",
            "status": "completed",
            "description": f"{n} step{'s' if n != 1 else ''} planned",
        }, id=tracking_id)
        return {"plan_steps": response.plan_steps, "plan_reasoning": response.plan_reasoning, "ui_tracking_id": tracking_id}
    except Exception as e:
        logger.error("Error in planner: %s", e)
        push_ui_message("finder_status", {
            "label": "Planning search strategy",
            "status": "completed",
            "description": "Using default plan due to planning error",
        }, id=tracking_id)
        return {
            "plan_steps": [
                "Use web search to understand the research topic",
                "Search academic database for relevant papers",
            ],
            "plan_reasoning": "Using default plan due to planning error",
            "ui_tracking_id": tracking_id,
        }

def completed_steps_formatter(completed_steps: List[Tuple[str, str]]) -> str:
    if not completed_steps:
        return "None"
    return "\n\n".join([f"Task: {task}\nResult: {result}" for task, result in completed_steps])

async def replan_agent(state: PaperFinderState):
    tracking_id = state.get("ui_tracking_id", "")
    push_ui_message("finder_status", {
        "label": "Evaluating results",
        "status": "running",
    }, id=tracking_id)

    system_prompt = """
    You are a senior researcher. The goal is to update a plan for your research assistant to find the most relevant papers to the user query.
    You are provided with a user query, the retrieved papers, the current plan to retrieve the papers and the steps your assistant has completed.
    You need to first determine if goal is achieved or not. If the goal is achieved, you can stop and mark the goal as achieved.
    If the goal is not achieved, you need to update the plan to find the most relevant papers to the user query and return the new plan.

    Your assistant has access to multiple search methods:
    1. General web search: Understand context and find famous/seminal papers
    2. Academic database search: Find papers with keyword queries and filters
    3. Citation chasing:
       - Forward snowball: Find papers that CITE seed papers (recent work)
       - Backward snowball: Find papers that seed papers CITE (foundations/references)

    !! CRITICAL — what counts as a retrieved paper !!
    Papers mentioned by name in a web search summary are NOT retrieved. They are just text.
    A paper is only retrieved when it appears in the "Papers information" list below.
    That list is the ONLY ground truth for retrieval success.
    Do NOT mark the goal as achieved because paper titles were mentioned in a web search step —
    those papers still need to be fetched from the academic database.

    When deciding if the goal is achieved:
    - Specific paper/author lookup: achieved when ≥1 matching paper appears in the Papers list.
      A web search that merely names the paper does NOT count.
    - Broad topic search: achieved when ≥7 relevant papers appear in the Papers list AND
      the completed steps have covered the core search approaches for the task.
    - Err on the side of marking done once papers are actually in the list — the user can always ask for more.

    If the goal is NOT yet achieved, update the plan. Guidelines for the new plan:
    - Every step must be a concrete search action (web search, database search, or citation chasing).
      Do NOT include steps like "review results", "filter papers", "select seed papers", or "compare papers" —
      result filtering and ranking is handled automatically after each step.
    - If a prior web search step named specific papers/authors, make the next database step explicit:
      e.g. "Search Semantic Scholar for papers by <author> to retrieve them into the paper list."
      This makes clear to your assistant that retrieval — not just discovery — is the goal.
    - Never use citation chasing (snowball) unless the user explicitly asks for related/citing/cited papers.
    - The completed steps should not be included in the new plan.
    - Keep each step concise and minimize the number of remaining steps.
    """

    paper_info_text = get_paper_info_text(state.get("papers", []), include_abstract=False)

    user_prompt = f"""
    Task: {state['search_task']}
    Current Plan: {state.get("plan_steps", [])}
    Completed Steps:
    {completed_steps_formatter(state.get("completed_steps", []))}
    Papers information:
    {paper_info_text}
    """

    class Replan(BaseModel):
        goal_achieved: bool = Field(description="Whether the goal is achieved. If True, plan_steps is ignored.")
        plan_steps: List[str] = Field(default_factory=list, description="The remaining steps to execute. Empty if goal is achieved.")
        plan_reasoning: str = Field(default="", description="The reasoning for the decision.")

    structured_model = model.with_structured_output(Replan)

    try:
        response = await structured_model.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        if response.goal_achieved:
            push_ui_message("finder_status", {
                "label": "Search complete",
                "status": "completed",
                "description": f"Found {len(state.get('papers', []))} papers",
            }, id=tracking_id)
            return {"goal_achieved": True}
        else:
            n = len(response.plan_steps)
            push_ui_message("finder_status", {
                "label": "Evaluating results",
                "status": "completed",
                "description": f"{n} step{'s' if n != 1 else ''} remaining",
            }, id=tracking_id)
            return {"goal_achieved": False, "plan_steps": response.plan_steps, "plan_reasoning": response.plan_reasoning}
    except Exception as e:
        logger.error("Error in replan_agent: %s", e)
        # Fallback to existing plan
        current_plan = state.get("plan_steps", [])
        push_ui_message("finder_status", {
            "label": "Evaluating results",
            "status": "completed",
            "description": "Continuing with adjusted plan due to replanning error",
        }, id=tracking_id)
        return {
            "goal_achieved": False,
            "plan_steps": current_plan[1:] if len(current_plan) > 1 else ["Search for more relevant papers"],
            "plan_reasoning": "Continuing with adjusted plan due to replanning error",
        }

def flexible_reducer(current: list, update: list) -> list:
    seen = {p.paperId for p in current}
    return current + [p for p in update if p.paperId not in seen]

class SearchAgentState(AgentState):
    search_task: str
    rerank_query: str
    papers: Annotated[List[S2Paper], flexible_reducer]
    plan_steps: List[str]

async def search_agent_node(state: SearchAgentState):
    paper_info_text = get_paper_info_text(state.get("papers", []), include_abstract=False)
    plan_steps = state.get("plan_steps", [])
    remaining_steps = plan_steps[1:] if len(plan_steps) > 1 else []
    remaining_steps_text = (
        "\n".join(f"  {i+2}. {s}" for i, s in enumerate(remaining_steps))
        if remaining_steps else "  (none — this is the last step)"
    )
    search_query_prompt = f"""
    You are a senior research assistant who helps finding academic papers based on a user query.
    You are executing ONE step of a multi-step search plan. Your mentor will handle subsequent steps.

    Your goal is to utilize the provided tools to finish the current step only.

    You have access to multiple search methods:
    1. General web search (tavily_research_overview): Use when your goal is related to using general web search to identify famous papers or understand context etc.

    2. Academic database search (s2_search_papers): Search Semantic Scholar's database of 200M+ papers.
       Use keyword queries, filters by year, venue, citation count, etc. to find relevant papers. Pick this tool when the goal prompts you to search academic database.

    3. Citation chasing tools:
       - forward_snowball: Find papers that CITE your seed papers (recent work building on them)
       - backward_snowball: Find papers that your seed papers CITE (their foundations/references)
       Use these when the goal explicitly asks you to.

    !! IMPORTANT — what counts as a retrieved paper !!
    A paper only enters the paper list when it is returned by a tool call (s2_search_papers,
    forward_snowball, or backward_snowball). Papers mentioned by name in a web search summary
    (tavily_research_overview) are NOT in the list — they are just text. Do NOT treat paper
    titles or names that appear in the "Completed Steps" section below as already retrieved.
    The ONLY ground truth for what has been retrieved is the "Current papers in your list"
    section. If that list is empty or missing papers you need, you must call a search tool.

    Exit condition for database/citation search steps:
    - You are done only when you have actually called s2_search_papers (or a snowball tool)
      and received results for the current goal.
    - Do NOT stop after zero or one trivial tool calls just because paper titles were mentioned
      in a prior web search step.

    Strict limits — you are executing ONE step of a larger plan:
    - Do NOT use citation chasing (snowball) unless the current goal explicitly asks for it.
    - Do NOT repeat a search with minor keyword variations — pick the best query and move on.
    - Do NOT pre-emptively do work that belongs to a later step (listed below).

    Upcoming steps you must NOT do yet:
{remaining_steps_text}

    Current papers in your list (these are the ONLY papers that have been retrieved so far):
    {paper_info_text}

    Review the tool calls you have already made in this session to avoid redundant searches.
    Once the current goal is complete, stop and provide a concise summary of what you found
    and how many papers were added to the list.
    """

    response = await search_agent_model.ainvoke([
        SystemMessage(content=search_query_prompt),
        *state.get("messages", [])
    ])
    return {"messages": [response]}

search_tool_node = ToolNode(tools)

async def rerank_papers(papers: List[S2Paper], query: str) -> List[S2Paper]:
    if not papers:
        return []

    if not query or not query.strip() or ranker is None:
        return papers[:MAX_PAPER_LIST_LENGTH]

    try:
        docs = [
            Document(
                text=f"Title: {p.title or 'No title'}\nAbstract: {p.abstract or 'No abstract'}\nAuthors: {p.authors or []}",
                doc_id=str(p.paperId),
                metadata=p.model_dump(),
            )
            for p in papers
        ]
        reranked = await asyncio.to_thread(ranker.rank, query=query, docs=docs)
        return [S2Paper.model_validate(m.document.metadata) for m in reranked.top_k(k=MAX_PAPER_LIST_LENGTH)]
    except Exception as e:
        logger.error("Reranking failed: %s", e)
        return papers[:MAX_PAPER_LIST_LENGTH]

search_graph = StateGraph[SearchAgentState, None, SearchAgentState, SearchAgentState](SearchAgentState)
search_graph.add_node("search_agent", search_agent_node)
search_graph.add_node("search_tool", search_tool_node)
search_graph.add_edge(START, "search_agent")
search_graph.add_conditional_edges("search_agent", tools_condition, {
        "tools": "search_tool",
        "__end__": END
    })
search_graph.add_edge("search_tool", "search_agent")
search_graph = search_graph.compile()

async def executor(state: PaperFinderState):
    iter = state.get("iter", 0)
    current_goal = state.get("plan_steps", [])[0]
    tracking_id = state.get("ui_tracking_id", "")
    total = iter + len(state.get("plan_steps", []))
    push_ui_message("finder_status", {
        "label": f"Searching (step {iter + 1} of {total})",
        "status": "running",
        "description": current_goal,
    }, id=tracking_id)

    user_prompt = f"""
    Task: {state.get("search_task", "")}
    Current Goal: {current_goal}
    Completed Steps:
    {completed_steps_formatter(state.get("completed_steps", []))}
    """
    search_agent_state = {
        "search_task": state.get("search_task", ""),
        "rerank_query": state.get("rerank_query", ""),
        "plan_steps": state.get("plan_steps", []),
        "papers": state.get("papers", []),
        "messages": [HumanMessage(content=user_prompt)]
    }

    response = await search_graph.ainvoke(search_agent_state)

    # Rerank once after the full plan step completes, not after every tool call
    papers = await rerank_papers(
        response.get("papers", state.get("papers", [])),
        state.get("rerank_query", ""),
    )

    if isinstance(response["messages"][-1].content, list):
        content = " ".join([item["text"] for item in response["messages"][-1].content])
    elif isinstance(response["messages"][-1].content, str):
        content = response["messages"][-1].content
    else:
        content = str(response["messages"][-1].content)
    step_summary = (current_goal, content)
    push_ui_message("finder_status", {
        "label": f"Searching (step {iter + 1} of {total})",
        "status": "completed",
        "description": f"{len(papers)} papers found so far",
    }, id=tracking_id)
    return {"papers": papers, "completed_steps": [step_summary], "iter": iter + 1}


def should_reply(state: PaperFinderState):
    goal_achieved = state.get("goal_achieved", False)
    iter = state.get("iter", 0)
    return END if goal_achieved or iter >= MAX_ITER else "executor"

paper_finder_builder = StateGraph(PaperFinderState)
paper_finder_builder.add_node("planner", planner)
paper_finder_builder.add_node("replan_agent", replan_agent)
paper_finder_builder.add_node("executor", executor)

paper_finder_builder.add_edge(START, "planner")
paper_finder_builder.add_edge("planner", "executor")
paper_finder_builder.add_edge("executor", "replan_agent")
paper_finder_builder.add_conditional_edges("replan_agent", should_reply)

paper_finder = paper_finder_builder.compile()
