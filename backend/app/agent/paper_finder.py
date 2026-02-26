import asyncio
import uuid
import logging
from datetime import date
from langgraph.graph.ui import push_ui_message
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from app.agent.states import PaperFinderState
from langgraph.prebuilt import ToolNode
from app.tools.search import s2_search_papers, tavily_research_overview, forward_snowball, backward_snowball
from app.core.config import settings
from app.agent.utils import get_paper_info_text
from app.agent.prompts import (
    PF_PLANNER_SYSTEM,
    PF_PLANNER_USER,
    PF_REPLAN_SYSTEM,
    PF_REPLAN_USER,
    PF_SEARCH_AGENT_SYSTEM,
    PF_EXECUTOR_USER,
)
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

    today = date.today()
    paper_info_text = get_paper_info_text(state.get("papers", []), include_abstract=False)
    system_prompt = PF_PLANNER_SYSTEM.format(today=today.strftime("%B %d, %Y"))
    user_prompt = PF_PLANNER_USER.format(search_task=state["search_task"], paper_info_text=paper_info_text)

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

    paper_info_text = get_paper_info_text(state.get("papers", []), include_abstract=False)
    system_prompt = PF_REPLAN_SYSTEM
    user_prompt = PF_REPLAN_USER.format(
        search_task=state["search_task"],
        plan_steps=state.get("plan_steps", []),
        completed_steps=completed_steps_formatter(state.get("completed_steps", [])),
        paper_info_text=paper_info_text,
    )

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
    today = date.today()
    search_query_prompt = PF_SEARCH_AGENT_SYSTEM.format(
        today=today.strftime("%B %d, %Y"),
        remaining_steps_text=remaining_steps_text,
        paper_info_text=paper_info_text,
    )

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

    user_prompt = PF_EXECUTOR_USER.format(
        search_task=state.get("search_task", ""),
        current_goal=current_goal,
        completed_steps=completed_steps_formatter(state.get("completed_steps", [])),
    )
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
