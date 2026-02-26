from typing import Annotated, List, Dict, Any, Optional, Literal, Tuple, Sequence
from langgraph.graph.ui import AnyUIMessage, ui_message_reducer
from langchain.messages import AIMessage
import operator
from langchain.agents import AgentState
from app.core.schema import S2Paper, Step
from langchain_core.documents import Document
from app.agent.utils import merge_evidences
from langgraph.graph.message import MessagesState

class SupervisorState(AgentState):
    is_clear: Optional[bool]
    papers: List[S2Paper]
    ui: Annotated[Sequence[AnyUIMessage], ui_message_reducer]
    plan_steps: List[Literal["find_papers", "retrieve_and_answer_question"]]
    selected_paper_ids: List[str]
    ui_tracking_message: AIMessage
    ui_tracking_id: str
    steps: List[Step]


class PaperFinderState(MessagesState):
    search_task: Optional[str]   # natural-language goal for the planner + search agent
    rerank_query: Optional[str]  # keyword-optimized query used only for reranking
    plan_steps: List[str]
    completed_steps: Annotated[List[Tuple[str, str]], operator.add]
    plan_reasoning: str
    papers: List[Dict[str, Any]]
    iter: int
    goal_achieved: bool
    ui_tracking_id: Optional[str]  # stable ID for in-place UIMessage updates

class QAAgentState(MessagesState):
    evidences: Annotated[List[Document], merge_evidences]
    limitation: str
    qa_iteration: int
    selected_paper_ids: List[str]
    unindexed_paper_ids: List[str]  # paper IDs with no vectors in Qdrant (failed/skipped ingestion)
    sufficient_evidence: bool
    user_query: str
    papers: List[S2Paper]
    final_answer: str
    qa_ui_tracking_id: Optional[str]  # stable ID for in-place UIMessage updates