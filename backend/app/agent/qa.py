import uuid
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph.ui import push_ui_message
from pydantic import BaseModel, Field
from app.core.config import settings
from app.tools.search import retrieve_evidence_from_selected_papers
from app.agent.utils import get_paper_abstract
import logging
from typing import Union
from app.agent.states import QAAgentState
from langgraph.graph import StateGraph, END, START
from app.agent.prompts import (
    QA_RETRIEVAL_SYSTEM,
    QA_RETRIEVAL_USER,
    QA_EVALUATION_SYSTEM,
    QA_EVALUATION_USER,
    QA_ANSWER_SYSTEM,
    QA_ANSWER_USER,
)

logger = logging.getLogger(__name__)

qa_model = init_chat_model(model=settings.QA_AGENT_MODEL_NAME)
qa_eval_model = init_chat_model(model=settings.QA_EVALUATION_MODEL_NAME)


async def qa_retrieve(state: QAAgentState) -> QAAgentState:
    iteration = state.get("qa_iteration", 0)
    tracking_id = state.get("qa_ui_tracking_id") or str(uuid.uuid4())
    label = "Retrieving evidence" if iteration == 0 else f"Refining retrieval (attempt {iteration + 1})"
    push_ui_message("qa_status", {"label": label, "status": "running"}, id=tracking_id)

    user_query = state.get("user_query", "")
    selected_paper_ids = state.get("selected_paper_ids", [])
    papers = state.get("papers", [])
    if not selected_paper_ids:
        logger.warning("No papers selected for QA!")
        push_ui_message("qa_status", {"label": label, "status": "completed", "description": "No papers selected"}, id=tracking_id)
        return {
            "evidences": [],
            "qa_ui_tracking_id": tracking_id,
            "messages": [AIMessage(content="No papers have been selected for Q&A. Please select papers first or use the paper finding mode.")]
        }

    abstracts = get_paper_abstract(papers=papers, selected_paper_ids=selected_paper_ids)
    evidences = state.get("evidences", [])
    evidences_text = "\n\n".join([
        f"Evidence {i+1}:\n{evidence.page_content}"
        for i, evidence in enumerate(evidences)
    ]) if evidences else "No evidence retrieved yet."
    limitation = state.get("limitation", "This is the first retrieval attempt, use search tools to retrieve more evidence.")

    abstracts_text = "\n".join([
        f"Paper {paper_id}:\n{abstract}"
        for paper_id, abstract in abstracts.items()
    ])

    retrieval_prompt = QA_RETRIEVAL_USER.format(user_query=user_query, abstracts_text=abstracts_text, evidences_text=evidences_text, limitation=limitation)

    tool_model = qa_model.bind_tools([retrieve_evidence_from_selected_papers], tool_choice="retrieve_evidence_from_selected_papers")
    tool_response = await tool_model.ainvoke([
        SystemMessage(content=QA_RETRIEVAL_SYSTEM),
        *state.get("messages", []),
        HumanMessage(content=retrieval_prompt)
    ])
    return {
        "messages": [tool_response],
        "qa_ui_tracking_id": tracking_id,
    }

async def qa_evaluate(state: QAAgentState) -> QAAgentState:
    tracking_id = state.get("qa_ui_tracking_id", "")
    push_ui_message("qa_status", {"label": "Evaluating evidence", "status": "running"}, id=tracking_id)

    user_query = state.get("user_query", "")
    abstracts = get_paper_abstract(state.get("papers", []), state.get("selected_paper_ids", []))
    abstracts_text = "\n".join([
        f"Paper {paper_id}:\n{abstract}"
        for paper_id, abstract in abstracts.items()
    ])
    evidences = state.get("evidences", [])
    evidences_text = "\n\n".join([
        f"Evidence {i+1}:\n{evidence.page_content}"
        for i, evidence in enumerate(evidences)
    ]) if evidences else "No evidence retrieved yet."

    limitation = state.get("limitation", "This is the first retrieval attempt.")
    evaluation_prompt = QA_EVALUATION_USER.format(user_query=user_query, abstracts_text=abstracts_text, evidences_text=evidences_text, limitation=limitation)

    class AskForMoreEvidence(BaseModel):
        limitation: str = Field(
            description="The limitation of the current retrieved evidence to help with the next retrieval attempt")

    class AnswerQuestion(BaseModel):
        reasoning: str = Field(
            description="The reasoning for why we should answer the user's question based on the retrieved evidence")

    class Evaluation(BaseModel):
        decision: Union[AskForMoreEvidence, AnswerQuestion] = Field(
            description="The decision for whether to retrieve more evidence or to answer the user's question")

    structured_model = qa_eval_model.with_structured_output(Evaluation)
    decision_response = await structured_model.ainvoke([
        SystemMessage(content=QA_EVALUATION_SYSTEM),
        HumanMessage(content=evaluation_prompt)
    ])

    if decision_response is None or decision_response.decision is None:
        push_ui_message("qa_status", {"label": "Evaluating evidence", "status": "completed", "description": "Proceeding to answer"}, id=tracking_id)
        return {
            "messages": [AIMessage(content="Evaluation failed, proceeding to answer.")],
            "limitation": "Structured output parsing failed.",
            "sufficient_evidence": False,
            "qa_iteration": state.get("qa_iteration", 0) + 1
        }

    if isinstance(decision_response.decision, AskForMoreEvidence):
        push_ui_message("qa_status", {"label": "Evaluating evidence", "status": "completed", "description": "Refining retrieval…"}, id=tracking_id)
        return {
            "messages": [AIMessage(content=decision_response.decision.limitation)],
            "limitation": decision_response.decision.limitation,
            "sufficient_evidence": False,
            "qa_iteration": state.get("qa_iteration", 0) + 1
        }
    elif isinstance(decision_response.decision, AnswerQuestion):
        push_ui_message("qa_status", {"label": "Evaluating evidence", "status": "completed", "description": "Evidence sufficient"}, id=tracking_id)
        return {
            "messages": [AIMessage(content=decision_response.decision.reasoning)],
            "sufficient_evidence": True,
            "qa_iteration": state.get("qa_iteration", 0) + 1
        }
    else:
        push_ui_message("qa_status", {"label": "Evaluating evidence", "status": "completed", "description": "Proceeding to answer"}, id=tracking_id)
        return {
            "messages": [AIMessage(content="Invalid decision")],
            "limitation": "Invalid decision",
            "sufficient_evidence": False,
            "qa_iteration": state.get("qa_iteration", 0) + 1
        }

async def qa_answer(state: QAAgentState) -> QAAgentState:
    """
    Generate a final answer based on retrieved segments and reasoning.
    Combines all evidence and provides a concise yet complete response.
    """
    tracking_id = state.get("qa_ui_tracking_id", "")
    push_ui_message("qa_status", {"label": "Generating answer", "status": "running"}, id=tracking_id)

    user_query = state.get("user_query", "")
    
    # Get all accumulated evidence
    evidences = state.get("evidences", [])
    evidences_text = "\n\n".join([
        f"Evidence {i}:\n{evidence.page_content}"
        for i, evidence in enumerate(evidences, 1)
    ])
    
    limitation = state.get("limitation", "No limitation")
    
    # Get paper abstracts for context
    abstracts = get_paper_abstract(state.get("papers", []), state.get("selected_paper_ids", []))
    abstracts_text = "\n".join([
        f"Paper {paper_id}:\n{abstract}"
        for paper_id, abstract in abstracts.items()
    ])
    answer_prompt = QA_ANSWER_USER.format(user_query=user_query, abstracts_text=abstracts_text, evidences_text=evidences_text, limitation=limitation)
    forced = state.get("qa_iteration", 0) >= 3 and not state.get("sufficient_evidence", False)
    if forced:
        answer_prompt += "\n\nNote: Maximum retrieval iterations reached. Evidence may be incomplete — acknowledge any gaps explicitly."
    response = await qa_model.ainvoke([
        SystemMessage(content=QA_ANSWER_SYSTEM),
        HumanMessage(content=answer_prompt)
    ])
    
    n = len(evidences)
    push_ui_message("qa_status", {
        "label": "Answer complete",
        "status": "completed",
        "description": f"Based on {n} evidence segment{'s' if n != 1 else ''}",
    }, id=tracking_id)
    return {
        "messages": [AIMessage(content=response.content)],
        "final_answer": response.content
    }

def should_answer(state: QAAgentState) -> str:
    if state.get("sufficient_evidence", False):
        return "answer"
    elif state.get("qa_iteration", 0) >= 3:
        return "answer"
    else:
        return "retrieve"


def build_qa_graph():
    """
    Build the QA subgraph for paper question answering.

    Flow:
    1. qa_prepare: Prepare state for QA (select papers if needed)
    1. qa_retrieve: Search within selected papers for relevant segments
    2. qa_assess_quality: Check if evidence is sufficient
    3. If insufficient: qa_refine_retrieval (max 1 refinement)
    4. qa_answer: Generate grounded answer
    """
    

    qa_builder = StateGraph(QAAgentState)

    # Add nodes
    qa_builder.add_node("qa_retrieve", qa_retrieve)
    qa_builder.add_node("tools", ToolNode([retrieve_evidence_from_selected_papers]))
    qa_builder.add_node("qa_evaluate", qa_evaluate)
    qa_builder.add_node("qa_answer", qa_answer)

    # Add edges
    qa_builder.add_edge(START, "qa_retrieve")
    qa_builder.add_edge("qa_retrieve", "tools")
    qa_builder.add_edge("tools", "qa_evaluate")
    qa_builder.add_conditional_edges("qa_evaluate", should_answer, {
        "answer": "qa_answer",
        "retrieve": "qa_retrieve",
    })
    qa_builder.add_edge("qa_answer", END)

    return qa_builder.compile()


# Export the compiled QA graph
qa_graph = build_qa_graph()
