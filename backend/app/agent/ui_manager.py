from typing import Literal, List
from langchain_core.messages import AIMessage
from langgraph.graph.ui import push_ui_message
import uuid
from app.core.schema import Step, StepName, StepStatus
from app.agent.states import SupervisorState

step_update_map = {}
step_update_map[StepStatus.RUNNING] = {
    "status": "pending",
    "description": f"Checking if query is clear...",
}

def get_template_step(step_name: StepName, step_status: StepStatus) -> Step:
    return {
        "id": step_name,
        "label": step_name,
        "status": step_status,
        "description": f"",
    }


def get_update_query_evaluation_step(step_name: StepName, step_status: StepStatus, decision: str = "clear", **kwargs):
    template_step = get_template_step(step_name, step_status)
    if step_status == StepStatus.RUNNING:
        template_step["description"] = "Evaluating query..."
    elif step_status == StepStatus.COMPLETED:
        if decision == "clear":
            template_step["description"] = "Query is valid — proceeding."
        elif decision == "needs_clarification":
            template_step["description"] = "Query too vague — asked user for clarification."
        elif decision == "unselected_paper":
            template_step["description"] = "Query targets a paper not yet selected — reminded user to select it."
        elif decision == "irrelevant":
            template_step["description"] = "Query is off-topic — explained what Corvus can do."
        elif decision == "inappropriate":
            template_step["description"] = "Query is inappropriate — declined."
        else:
            template_step["description"] = f"Evaluation result: {decision}."
    else:
        raise ValueError(f"Invalid step status: {step_status}")
    return template_step
    

def get_update_query_optimization_step(step_name: StepName, step_status: StepStatus, optimized_query: str | None = None, *args: str):
    template_step = get_template_step(step_name, step_status)
    if step_status == StepStatus.RUNNING:
        template_step["description"] = f"Optimizing query..."
    elif step_status == StepStatus.COMPLETED:
        template_step["description"] = f"The optimized query is: {optimized_query}"
    else:
        raise ValueError(f"Invalid step status: {step_status}")
    return template_step


def get_update_plan_step(step_name: StepName, step_status: StepStatus, plan_choice: Literal["find_then_qa", "find_only", "qa_only"] | None = None, *args: tuple[str, ...]):
    template_step = get_template_step(step_name, step_status)
    if step_status == StepStatus.RUNNING:
        template_step["description"] = f"Supervisor agent planning the next steps..."
    elif step_status == StepStatus.COMPLETED:
        description = ""
        if plan_choice == "find_then_qa":
            description = "Based on the user's intent, first retrieve the papers then answer the user's question"
        elif plan_choice == "find_only":
            description = "Based on the user's intent, only retrieve the papers"
        elif plan_choice == "qa_only":
            description = "Based on the user's intent, only answer the user's question"
        else:
            raise ValueError(f"Invalid plan choice: {plan_choice}")
        template_step["description"] = f"{description}"
    else:
        raise ValueError(f"Invalid step status: {step_status}")
    return template_step    

def get_update_find_papers_step(step_name: StepName, step_status: StepStatus, num_papers: int | None = None, *args: tuple[str, ...]):
    template_step = get_template_step(step_name, step_status)
    if step_status == StepStatus.RUNNING:
        template_step["description"] = f"Delegating to find papers agent, this might take a while..."
    elif step_status == StepStatus.COMPLETED:
        template_step["description"] = f"Found {num_papers} papers."
    else:
        raise ValueError(f"Invalid step status: {step_status}")
    return template_step

def get_update_replanning_step(step_name: StepName, step_status: StepStatus, *args: tuple[str, ...]):
    template_step = get_template_step(step_name, step_status)
    if step_status == StepStatus.RUNNING:
        template_step["description"] = "Re-evaluating plan based on new context..."
    elif step_status == StepStatus.COMPLETED:
        template_step["description"] = "Plan updated."
    else:
        raise ValueError(f"Invalid step status: {step_status}")
    return template_step

def get_update_retrieve_and_answer_question_step(step_name: StepName, step_status: StepStatus, *args: tuple[str, ...]):
    template_step = get_template_step(step_name, step_status)
    if step_status == StepStatus.RUNNING:
        template_step["description"] = f"Retrieving evidence to answer the user's question, this might take a while..."
    elif step_status == StepStatus.COMPLETED:
        template_step["description"] = f"Finished retrieving evidence to answer the user's question."
    else:
        raise ValueError(f"Invalid step status: {step_status}")
    return template_step

def get_update_step(step_name: StepName, step_status: StepStatus, *args, **kwargs):
    if step_name == StepName.QUERY_EVALUATION:
        return get_update_query_evaluation_step(step_name, step_status, *args, **kwargs)
    elif step_name == StepName.QUERY_OPTIMIZATION:
        return get_update_query_optimization_step(step_name, step_status, *args)
    elif step_name == StepName.PLAN:
        return get_update_plan_step(step_name, step_status, *args)
    elif step_name == StepName.FIND_PAPERS:
        return get_update_find_papers_step(step_name, step_status, *args)
    elif step_name == StepName.RETRIEVE_AND_ANSWER_QUESTION:
        return get_update_retrieve_and_answer_question_step(step_name, step_status, *args)
    elif step_name == StepName.REPLANNING:
        return get_update_replanning_step(step_name, step_status, *args)
    else:
        raise ValueError(f"Invalid step name: {step_name}")

class UIManager:
    def __init__(self, steps: List[Step], ui_tracking_message: AIMessage, ui_tracking_id: str):
        self.steps = steps
        self.ui_tracking_message = ui_tracking_message
        self.ui_tracking_id = ui_tracking_id

    @classmethod
    def from_state(cls, state: SupervisorState) -> "UIManager":
        return cls(state.get("steps", []), state.get("ui_tracking_message", AIMessage(id=str(uuid.uuid4()), content="")), state.get("ui_tracking_id", str(uuid.uuid4())))

    def update_ui(self, step_name: StepName, step_status: StepStatus, *args, **kwargs) -> List[Step]:
        new_step = get_update_step(step_name, step_status, *args, **kwargs)
        for i, step in enumerate(self.steps):
            if step["id"] == step_name:
                self.steps[i] = new_step
                break
        else:
            self.steps.append(new_step)
        push_ui_message(
            "steps",
            {
                "steps": self.steps
            },
            message=self.ui_tracking_message,
            id=self.ui_tracking_id
        )
        return self.steps