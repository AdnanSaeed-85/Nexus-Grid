from agent_state import (SupervisorState, QueryAnalyzerOutput, ChildTask, ChildTaskDraft, ReviewOutput, AttemptEntry, Attempts, WorkerState)
from prompt import question_checker, supervisor_agent_prompt, sub_agent_prompt, supervisor_review_prompt, report_generator
from agent_tool import tavily_search
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send
from pydantic import BaseModel
from typing import List
import json
from uuid import uuid4

supervisor_agent_id = str(uuid4())

load_dotenv()

# ── structured output schema for supervisor LLM only ──
class SupervisorLLMOutput(BaseModel):
    child_tasks: List[ChildTaskDraft]

llm = ChatOpenAI(model='gpt-4o-mini')
supervisor_llm = llm.with_structured_output(SupervisorLLMOutput)
analyzer_llm = llm.with_structured_output(QueryAnalyzerOutput)
worker_llm = llm.bind_tools([tavily_search])
supervisor_review_llm = llm.with_structured_output(ReviewOutput)

# ──────────────────────────────────────────────
# NODES
# ──────────────────────────────────────────────

def query_analyzer(state: SupervisorState) -> dict:
    output = analyzer_llm.invoke(question_checker(state["parent_question"]))
    agent_type = "multi_agent" if (output.is_question and output.is_research_able) else "simple_agent"
    return {
        "is_question": output.is_question,
        "is_research_able": output.is_research_able,
        "agent_type": agent_type}


def simple_agent(state: SupervisorState) -> dict:
    response = llm.invoke(state["parent_question"])
    return {
        "final_report": response.content,
        "state": "done"}



def supervisor_agent(state: SupervisorState) -> dict:
    # ── review mode ──
    review_queue = state.get("review_queue", [])
    reviewed_count = state.get("reviewed_count", 0)
    if len(review_queue) > reviewed_count:
        approved = []
        rejected = []
        child_tasks = list(state["child_tasks"])
        tasks_by_id = {task.child_task_id: task for task in child_tasks}

        for item in review_queue[reviewed_count:]:
            parsed = json.loads(item)

            review_output = supervisor_review_llm.invoke(
                supervisor_review_prompt(
                    task=parsed["task"],
                    context=parsed["context"],
                    success_criteria=parsed["success_criteria"],
                    output=parsed["output"],
                    attempt_number=parsed["attempt_number"]
                )
            )

            task = tasks_by_id[parsed["child_task_id"]]
            attempt = AttemptEntry(
                attempt_number=parsed["attempt_number"],
                supervisor_feedback=(
                    review_output.feedback
                    if review_output.result == "rejected"
                    else None
                ),
                result=review_output.result,
            )
            updated_task = task.model_copy(
                update={
                    "status": review_output.result,
                    "attempts": Attempts(
                        count=parsed["attempt_number"],
                        history=[*task.attempts.history, attempt],
                    ),
                    "final_output": json.dumps(parsed["output"]),
                }
            )
            tasks_by_id[task.child_task_id] = updated_task

            if review_output.result == "approved":
                approved.append(parsed["output"]["findings"])
            else:
                rejected.append(json.dumps({
                    "task": parsed["task"],
                    "context": parsed["context"],
                    "success_criteria": parsed["success_criteria"],
                    "attempt_number": parsed["attempt_number"],
                    "feedback": review_output.feedback
                }))

        approved_outputs = [*state.get("approved_outputs", []), *approved]
        retryable_tasks = [
            task for task in tasks_by_id.values()
            if task.status == "rejected" and task.attempts.count < 5
        ]

        result = {
            "approved_outputs": approved,
            "rejected_outputs": rejected,
            "child_tasks": list(tasks_by_id.values()),
            "reviewed_count": len(review_queue),
            "state": "review"
        }

        if not retryable_tasks:
            heading = llm.invoke(
                "Return only a short 2-3 word heading for these research findings:\n"
                + "\n".join(approved_outputs)
            )
            result["final_report"] = (
                f"{heading.content}\n\n"
                f"{state['parent_question']}\n\n"
                f"{chr(10).join(approved_outputs)}"
            )
            result["state"] = "done"

        return result

    # ── decompose mode ──
    if state.get("child_tasks"):
        return {"state": "assign"}

    output = supervisor_llm.invoke(supervisor_agent_prompt(state["parent_question"]))
    child_tasks = [
        ChildTask(child_task_id=str(uuid4()), sub_agent_id=str(uuid4()),
                  task=task.task,
                  context=task.context,
                  success_criteria=task.success_criteria) for task in output.child_tasks]

    return {
        "supervisor_id": supervisor_agent_id,
        "child_tasks": child_tasks,
        "n_agents": len(child_tasks),
        "state": "assign"
    }


def worker(state: WorkerState) -> dict:
    child_task = state["child_task"]
    last_feedback = None
    if child_task.attempts.history:
        last_entry = child_task.attempts.history[-1]
        last_feedback = last_entry.supervisor_feedback

    prompt = sub_agent_prompt(
        task=child_task.task,
        context=child_task.context,
        success_criteria=child_task.success_criteria.model_dump(),
        attempt_number=child_task.attempts.count + 1,
        previous_feedback=last_feedback
    )

    messages = [HumanMessage(content=prompt)]

    # ── agentic loop: keep going until no more tool calls ──
    while True:
        response = worker_llm.invoke(messages)
        messages.append(response)

        # no tool calls → LLM is done researching
        if not response.tool_calls:
            break

        # execute each tool call and feed results back
        for tool_call in response.tool_calls:
            if tool_call["name"] == "tavily_search":
                search_result = tavily_search.invoke(tool_call["args"])
                messages.append(
                    ToolMessage(
                        content=json.dumps(search_result),
                        tool_call_id=tool_call["id"]
                    )
                )

    # ── final response is the last message content ──
    final_output = response.content

    return {
        "review_queue": [json.dumps({
            "child_task_id": child_task.child_task_id,
            "sub_agent_id": child_task.sub_agent_id,
            "task": child_task.task,
            "context": child_task.context,
            "success_criteria": child_task.success_criteria.model_dump(),
            "attempt_number": child_task.attempts.count + 1,
            "output": json.loads(final_output)
        })]
    }

# ──────────────────────────────────────────────
# EDGES
# ──────────────────────────────────────────────

def router(state: SupervisorState) -> str:
    if state["agent_type"] == "simple_agent":
        return "simple_agent"
    return "supervisor_agent"


def supervisor_router(state: SupervisorState):
    if state.get("final_report"):
        return END

    pending_tasks = [
        task for task in state.get("child_tasks", [])
        if task.status == "pending"
    ]
    rejected_tasks = [task for task in state.get("child_tasks", []) if task.status == "rejected" and task.attempts.count < 5]
    tasks_to_send = [*pending_tasks, *rejected_tasks]
    if tasks_to_send:
        return [Send("worker", {"child_task": task}) for task in tasks_to_send]
    
    return END


# ──────────────────────────────────────────────
# GRAPH
# ──────────────────────────────────────────────

graph = StateGraph(SupervisorState)

graph.add_node("query_analyzer", query_analyzer)
graph.add_node("supervisor_agent", supervisor_agent)
graph.add_node("simple_agent", simple_agent)
graph.add_node("worker", worker)

graph.add_edge(START, "query_analyzer")
graph.add_conditional_edges("query_analyzer", router, {
    "simple_agent": "simple_agent",
    "supervisor_agent": "supervisor_agent"
})
graph.add_edge("simple_agent", END)
graph.add_conditional_edges("supervisor_agent", supervisor_router)
graph.add_edge("worker", "supervisor_agent")

app = graph.compile()



respo = app.invoke(
    {
        "parent_question": "What is the current scientific consensus on training strategies for Large Language Models — covering pre-training, fine-tuning, RLHF, RAG vs fine-tuning tradeoffs, emergent abilities, and where the field is actually heading in 2025?"
    }
)

print(respo.get("final_report"))