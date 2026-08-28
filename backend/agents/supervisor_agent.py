from agent_state import SupervisorState, QueryAnalyzerOutput, ChildTask, ReviewOutput
from prompt import question_checker, supervisor_agent_prompt, sub_agent_prompt, supervisor_review_prompt
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
sub_agent_id = str(uuid4())
sub_task_id = str(uuid4())

load_dotenv()

# ── structured output schema for supervisor LLM only ──
class SupervisorLLMOutput(BaseModel):
    child_tasks: List[ChildTask]

llm = ChatOpenAI(model='gpt-4o-mini')
supervisor_llm = llm.with_structured_output(SupervisorLLMOutput)
analyzer_llm = llm.with_structured_output(QueryAnalyzerOutput)
worker_llm = llm.bind_tools([tavily_search])
supervisor_review_llm = llm.with_structured_output(ReviewOutput)
# refine_llm = llm.with_structured_output(ReviewOutput)

# ──────────────────────────────────────────────
# NODES
# ──────────────────────────────────────────────

def query_analyzer(state: SupervisorState) -> dict:
    output = analyzer_llm.invoke(question_checker(state["parent_question"]))

    agent_type = "multi_agent" if (output.is_question and output.is_research_able) else "simple_agent"

    return {
        "is_question": output.is_question,
        "is_research_able": output.is_research_able,
        "agent_type": agent_type
    }


def simple_agent(state: SupervisorState) -> dict:
    response = llm.invoke(state["parent_question"])
    return {
        "final_report": response.content,
        "state": "done"
    }


def supervisor_agent(state: SupervisorState) -> dict:
    
    # ── review mode ──
    if state.get("review_queue"):
        approved = []
        rejected = []

        for item in state["review_queue"]:
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

        return {
            "approved_outputs": approved,
            "rejected_outputs": rejected,
            "review_queue": [],
            "state": "review"
        }

    # ── decompose mode ──
    output = supervisor_llm.invoke(supervisor_agent_prompt(state["parent_question"]))

    return {
        "supervisor_id": supervisor_agent_id,
        "child_tasks": output.child_tasks,
        "n_agents": len(output.child_tasks),
        "state": "assign"
    }


def worker(state: ChildTask) -> dict:
    last_feedback = None
    if state.attempts.history:
        last_entry = state.attempts.history[-1]
        last_feedback = last_entry.supervisor_feedback

    prompt = sub_agent_prompt(
        task=state.task,
        context=state.context,
        success_criteria=state.success_criteria.model_dump(),
        attempt_number=state.attempts.count + 1,
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
            "task": state.task,
            "context": state.context,
            "success_criteria": state.success_criteria.model_dump(),
            "attempt_number": state.attempts.count + 1,
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
    # first time — no approved, no rejected → fanout all
    if not state.get("approved_outputs") and not state.get("rejected_outputs"):
        return [Send("worker", task) for task in state["child_tasks"]]
    
    # some rejected → re-send only rejected tasks
    if state.get("rejected_outputs"):
        return [Send("worker", task) for task in state["child_tasks"]
                if task.status == "rejected"]
    
    # all approved, nothing rejected → done
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

print(respo)