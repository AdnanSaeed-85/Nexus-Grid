from agent_state import SupervisorState, QueryAnalyzerOutput, ChildTask
from prompt import question_checker, supervisor_agent_prompt, sub_agent_prompt
from agent_tool import tavily_search
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send
from pydantic import BaseModel
from typing import List
import json

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini')
analyzer_llm = llm.with_structured_output(QueryAnalyzerOutput)
worker_llm = llm.bind_tools([tavily_search])


# ── structured output schema for supervisor LLM only ──
class SupervisorLLMOutput(BaseModel):
    child_tasks: List[ChildTask]


supervisor_llm = llm.with_structured_output(SupervisorLLMOutput)


# ──────────────────────────────────────────────
# NODES
# ──────────────────────────────────────────────

def query_analyzer(state: SupervisorState) -> dict:
    output = analyzer_llm.invoke(question_checker(state.parent_question))

    agent_type = "multi_agent" if (output.is_question and output.is_research_able) else "simple_agent"

    return {
        "is_question": output.is_question,
        "is_research_able": output.is_research_able,
        "agent_type": agent_type
    }


def supervisor_agent(state: SupervisorState) -> dict:
    output = supervisor_llm.invoke(supervisor_agent_prompt(state.parent_question))

    return {
        "child_tasks": output.child_tasks,
        "n_agents": len(output.child_tasks),
        "state": "assign"
    }


def simple_agent(state: SupervisorState) -> dict:
    response = llm.invoke(state.parent_question)
    return {
        "final_report": response.content,
        "state": "done"
    }


def worker(state: ChildTask) -> dict:
    # ── build feedback from last attempt if any ──
    last_feedback = None
    if state.attempts.history:
        last_entry = state.attempts.history[-1]
        last_feedback = last_entry.supervisor_feedback

    # ── build prompt ──
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
        "review_queue": [final_output],
        "state": "review"
    }


# ──────────────────────────────────────────────
# EDGES
# ──────────────────────────────────────────────

def router(state: SupervisorState) -> str:
    if state.agent_type == "simple_agent":
        return "simple_agent"
    return "supervisor_agent"


def fanout(state: SupervisorState) -> list:
    return [
        Send("worker", task)
        for task in state.child_tasks
    ]


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
graph.add_conditional_edges("supervisor_agent", fanout, ["worker"])
graph.add_edge("simple_agent", END)
graph.add_edge("worker", END)

app = graph.compile()


app.invoke(
    {
        "parent_question": "What is the current scientific consensus on training strategies for Large Language Models — covering pre-training, fine-tuning, RLHF, RAG vs fine-tuning tradeoffs, emergent abilities, and where the field is actually heading in 2025?"
    }
)