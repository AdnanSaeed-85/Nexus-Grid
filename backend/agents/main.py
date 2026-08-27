# ======================================================================
#                           ALL IMPORTS + IDs
# ======================================================================

from agent_state import SupervisorState, QueryAnalyzerOutput
from prompt import question_checker
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from uuid import uuid4

supervisor_agent_id = str(uuid4())
sub_agent_id = str(uuid4())
sub_task_id = str(uuid4())

load_dotenv()


# ======================================================================
#                              LLM BINDES
# ======================================================================

llm = ChatOpenAI(model='gpt-4o-mini')
analyzer_llm = llm.with_structured_output(QueryAnalyzerOutput)

# ======================================================================
#                                 NODES
# ======================================================================

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


def supervisor_agent(state: SupervisorState):
    print("it's me SUPERVISOR AGENT")


# ======================================================================
#                                 EDGES
# ======================================================================

def agent_router(state: SupervisorState) -> str:
    if state["agent_type"] == "simple_agent":
        return "simple_agent"
    return "supervisor_agent"


# ======================================================================
#                             GRAPH STRUCTURE
# ======================================================================

graph = StateGraph(SupervisorState)

graph.add_node('query_analyzer', query_analyzer)
graph.add_node('simple_agent', simple_agent)
graph.add_node('supervisor_agent', supervisor_agent)

graph.add_edge(START, 'query_analyzer')
graph.add_conditional_edges('query_analyzer', agent_router, {
    'simple_agent': 'simple_agent',
    'supervisor_agent': 'supervisor_agent'
})
graph.add_edge('supervisor_agent', END)

app = graph.compile()


# ======================================================================
#                              INVOKING TIME
# ======================================================================


agent = app.invoke(
    {
        "parent_question": "What is the current scientific consensus on training strategies for Large Language Models — covering pre-training, fine-tuning, RLHF, RAG vs fine-tuning tradeoffs, emergent abilities, and where the field is actually heading in 2025?"
    }
)


# ======================================================================
#                              PRINTING STAGE
# ======================================================================