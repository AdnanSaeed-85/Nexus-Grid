from agent_state import SupervisorState
from prompt import question_checker, supervisor_agent_prompt, sub_agent_prompt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini')
structured_llm = llm.with_structured_output(SupervisorState)

def query_analyzer(state: SupervisorState):
    "You are a query analyzer agent who just deeply study the user's provided query"
    user_query = state.parent_question
    output = structured_llm.invoke(question_checker(user_query))
    if output.is_research_able and output.is_question:
        return {"is_research_able": output.is_research_able, 'is_question': output.is_question, 'agent_type': 'multi_agent'}
    return {"is_research_able": output.is_research_able, 'is_question': output.is_question, 'agent_type': 'simple_agent'}

    
def supervisor_agent(state: SupervisorState):
    "Your job is supervision and desicion making"

    user_query = state.parent_question
    if not state.is_research_able and state.is_question:
        return user_query

    output = structured_llm.invoke(supervisor_agent_prompt(user_query))

    return {
        "parent_question": user_query,
        "state": output.state,
        "child_tasks": output.child_tasks,
        "n_agents": len(output.child_tasks)
    }

def router(state: SupervisorState):
    agent = state.agent_type

    if agent == 'simple_agent':
        pass
    elif agent == 'multi_agent':
        pass

    

graph = StateGraph(SupervisorState)
graph.add_node('query_analyzer', query_analyzer)
graph.add_node('supervisor_agent', supervisor_agent)

graph.add_edge(START, 'query_analyzer')
graph.add_edge('query_analyzer', 'supervisor_agent')
graph.add_edge('supervisor_agent', END)

output = graph.compile()


output.invoke(
    {
        'parent_question': "What is the current scientific consensus on training strategies for Large Language Models — covering pre-training, fine-tuning, RLHF, RAG vs fine-tuning tradeoffs, emergent abilities, and where the field is actually heading in 2025?"
    }
)