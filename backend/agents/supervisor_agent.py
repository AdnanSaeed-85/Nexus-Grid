from agent_state import SupervisorState
from prompt import question_checker, supervisor_agent_prompt, sub_agent_prompt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini')
structured_llm = llm.with_structured_output(SupervisorState)

def query_analyzer(state: SupervisorState):
    "You are a query analyzer agent who just deeply study the user's provided query"

    user_query = state.parent_question
    output = structured_llm.invoke(question_checker(user_query))

    print(output)
    return {"is_research_able": output.is_research_able, 'is_question': output.is_question}


def supervisor_agent(state: SupervisorState):
    "Your job is supervision and desicion making"

    user_query = state.parent_question
    if not state.is_research_able and state.is_question:
        print(user_query)
        return user_query

    





graph = StateGraph(SupervisorState)
graph.add_node('query_analyzer', query_analyzer)
graph.add_node('supervisor_agent', supervisor_agent)

graph.add_edge(START, 'query_analyzer')
graph.add_edge('query_analyzer', 'supervisor_agent')
graph.add_edge('supervisor_agent', END)

output = graph.compile()


output.invoke(
    {
        'parent_question': 'LLM stands for large language model'
    }
)