from agent_state import SupervisorState, QueryCheck
from prompt import question_checker, supervisor_agent_prompt, sub_agent_prompt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini')
structured_llm = llm.with_structured_output(QueryCheck)

def query_analyzer(state: SupervisorState):
    "You are a query analyzer agent who just deeply study the user's provided query"

    user_query = state.parent_question
    output = structured_llm.invoke(question_checker(user_query))

    print(output)
    return {"is_research_able": output.is_research_able}


graph = StateGraph(SupervisorState)

graph.add_node('query_analyzer', query_analyzer)

graph.add_edge(START, 'query_analyzer')
graph.add_edge('query_analyzer', END)

output = graph.compile()

output.invoke(
    {
        'parent_question': "What is the current scientific consensus on training strategies for Large Language Models — covering pre-training, fine-tuning, RLHF, RAG vs fine-tuning tradeoffs, emergent abilities, and where the field is actually heading in 2025?"
    }
)