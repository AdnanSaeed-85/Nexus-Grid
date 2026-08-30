from agent_state import SupervisorState
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini')
embedded_model = OpenAIEmbeddings(model='text-embedding-3-small')