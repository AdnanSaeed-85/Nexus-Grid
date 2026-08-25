import os
from dotenv import load_dotenv
from tavily import TavilyClient
from requests.exceptions import HTTPError
from langchain_core.tools import tool

load_dotenv()

TAVILY_KEYS = [
    os.getenv("TAVILY_API_KEY_1"),
    os.getenv("TAVILY_API_KEY_2"),
    os.getenv("TAVILY_API_KEY_3"),
]

current_key_index = 0


def get_tavily_client() -> TavilyClient:
    global current_key_index
    start_index = current_key_index

    while True:
        key = TAVILY_KEYS[current_key_index]
        try:
            return TavilyClient(api_key=key)
        except HTTPError as e:
            if e.response.status_code == 429:
                current_key_index = (current_key_index + 1) % len(TAVILY_KEYS)
                if current_key_index == start_index:
                    raise Exception("All Tavily API keys exhausted")
            else:
                raise


@tool
def tavily_search(query: str) -> dict:
    """
    Search the web using Tavily and return relevant results for a given query.
    Automatically rotates through multiple API keys if credits are exhausted.
    Use this tool when you need to find current, factual, or research-based information from the web.
    """
    client = get_tavily_client()
    return client.search(query=query, max_results=2, search_depth='basic')
