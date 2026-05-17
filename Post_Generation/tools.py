from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()


@tool
def web_search(query: str) -> str:
    """
    Search the web for accurate, in-depth technical information on a given topic.
    Args:
        query: A specific, targeted search query focused on one technical aspect.

    Returns:
        Raw web search results with titles, URLs, and content snippets. Extract
        key facts, numbers, expert insights, and concrete examples from these results
        to use as source material for generating accurate, credible content.
    """
    client = TavilySearch(max_results=5)
    results = client.run(query)
    return f"Search results for '{query}':\n\n{results}"