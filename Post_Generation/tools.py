from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()


@tool
def web_search(query: str) -> str:
    """
    Search the web for accurate, in-depth technical information on a given topic.

    Use this tool to retrieve:
    - How something works (architecture, internals, mechanisms)
    - Latest developments, releases, or research (2024–2025)
    - Real-world production use cases and adoption examples
    - Tradeoffs, limitations, and comparisons with alternatives
    - Expert opinions, benchmarks, and engineering insights
    - Best practices and lessons learned from practitioners

    Write queries that are specific and technically precise. Prefer targeted questions
    over broad keywords. Run multiple queries on different angles of the same topic
    to build a well-rounded understanding before generating content.

    Good query examples:
    - "LangGraph stateful agent architecture how it works 2025"
    - "RAG vs fine-tuning tradeoffs production LLM systems"
    - "vector database HNSW index performance benchmarks"
    - "Kubernetes operator pattern real-world use cases"

    Bad query examples (too vague — avoid these):
    - "AI"
    - "machine learning trends"
    - "vector databases"

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