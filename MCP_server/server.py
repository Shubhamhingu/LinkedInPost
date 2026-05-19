from dotenv import load_dotenv
import json

from mcp.server.fastmcp import FastMCP

try:
    from langchain_tavily import TavilySearch
except Exception:
    class TavilySearch:
        def __init__(self, max_results: int = 5):
            self.max_results = max_results

        async def ainvoke(self, query):
            return f"[fallback] received query: {query}"

load_dotenv()

# Create MCP server
mcp = FastMCP("server")


# Tool registration
@mcp.tool()
async def web_search_tool(query: str) -> str:
    """
    Search the web using Tavily.
    """
    client = TavilySearch(max_results=3)
    results = await client.ainvoke({"query": query})
    cleaned_results = []
    for r in results.get("results", []):
        cleaned_results.append({
            "title": r.get("title"),
            "content": r.get("content"),
            "url": r.get("url"),
        })
    return json.dumps(cleaned_results)  # ← moved outside the loop


if __name__ == "__main__":
    # Runs HTTP MCP server
    mcp.run(transport="streamable-http")