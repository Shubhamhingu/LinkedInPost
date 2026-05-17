from langchain_tavily import TavilySearch
from langchain_core.tools import tool
import os
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
import json
from dotenv import load_dotenv
load_dotenv()


async def web_search_client(query: str):
    async with streamable_http_client(os.getenv("SERVER_URL")) as ( read_stream, write_stream, _):
        async with ClientSession( read_stream, write_stream) as session:
            init_result = await session.initialize()
            tools = await session.list_tools()
            # for tool in tools.tools:print(f" - {tool.name}") # Uncomment to debug available tools
            results = await session.call_tool("web_search_tool", {"query": query})
            for content in results.content:
                if content.type == "text":
                    parsed_json = json.loads(content.text)
                    return str(parsed_json)
            # return json.dumps(results)