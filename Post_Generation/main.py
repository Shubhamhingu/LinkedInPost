import asyncio
import json
import logging
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.graph.message import add_messages
from chains import ReflectionOutput
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from chains import generate_chain, reflect_chain, synthesizer_chain
import sys
import os
load_dotenv()

SERVER_URL = "http://127.0.0.1:8000/mcp"

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    reflection: ReflectionOutput
    iteration: int
    search_results: Optional[str]
    refined_search_results: Optional[str]


REFLECT = "reflect"
GENERATE = "generate"
SEARCH = "search"
SYNTHESIZE = "synthesize"



async def web_search_client(query: str):
    async with streamable_http_client(SERVER_URL) as ( read_stream, write_stream, _):
        async with ClientSession( read_stream, write_stream) as session:
            init_result = await session.initialize()
            tools = await session.list_tools()
            for tool in tools.tools:print(f" - {tool.name}")
            results = await session.call_tool("web_search_tool", {"query": query})
            for content in results.content:
                if content.type == "text":
                    parsed_json = json.loads(content.text)
                    return str(parsed_json)
            # return json.dumps(results)


async def search_node(state: MessageGraph):
    # Always search from the user's original message — preserves their authentic framing
    user_message = state["messages"][0]
    search_results = await web_search_client(user_message.content)
    return {"messages": [search_results]} 

def synthesize_node(state: MessageGraph):
    search_results = state.get("search_results", "")
    response = synthesizer_chain.invoke({"search_results": search_results})

    # Format the structured output into a well-structured string
    lines = [
        f"Main Topic: {response.main_topic}",
        "",
        "Key Insights:",
        *[f"- {i}" for i in (response.key_insights or [])],
    ]

    if response.technical_concepts:
        lines.extend(["", "Technical Concepts:", *[f"- {c}" for c in response.technical_concepts]])

    if response.real_world_applications:
        lines.extend(["", "Real-world Applications:", *[f"- {a}" for a in response.real_world_applications]])

    if response.challenges_tradeoffs:
        lines.extend(["", "Challenges and Tradeoffs:", *[f"- {t}" for t in response.challenges_tradeoffs]])

    if response.emerging_trends:
        lines.extend(["", "Trends:", *[f"- {t}" for t in response.emerging_trends]])

    structured_output = "\n".join(lines)

    return {
        "refined_search_results": structured_output
    }

def generation_node(state: MessageGraph):
    iteration = state.get("iteration", 0)
    messages = list(state["messages"])
    refined_search_results = state.get("refined_search_results", "")

    # On first generation, inject search results as a hidden context message
    # so GENERATE sees both the user's raw voice AND supporting research
    if iteration == 0 and refined_search_results:
        context = messages + [HumanMessage(content=(
            "--- Supporting web research (use as factual backing, not as the post itself) ---\n"
            f"{refined_search_results}\n"
            "---\n\n"
            "Now write the post. Start from what the user described above in their own words. "
            "Enhance it with specific facts, examples, or data from the research. Keep their authentic perspective."
        ))]
    else:
        # On subsequent iterations, messages already contain the post + reflection feedback
        context = messages

    response = generate_chain.invoke({"messages": context})
    return {
        "messages": [response],
        "iteration": iteration + 1
    }


# def reflection_node(state: MessageGraph) -> MessagesState:
#     response = reflect_chain.invoke({"messages": state["messages"]})
#     return {"messages":[HumanMessage(content=response.content)]}

def reflection_node(state: MessageGraph):
    response = reflect_chain.invoke({"messages": state["messages"]})

    reflection_text = f"""
        Approved: {response.approved}
        Quality Score: {response.quality_score}

        Strengths:
        {chr(10).join(f"- {s}" for s in response.strengths)}

        Weaknesses:
        {chr(10).join(f"- {w}" for w in response.weaknesses)}

        Recommendations:
        {chr(10).join(f"- {r}" for r in response.recommendations)}
        """

    return {
        "messages": [HumanMessage(content=reflection_text)],
        "reflection": response
    }

builder = StateGraph(state_schema=MessageGraph)
builder.add_node(SEARCH, search_node)
builder.add_node(SYNTHESIZE, synthesize_node)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(SEARCH)

def should_continue(state: MessageGraph):

    reflection = state.get("reflection")
    iteration = state.get("iteration", 0)

    if reflection:
        if reflection.approved:
            return END

        if reflection.quality_score >= 70:
            return END

    if iteration >= 10:
        return END

    return REFLECT


builder.add_edge(SEARCH, SYNTHESIZE)
builder.add_edge(SYNTHESIZE, GENERATE)
builder.add_conditional_edges(GENERATE, should_continue, path_map={END:END, REFLECT:REFLECT})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
# print(graph.get_graph().draw_mermaid())
# print(graph.get_graph().print_ascii())


async def main():
    user_input = """
        I learnt the basics of LangGraph and I found it really fascinating, It allows us to hav control over the 
        flow of the conversation and also allows to use tools in efficient way.
        It also allows us to loop over the agents as compared to LangChain offering sequential flow.
        I am planning my next project using LangGraph and I am really excited about it.
    """
    res = await graph.ainvoke({"messages":[HumanMessage(content=user_input)]}, config={"recursion_limit": 30})
    print(res["messages"][-1].content)

if __name__ == "__main__":
    # print("Hello from main!")
    # I recently came across Pydantic which is very useful for data validation especially when working with LLMs, 
    #     so the output is well structured and less chances of hallucination.
    # user_input = """
    #     I learnt the basics of LangGraph and I found it really fascinating, It allows us to hav control over the 
    #     flow of the conversation and also allows to use tools in efficient way.
    #     It also allows us to loop over the agents as compared to LangChain offering sequential flow.
    #     I am planning my next project using LangGraph and I am really excited about it.
    # """
    # res = await graph.ainvoke({"messages":[HumanMessage(content=user_input)]}, config={"recursion_limit": 30})
    # # print(res["messages"][-1].content)

    # I learnt the basics of LangGraph and I found it really fascinating, It allows us to hav control over the 
    #     flow of the conversation and also allows to use tools in efficient way.
    #     It also allows us to loop over the agents as compared to LangChain offering sequential flow.
    #     I am planning my next project using LangGraph and I am really excited about it.
    asyncio.run(main())