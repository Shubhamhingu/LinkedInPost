import os
import sys
from datetime import datetime
from typing import List

from agents import Agent, Runner
from dotenv import load_dotenv
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from logger.logger import (
#     log_start, log_search_start, log_search_done,
#     log_synthesize_start, log_synthesize_done,
#     log_generate_start, log_generate_done,
#     log_reflect_start, log_reflect_done,
#     log_pipeline_decision, log_final_post,
#     log_error, log_run_summary, LOG_FILE,
# )
load_dotenv()


class ResearchTopic(BaseModel):
    title: str = Field(description="Short title of the discovered AI/Data Science topic.")
    description: str = Field(description="Brief explanation of what the topic is about.")
    category: str = Field(description="Category such as GenAI, Data Science, ML Engineering, RAG, LLMOps, MCP, or Agentic AI.")
    why_relevant: str = Field(description="Why this topic is useful for a LinkedIn technical post.")
    linkedin_angle: str = Field(description="The unique angle or perspective for writing a LinkedIn post.")
    search_query: str = Field(description="A strong search query that can be passed into the existing post generation pipeline.")

class ResearchTopicsOutput(BaseModel):
    topics: List[ResearchTopic] = Field(
        description="List of recent AI/Data Science/GenAI technical topics."
    )

class SelectedTopicOutput(BaseModel):
    selected_title: str = Field(description="The final selected topic title.")
    selected_description: str = Field(description="Short description of the selected topic.")
    selected_category: str = Field(description="Category of the selected topic.")
    reason_for_selection: str = Field(description="Why this topic was selected over the others.")
    final_user_input: str = Field(
        description="The final topic input that should be passed into main.py run_pipeline(user_input)."
    )

RESEARCH_AGENT_SYSTEM = """
You are a Research Agent for an autonomous LinkedIn technical content system.

Your job is to identify recent and high-value topics in:
- Artificial Intelligence
- Data Science
- Machine Learning Engineering
- Generative AI
- Agentic AI
- RAG
- LLMOps
- Vector Databases
- MCP
- AI Evaluation
- AI Observability

You will receive raw web search results or a broad research instruction.

Important rule:
You must base the topics only on the provided web search results. Do not invent topics that are not supported by the search results.

Your task:
- Extract 5 strong topic ideas.
- Each topic should be specific, technical, and useful for a LinkedIn post.
- Avoid generic topics like "AI is changing the world".
- Prefer topics that have practical engineering value.
- Prefer topics that can teach something to data scientists, AI engineers, ML engineers, or software engineers.
- For each topic, create a search_query that can be used later for deeper web search.

Return structured output only.
"""


TOPIC_SELECTION_SYSTEM = """
You are a Topic Selection Agent for a LinkedIn AI/Data Science content pipeline.

You will receive a list of candidate topics from the Research Agent.

Your job is to select exactly one topic.

Selection criteria:
1. The topic should be recent or connected to current AI/Data Science trends.
2. The topic should be technical enough for a strong LinkedIn post.
3. The topic should not be too generic.
4. The topic should have practical value for engineers or data professionals.
5. The topic should have a strong writing angle.
6. The topic should be suitable for the existing post generation pipeline.

Return one final selected topic.

The final_user_input should be a clear paragraph that can be directly passed into the existing main.py run_pipeline(user_input).
"""

research_agent = Agent(
    name="Research Agent",
    instructions=RESEARCH_AGENT_SYSTEM,
    model="gpt-4o-mini",
    output_type=ResearchTopicsOutput,
)

topic_selection_agent = Agent(
    name="Topic Selection Agent",
    instructions=TOPIC_SELECTION_SYSTEM,
    model="gpt-4o-mini",
    output_type=SelectedTopicOutput,
)


async def run_web_search_for_topics(mcp_server) -> str:
    current_date = datetime.now().strftime("%Y-%m-%d")

    search_query = f"""
    Latest AI, Data Science, Machine Learning Engineering, GenAI, RAG,
    LLMOps, MCP, AI agents, vector database, and AI evaluation technology
    updates from this week. Current date: {current_date}.
    Focus on topics that are useful for technical LinkedIn posts.
    """

    result = await mcp_server.call_tool(
        "web_search_tool",
        {"query": search_query},
    )

    raw = result.content[0].text
    return raw


async def run_research_agent(research_input: str) -> ResearchTopicsOutput:
    result = await Runner.run(research_agent, research_input)
    return result.final_output


async def run_topic_selection_agent(
    research_output: ResearchTopicsOutput,
) -> SelectedTopicOutput:
    topic_data = research_output.model_dump_json(indent=2)

    selection_input = f"""
    Candidate topics from Research Agent:

    {topic_data}

    Select the best topic for a LinkedIn technical post.
    """

    result = await Runner.run(topic_selection_agent, selection_input)
    return result.final_output


async def get_selected_topic(mcp_server) -> str:
    raw_search_results = await run_web_search_for_topics(mcp_server)

    research_input = f"""
    Recent web search results:

    {raw_search_results}

    Based only on these web search results, identify strong recent AI/Data Science topics.
    """

    research_output = await run_research_agent(research_input)
    selected_topic = await run_topic_selection_agent(research_output)

    return selected_topic.final_user_input


# async def main():
#     server_url = os.getenv("SERVER_URL")

#     if not server_url:
#         raise ValueError("SERVER_URL is missing in .env. Start your MCP server and set SERVER_URL first.")

#     async with MCPServerStreamableHttp(
#         name="web-search-mcp",
#         params={"url": server_url},
#     ) as mcp_server:

#         raw_search_results = await run_web_search_for_topics(mcp_server)

#         print("\nRaw Web Search Results:")
#         print("=" * 70)
#         print(raw_search_results)

#         research_input = f"""
#         Recent web search results:

#         {raw_search_results}

#         Based only on these web search results, identify strong recent AI/Data Science topics.
#         """

#         research_output = await run_research_agent(research_input)

#         print("\nCandidate Topics from Research Agent:")
#         print("=" * 70)

#         for index, topic in enumerate(research_output.topics, start=1):
#             print(f"\nTopic {index}: {topic.title}")
#             print(f"Category: {topic.category}")
#             print(f"Description: {topic.description}")
#             print(f"Why Relevant: {topic.why_relevant}")
#             print(f"LinkedIn Angle: {topic.linkedin_angle}")
#             print(f"Search Query: {topic.search_query}")

#         selected_topic = await run_topic_selection_agent(research_output)

#         print("\n\nSelected Topic for Pipeline:")
#         print("=" * 70)
#         print(f"Title: {selected_topic.selected_title}")
#         print(f"Category: {selected_topic.selected_category}")
#         print(f"Description: {selected_topic.selected_description}")
#         print(f"Reason: {selected_topic.reason_for_selection}")

#         print("\nFinal user_input for main.py:")
#         print("-" * 70)
#         print(selected_topic.final_user_input)


# if __name__ == "__main__":
#     asyncio.run(main())