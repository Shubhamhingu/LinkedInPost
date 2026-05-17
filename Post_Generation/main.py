import asyncio
from dotenv import load_dotenv
import json
load_dotenv()
from langsmith.integrations.openai_agents_sdk import OpenAIAgentsTracingProcessor
from agents import set_trace_processors
set_trace_processors([OpenAIAgentsTracingProcessor()])
from agents import Agent, Runner, trace
from agents.mcp import MCPServerStreamableHttp
from models import ReflectionOutput, SynthesisOutput
from prompts import GENERATION_SYSTEM, REFLECTION_SYSTEM, SYNTHESIZER_SYSTEM
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger.logger import (
    log_start, log_search_start, log_search_done,
    log_synthesize_start, log_synthesize_done,
    log_generate_start, log_generate_done,
    log_reflect_start, log_reflect_done,
    log_pipeline_decision, log_final_post,
    log_error, log_run_summary, LOG_FILE,
)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

synthesizer_agent = Agent(
    name="Synthesizer",
    instructions=SYNTHESIZER_SYSTEM,
    model="gpt-4o-mini",
    output_type=SynthesisOutput,
)

generator_agent = Agent(
    name="Generator",
    instructions=GENERATION_SYSTEM,
    model="gpt-4o-mini",
)

reflector_agent = Agent(
    name="Reflector",
    instructions=REFLECTION_SYSTEM,
    model="gpt-4o-mini",
    output_type=ReflectionOutput,
)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
async def run_search(user_input: str, mcp_server=None) -> str:
    if mcp_server:
        # Direct MCP tool call — no agent needed
        result = await mcp_server.call_tool("web_search_tool", {"query": user_input})
        return result.content[0].text
    else:
        from agents import WebSearchTool
        search_agent = Agent(
            name="Searcher",
            instructions="Search for information relevant to the user's query and return the results.",
            model="gpt-4o-mini",
            tools=[WebSearchTool()],
        )
        result = await Runner.run(search_agent, user_input)
        return result.final_output

async def run_pipeline(user_input: str, mcp_server=None) -> str:
    # ── 1. SEARCH ─────────────────────────────────────────────────────────
    log_start(user_input)
    log_search_start()
    # raw_search = await run_search(user_input, mcp_server)
    if mcp_server:
        result = await mcp_server.call_tool("web_search_tool", {"query": user_input})
        raw = result.content[0].text
        parsed = json.loads(raw)
        raw_search = "\n\n".join(
            f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}"
            for r in parsed
        )
    log_search_done(raw_search)

    # ── 2. SYNTHESIZE ─────────────────────────────────────────────────────────
    log_synthesize_start()
    try:
        synthesis_prompt = (
            f"User topic / notes:\n{user_input}\n\n"
            f"Web search results:\n{raw_search}"
        )
        with trace("synthesize"):
            synthesis_result = await Runner.run(synthesizer_agent, synthesis_prompt)

        synthesis: SynthesisOutput = synthesis_result.final_output

        lines = [
            f"Main Topic: {synthesis.main_topic}", "",
            "Key Insights:",
            *[f"- {i}" for i in (synthesis.key_insights or [])],
        ]
        if synthesis.technical_concepts:
            lines += ["", "Technical Concepts:", *[f"- {c}" for c in synthesis.technical_concepts]]
        if synthesis.real_world_applications:
            lines += ["", "Real-world Applications:", *[f"- {a}" for a in synthesis.real_world_applications]]
        if synthesis.challenges_tradeoffs:
            lines += ["", "Challenges and Tradeoffs:", *[f"- {t}" for t in synthesis.challenges_tradeoffs]]
        if synthesis.emerging_trends:
            lines += ["", "Trends:", *[f"- {t}" for t in synthesis.emerging_trends]]

        refined_research = "\n".join(lines)
        log_synthesize_done(synthesis.main_topic, len(synthesis.key_insights or []))

    except Exception as e:
        log_error("Synthesis failed", e)
        raise

    # ── 3. GENERATE + REFLECT LOOP ────────────────────────────────────────────
    iteration = 0
    post = ""
    reflection: ReflectionOutput | None = None
    MAX_ITERATIONS = 10

    generation_input = (
        f"{user_input}\n\n"
        "--- Supporting web research (use as factual backing, not as the post itself) ---\n"
        f"{refined_research}\n"
        "---\n\n"
        "Now write the post. Start from what the user described above in their own words. "
        "Enhance it with specific facts, examples, or data from the research. Keep their authentic perspective."
    )

    while iteration < MAX_ITERATIONS:

        # GENERATE
        log_generate_start(iteration + 1)
        try:
            with trace(f"generate_iter_{iteration}"):
                gen_result = await Runner.run(generator_agent, generation_input)
            post = gen_result.final_output
            iteration += 1
            log_generate_done(iteration, post)
        except Exception as e:
            log_error(f"Generation failed at iteration {iteration + 1}", e)
            raise

        # REFLECT
        log_reflect_start(iteration)
        try:
            with trace(f"reflect_iter_{iteration}"):
                ref_result = await Runner.run(reflector_agent, post)
            reflection = ref_result.final_output
            log_reflect_done(
                score=reflection.quality_score,
                approved=reflection.approved,
                strengths=reflection.strengths,
                weaknesses=reflection.weaknesses,
                recommendations=reflection.recommendations,
            )
        except Exception as e:
            log_error(f"Reflection failed at iteration {iteration}", e)
            raise

        # Decision
        if reflection.approved or reflection.quality_score >= 80:
            log_pipeline_decision(
                f"Quality threshold met (score={reflection.quality_score}) — stopping after {iteration} iteration(s)."
            )
            break

        if iteration < MAX_ITERATIONS:
            log_pipeline_decision(
                f"Score {reflection.quality_score} < 80 — sending back for revision."
            )

        feedback_lines = [
            f"Quality Score: {reflection.quality_score}", "",
            "Strengths:", *[f"- {s}" for s in reflection.strengths], "",
            "Weaknesses:", *[f"- {w}" for w in reflection.weaknesses], "",
            "Recommendations:", *[f"- {r}" for r in reflection.recommendations],
        ]
        generation_input = (
            f"Previous post:\n{post}\n\n"
            f"Reviewer feedback:\n" + "\n".join(feedback_lines) + "\n\n"
            "Revise the post applying the recommendations. Preserve the author's voice."
        )

    log_final_post(post)
    log_run_summary(
        iterations=iteration,
        final_score=reflection.quality_score if reflection else 0,
        log_file=LOG_FILE,
    )

    return post


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    user_input = """
        I learnt about openai agents sdk, how it can be used to create agents with tools and how it can be used to
        create a feedback loop where the agent can critique its own output and improve it.
    """

    server_url = os.getenv("SERVER_URL")

    try:
        if server_url:
            async with MCPServerStreamableHttp(
                name="web-search-mcp",
                params={"url": server_url},
            ) as mcp_server:
                await run_pipeline(user_input, mcp_server=mcp_server)
        else:
            await run_pipeline(user_input)
    except Exception as e:
        log_error("Pipeline terminated with an unhandled error", e)


if __name__ == "__main__":
    asyncio.run(main())
