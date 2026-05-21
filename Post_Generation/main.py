import asyncio
import json

from dotenv import load_dotenv

load_dotenv()
from agents import set_trace_processors
from langsmith.integrations.openai_agents_sdk import \
    OpenAIAgentsTracingProcessor

set_trace_processors([OpenAIAgentsTracingProcessor()])
import os
import sys

from agents import Agent, Runner, trace
from agents.mcp import MCPServerStreamableHttp
from models import ReflectionOutput, SynthesisOutput
from plagiarism import PlagiarismOutput, check_plagiarism
from prompts import GENERATION_SYSTEM, REFLECTION_SYSTEM, SYNTHESIZER_SYSTEM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger.logger import (LOG_FILE, log_error, log_final_post,
                           log_generate_done, log_generate_start,
                           log_pipeline_decision, log_plagiarism_result,
                           log_plagiarism_start, log_post_stored,
                           log_reflect_done, log_reflect_start,
                           log_run_summary, log_search_done, log_search_start,
                           log_start, log_synthesize_done, log_synthesize_start,
                           log_user_decision)
from Pre_Generation.topic_agents import get_selected_topic
from vector_store.store import LinkedInPostStore

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
# Helpers
# ---------------------------------------------------------------------------

async def prompt_user_approval() -> bool:
    response = await asyncio.to_thread(
        input,
        "\nDo you want to keep this post and store it? [y/n]: ",
    )
    return response.strip().lower() in ("y", "yes")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(user_input: str, mcp_server=None) -> str:
    store = LinkedInPostStore()

    # ── 1. SEARCH ─────────────────────────────────────────────────────────
    log_start(user_input)
    log_search_start()
    if mcp_server:
        result = await mcp_server.call_tool("web_search_tool", {"query": user_input})
        raw = result.content[0].text
        parsed = json.loads(raw)
        raw_search = "\n\n".join(
            f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}"
            for r in parsed
        )
    log_search_done(raw_search)

    # ── 2. SYNTHESIZE ──────────────────────────────────────────────────────
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

    # ── 3. GENERATE + REFLECT + PLAGIARISM LOOP ───────────────────────────
    iteration = 0
    plagiarism_attempts = 0
    post = ""
    reflection: ReflectionOutput | None = None
    MAX_ITERATIONS = 10
    MAX_PLAGIARISM_ATTEMPTS = 3

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

        # Quality gate
        quality_met = reflection.approved or reflection.quality_score >= 70

        if quality_met:
            log_pipeline_decision(
                f"Quality threshold met (score={reflection.quality_score}) — running plagiarism check."
            )

            # PLAGIARISM CHECK
            try:
                similar_posts = store.search_similar(post, top_k=5, distance_threshold=0.3)

                if not similar_posts:
                    log_plagiarism_start(0)
                    log_plagiarism_result(False, "No similar posts found in the store.", [])
                    log_final_post(post)
                    accepted = await prompt_user_approval()
                    log_user_decision(accepted)
                    if accepted:
                        post_id = store.add_post(
                            post,
                            metadata={
                                "topic": synthesis.main_topic,
                                "quality_score": reflection.quality_score,
                                "iterations": iteration,
                            },
                        )
                        log_post_stored(post_id, store.count())
                    break

                log_plagiarism_start(len(similar_posts))
                with trace(f"plagiarism_check_attempt_{plagiarism_attempts}"):
                    plag_result: PlagiarismOutput = await check_plagiarism(post, similar_posts)

                log_plagiarism_result(
                    plag_result.is_plagiarized,
                    plag_result.reason,
                    plag_result.suggestions,
                )

                if not plag_result.is_plagiarized:
                    log_final_post(post)
                    accepted = await prompt_user_approval()
                    log_user_decision(accepted)
                    if accepted:
                        post_id = store.add_post(
                            post,
                            metadata={
                                "topic": synthesis.main_topic,
                                "quality_score": reflection.quality_score,
                                "iterations": iteration,
                            },
                        )
                        log_post_stored(post_id, store.count())
                    break

                # Plagiarized — try a different angle
                plagiarism_attempts += 1
                if plagiarism_attempts >= MAX_PLAGIARISM_ATTEMPTS:
                    log_pipeline_decision(
                        f"Max plagiarism attempts ({MAX_PLAGIARISM_ATTEMPTS}) reached — "
                        "could not generate a sufficiently unique post. Pipeline stopped."
                    )
                    break

                log_pipeline_decision(
                    f"Post is too similar to existing content (attempt {plagiarism_attempts}) — regenerating with a different angle."
                )
                suggestion_text = "\n".join(f"- {s}" for s in plag_result.suggestions)
                generation_input = (
                    f"Previous post:\n{post}\n\n"
                    f"IMPORTANT: This post is too similar to previously published content.\n"
                    f"Reason: {plag_result.reason}\n\n"
                    f"Suggestions for a genuinely different angle:\n{suggestion_text}\n\n"
                    "Rewrite the post taking a clearly different approach. "
                    "Different opening, different examples, different framing. "
                    "Preserve technical accuracy but make it feel like a fresh perspective."
                )
                continue  # re-enter the loop without the quality-rejection feedback path

            except Exception as e:
                log_error("Plagiarism check failed — pipeline stopped", e)
                break

        # Quality not met — send back for revision
        log_pipeline_decision(
            f"Score {reflection.quality_score} < 70 — sending back for revision."
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
    server_url = os.getenv("SERVER_URL")
    try:
        if server_url:
            async with MCPServerStreamableHttp(
                name="web-search-mcp",
                params={"url": server_url},
            ) as mcp_server:
                user_input = await get_selected_topic(mcp_server)
                await run_pipeline(user_input, mcp_server=mcp_server)
        else:
            user_input = await get_selected_topic(None)
            await run_pipeline(user_input)
    except Exception as e:
        log_error("Pipeline terminated with an unhandled error", e)


if __name__ == "__main__":
    asyncio.run(main())
