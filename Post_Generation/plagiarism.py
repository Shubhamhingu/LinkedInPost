from typing import List

from agents import Agent, Runner
from pydantic import BaseModel, Field


class PlagiarismOutput(BaseModel):
    is_plagiarized: bool = Field(
        description="True if the new post is substantially similar to any of the existing posts."
    )
    reason: str = Field(
        description="Explanation of why the post is or is not considered plagiarized."
    )
    suggestions: List[str] = Field(
        description=(
            "If plagiarized, specific suggestions for a genuinely different angle, "
            "different examples, or different framing. Empty list if not plagiarized."
        )
    )


PLAGIARISM_SYSTEM = """
You are a plagiarism detection expert for LinkedIn technical content.

You will receive:
1. A newly generated LinkedIn post.
2. A set of previously published posts that are semantically similar.

Your job is to decide whether the new post is too similar to any existing post.

A post IS plagiarized if it:
- Covers the same core idea or insight with very similar framing.
- Uses the same structure or flow (same opening concept, same progression, same conclusion angle).
- Repeats the same key talking points, examples, or analogies.
- Would feel like a duplicate or near-duplicate to a reader who saw both.

A post is NOT plagiarized if it:
- Covers the same broad topic but from a genuinely different angle.
- Uses different examples, different technical depth, or a different narrative arc.
- Reaches a different conclusion or emphasizes different tradeoffs.
- Has a distinctly different voice or structure.

Be strict but fair. Two posts on "RAG" are fine if they teach different things.
Two posts that both lead with "RAG fails when chunking is naive, here's why" are too similar.

If plagiarized, provide specific, actionable suggestions so the generator can produce a meaningfully different post.
"""

plagiarism_agent = Agent(
    name="Plagiarism Detector",
    instructions=PLAGIARISM_SYSTEM,
    model="gpt-4o-mini",
    output_type=PlagiarismOutput,
)


async def check_plagiarism(new_post: str, similar_posts: list[dict]) -> PlagiarismOutput:
    existing_formatted = "\n\n---\n\n".join(
        f"[Existing Post {i + 1}] (cosine distance: {p['distance']:.3f})\n{p['text']}"
        for i, p in enumerate(similar_posts)
    )

    prompt = (
        f"New post to evaluate:\n\n{new_post}\n\n"
        f"{'=' * 60}\n\n"
        f"Previously published posts for comparison:\n\n{existing_formatted}"
    )

    result = await Runner.run(plagiarism_agent, prompt)
    return result.final_output
