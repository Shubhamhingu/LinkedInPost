import os
import re
import sys
from typing import List

from agents import Agent, Runner
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger.logger import log_plagiarism_gate


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


PLAGIARISM_SYSTEM = """You are an expert reviewer for detecting presentation-level plagiarism in LinkedIn technical posts.

You will receive:

1. A newly generated post.
2. A set of previously published posts.

Your task is to determine whether the NEW post is too similar in presentation to any existing post.

IMPORTANT PRINCIPLE:
Technical concepts, educational explanations, and common examples are naturally repeated online and are NOT considered plagiarism by themselves.

The SAME:

* topic
* technical insight
* educational concept
* industry terminology
* standard example
* common analogy

are ALL acceptable.

Examples alone should NOT be treated as plagiarism because many technical concepts are commonly explained using the same canonical examples.

A post should ONLY be flagged if the OVERALL PRESENTATION is too similar.

Focus on:

* opening hook
* framing of the idea
* sequence of explanation
* narrative structure
* rhetorical style
* pacing of insights
* transitions between ideas
* conclusion/takeaway
* repeated phrasing patterns
* overall “feel” of the post

Do NOT heavily penalize:

* shared technical examples
* similar analogies
* discussing the same tradeoffs
* similar educational explanations

A post is PLAGIARIZED only if a reader would feel:
“I’ve essentially read this exact post before.”

Examples of acceptable overlap:

* Two posts explain RAG chunking using the same PDF example but structure the explanation differently.
* Two posts use the same caching analogy but emphasize different lessons.
* Two posts discuss hallucinations but use different narrative flow and conclusions.

Examples of plagiarism:

* Same opening premise
* Same progression of ideas in the same order
* Same insight reveal pattern
* Same rhetorical cadence
* Same conclusion framing
* Near-identical wording across multiple sections

Evaluation priority:

1. Presentation structure
2. Narrative flow
3. Framing originality
4. Writing style similarity
5. Repeated phrasing

LOW priority:

* topic overlap
* concept overlap
* shared examples

Output format:

Decision: <SAFE / BORDERLINE / TOO_SIMILAR>

Confidence: <LOW / MEDIUM / HIGH>

Reasoning:

* Explain whether overlap is conceptual or structural.
* Mention which elements are genuinely similar.
* Explicitly state whether similarities are merely educational/common.

If TOO_SIMILAR:
Provide actionable suggestions to change:

* framing
* structure
* ordering
* hook
* narrative angle
* takeaway
"""

plagiarism_agent = Agent(
    name="Plagiarism Detector",
    instructions=PLAGIARISM_SYSTEM,
    model="gpt-4o-mini",
    output_type=PlagiarismOutput,
)

LEXICAL_SIMILARITY_THRESHOLD = 0.12


def _trigrams(text: str) -> set[tuple]:
    words = re.findall(r'\b\w+\b', text.lower())
    return set(zip(words, words[1:], words[2:]))


def compute_lexical_similarity(text1: str, text2: str) -> float:
    """Word trigram Jaccard similarity — captures structural/phrasing overlap, not topic."""
    t1, t2 = _trigrams(text1), _trigrams(text2)
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


async def check_plagiarism(new_post: str, similar_posts: list[dict]) -> PlagiarismOutput:
    # Compute jaccard for all retrieved posts first so we can log passed vs filtered.
    scored = [
        {**post, "lexical_similarity": round(compute_lexical_similarity(new_post, post["text"]), 4)}
        for post in similar_posts
    ]

    candidates = [p for p in scored if p["lexical_similarity"] >= LEXICAL_SIMILARITY_THRESHOLD]
    log_plagiarism_gate(len(scored), len(candidates), LEXICAL_SIMILARITY_THRESHOLD, scored)

    if not candidates:
        return PlagiarismOutput(
            is_plagiarized=False,
            reason="Posts share the same topic area but no significant structural or phrasing overlap was found.",
            suggestions=[],
        )

    candidates.sort(key=lambda x: x["lexical_similarity"], reverse=True)

    existing_formatted = "\n\n---\n\n".join(
        f"[Existing Post {i + 1}] (cosine distance: {p['distance']:.3f}, lexical similarity: {p['lexical_similarity']:.3f})\n{p['text']}"
        for i, p in enumerate(candidates)
    )

    prompt = (
        f"New post to evaluate:\n\n{new_post}\n\n"
        f"{'=' * 60}\n\n"
        f"Previously published posts that passed structural similarity gate:\n\n{existing_formatted}"
    )

    result = await Runner.run(plagiarism_agent, prompt)
    return result.final_output
