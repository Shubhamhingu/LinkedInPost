"""
logger.py — colored terminal + file logging for the LinkedIn post pipeline.

Uses `rich` for beautiful terminal output and Python's built-in `logging`
for a plain-text log file written alongside every run.

Color scheme:
  SEARCH     →  bold cyan
  SYNTHESIZE →  bold blue
  GENERATE   →  bold yellow
  REFLECT    →  bold magenta
  PIPELINE   →  bold green  (decisions / final output)
  ERROR      →  bold red
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# for noisy in ["httpcore", "httpx", "openai.agents", "mcp.client.streamable_http", "asyncio"]:
#     logging.getLogger(noisy).setLevel(logging.WARNING)
# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

THEME = Theme({
    "search":     "bold cyan",
    "synthesize": "bold blue",
    "generate":   "bold yellow",
    "reflect":    "bold magenta",
    "pipeline":   "bold green",
    "error":      "bold red",
    "label":      "dim white",
    "score.low":  "bold red",
    "score.mid":  "bold yellow",
    "score.high": "bold green",
    "divider":    "dim white",
})

console = Console(theme=THEME, highlight=False)


# ---------------------------------------------------------------------------
# File logger (plain text, no ANSI codes)
# ---------------------------------------------------------------------------

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"pipeline_{_run_id}.log"

_file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))

_rich_handler = RichHandler(
    console=console,
    show_time=False,
    show_path=False,
    markup=True,
    rich_tracebacks=True,
)
_rich_handler.setLevel(logging.INFO)

for noisy in ["httpcore", "httpx", "openai._base_client", "openai.agents", "mcp.client.streamable_http", "asyncio"]:
    noisy_logger = logging.getLogger(noisy)
    noisy_logger.setLevel(logging.DEBUG)        # allow DEBUG into the file
    noisy_logger.propagate = True               # route through root logger


logging.basicConfig(
    level=logging.DEBUG,
    handlers=[_file_handler, _rich_handler],
)

log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Pipeline-specific helpers
# ---------------------------------------------------------------------------

def _score_style(score: int) -> str:
    if score >= 80:
        return "score.high"
    if score >= 60:
        return "score.mid"
    return "score.low"


def log_start(user_input: str) -> None:
    console.print()
    console.print(Rule("[pipeline]▶  PIPELINE START[/pipeline]", style="pipeline"))
    console.print(Panel(
        user_input.strip(),
        title="[label]User Input[/label]",
        border_style="dim white",
        padding=(0, 2),
    ))
    log.info("Pipeline started")
    log.debug("User input: %s", user_input.strip())


def log_search_start() -> None:
    console.print()
    console.print(Rule("[search]  SEARCH[/search]", style="search"))
    log.info("[SEARCH] Starting web search")


def log_search_done(raw: str) -> None:
    preview = raw[:300].replace("\n", " ") + ("…" if len(raw) > 300 else "")
    console.print(f"  [search]✓ Search complete[/search]  [label]({len(raw)} chars)[/label]")
    log.info("[SEARCH] Done — %d chars retrieved", len(raw))
    log.debug("[SEARCH] Raw preview: %s", preview)


def log_synthesize_start() -> None:
    console.print()
    console.print(Rule("[synthesize]  SYNTHESIZE[/synthesize]", style="synthesize"))
    log.info("[SYNTHESIZE] Starting synthesis")


def log_synthesize_done(main_topic: str, insight_count: int) -> None:
    console.print(f"  [synthesize]✓ Synthesized[/synthesize]  "
                  f"[label]topic:[/label] {main_topic}  "
                  f"[label]insights:[/label] {insight_count}")
    log.info("[SYNTHESIZE] Done — topic: %s | insights: %d", main_topic, insight_count)


def log_generate_start(iteration: int) -> None:
    console.print()
    console.print(Rule(
        f"[generate]  GENERATE  iteration {iteration}[/generate]",
        style="generate",
    ))
    log.info("[GENERATE] Iteration %d started", iteration)


def log_generate_done(iteration: int, post: str) -> None:
    word_count = len(post.split())
    console.print(f"  [generate]✓ Post generated[/generate]  [label]{word_count} words[/label]")
    log.info("[GENERATE] Iteration %d done — %d words", iteration, word_count)
    log.debug("[GENERATE] Post:\n%s", post)


def log_reflect_start(iteration: int) -> None:
    console.print()
    console.print(Rule(
        f"[reflect]  REFLECT  iteration {iteration}[/reflect]",
        style="reflect",
    ))
    log.info("[REFLECT] Iteration %d started", iteration)


def log_reflect_done(score: int, approved: bool, strengths: list, weaknesses: list, recommendations: list) -> None:
    style = _score_style(score)

    # Score badge
    score_text = Text(f"  Score: {score}/100", style=style)
    approved_text = Text(
        "  ✓ APPROVED" if approved else "  ✗ needs revision",
        style="score.high" if approved else "score.low",
    )
    console.print(score_text, approved_text)

    # Feedback table
    table = Table.grid(padding=(0, 2))
    table.add_column(style="label", no_wrap=True)
    table.add_column()

    def bullet_list(items: list) -> str:
        return "\n".join(f"• {i}" for i in items) if items else "—"

    table.add_row("Strengths",      bullet_list(strengths))
    table.add_row("Weaknesses",     bullet_list(weaknesses))
    table.add_row("Recommendations",bullet_list(recommendations))

    console.print(Panel(table, border_style="dim magenta", padding=(0, 1)))

    log.info("[REFLECT] Score: %d | Approved: %s", score, approved)
    for s in strengths:
        log.debug("[REFLECT] Strength: %s", s)
    for w in weaknesses:
        log.debug("[REFLECT] Weakness: %s", w)
    for r in recommendations:
        log.debug("[REFLECT] Recommendation: %s", r)


def log_pipeline_decision(reason: str) -> None:
    console.print(f"\n  [pipeline]⚡ {reason}[/pipeline]")
    log.info("[PIPELINE] %s", reason)


def log_final_post(post: str) -> None:
    console.print()
    console.print(Rule("[pipeline]  FINAL POST[/pipeline]", style="pipeline"))
    console.print(Panel(
        post,
        border_style="green",
        padding=(1, 2),
    ))
    log.info("[PIPELINE] Final post delivered (%d words)", len(post.split()))
    log.info("[PIPELINE] Full post:\n%s", post)


def log_error(msg: str, exc: Exception | None = None) -> None:
    console.print(f"\n  [error]✗ ERROR: {msg}[/error]")
    if exc:
        console.print_exception()
    log.error("[ERROR] %s", msg, exc_info=exc)


def log_user_decision(accepted: bool) -> None:
    if accepted:
        console.print("  [pipeline]✓ User accepted the post — storing.[/pipeline]")
        log.info("[USER] Post accepted for storage.")
    else:
        console.print("  [error]✗ User declined the post — not storing.[/error]")
        log.info("[USER] Post declined — not stored.")


def log_plagiarism_gate(total: int, passed: int, threshold: float, scored_posts: list[dict]) -> None:
    passed_style = "pipeline" if passed > 0 else "label"
    console.print(
        f"  [reflect]Jaccard gate:[/reflect] "
        f"[{passed_style}]{passed} / {total} passed[/{passed_style}]  "
        f"[label](threshold={threshold})[/label]"
    )

    table = Table.grid(padding=(0, 3))
    table.add_column(style="label", no_wrap=True)
    table.add_column(style="label", no_wrap=True)
    table.add_column(style="label", no_wrap=True)
    table.add_column(no_wrap=True)

    for i, p in enumerate(scored_posts):
        lex = p["lexical_similarity"]
        gate_passed = lex >= threshold
        status = Text("✓ passed", style="pipeline") if gate_passed else Text("✗ filtered", style="error")
        table.add_row(
            f"Post {i + 1}",
            f"cosine={p['distance']:.3f}",
            f"jaccard={lex:.3f}",
            status,
        )

    console.print(table)
    log.info("[PLAGIARISM] Jaccard gate: %d/%d passed (threshold=%.2f)", passed, total, threshold)
    for i, p in enumerate(scored_posts):
        gate_passed = p["lexical_similarity"] >= threshold
        log.debug(
            "[PLAGIARISM] Post %d — cosine=%.3f jaccard=%.3f %s",
            i + 1, p["distance"], p["lexical_similarity"],
            "PASSED" if gate_passed else "FILTERED",
        )


def log_plagiarism_start(similar_count: int) -> None:
    console.print()
    console.print(Rule("[reflect]  PLAGIARISM CHECK[/reflect]", style="reflect"))
    console.print(f"  [reflect]Checking against {similar_count} similar post(s) from the store[/reflect]")
    log.info("[PLAGIARISM] Checking against %d similar post(s)", similar_count)


def log_plagiarism_result(is_plagiarized: bool, reason: str, suggestions: list) -> None:
    if is_plagiarized:
        console.print("  [error]✗ PLAGIARIZED[/error]")
        console.print(f"  [label]Reason:[/label] {reason}")
        for s in suggestions:
            console.print(f"    [label]→[/label] {s}")
    else:
        console.print("  [pipeline]✓ Original — no plagiarism detected[/pipeline]")
        console.print(f"  [label]Reason:[/label] {reason}")
    log.info("[PLAGIARISM] is_plagiarized=%s | reason: %s", is_plagiarized, reason)
    for s in suggestions:
        log.debug("[PLAGIARISM] Suggestion: %s", s)


def log_post_stored(post_id: str, total: int) -> None:
    console.print(f"  [pipeline]✓ Post stored in vector DB[/pipeline]  "
                  f"[label]id:[/label] {post_id}  [label]total posts:[/label] {total}")
    log.info("[STORE] Post stored — id: %s | total in store: %d", post_id, total)


def log_run_summary(iterations: int, final_score: int, log_file: Path = LOG_FILE) -> None:
    console.print()
    console.print(Rule("[pipeline]  RUN SUMMARY[/pipeline]", style="pipeline"))

    table = Table.grid(padding=(0, 3))
    table.add_column(style="label", no_wrap=True)
    table.add_column()

    table.add_row("Iterations",  str(iterations))
    table.add_row("Final score", Text(f"{final_score}/100", style=_score_style(final_score)))
    table.add_row("Log file",    str(log_file))

    console.print(table)
    console.print()
    log.info("[SUMMARY] iterations=%d | final_score=%d | log=%s", iterations, final_score, log_file)
