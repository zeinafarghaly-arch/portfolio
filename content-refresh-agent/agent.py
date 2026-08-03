"""
The reasoning layer: takes a scored page and produces a short,
editor-facing triage explanation. This is where the "agent" part
lives -- the scoring in scoring.py is a deterministic tool; this
function is the LLM step that reasons over the tool's output.
"""

import os
from anthropic import Anthropic

SYSTEM_PROMPT = """You are a content-refresh triage assistant for an SEO/content team.
You are given one page's metrics and a priority_score (0-1, higher = review sooner)
that was already computed by a scoring tool. Your job is NOT to recompute the score --
trust it. Your job is to explain in 2-3 short sentences, in plain language a non-technical
editor would understand, why this page landed where it did and what to check first.

Be concrete: name the 1-2 metrics that most likely drove the score (long time since
last update, low CTR, or -- counterintuitively -- fairly recent content age, since
the underlying model found older content is actually less likely to be declining).
If the score is low, say clearly that this page probably does NOT need attention
right now. Never invent metrics you weren't given. Keep it to 2-3 sentences, no
headers, no bullet points."""


def explain_page(page: dict, api_key: str | None = None) -> str:
    """
    page: dict with keys content_id/url, the 6 feature columns, and priority_score.
    Returns a short natural-language triage explanation from Claude.
    """
    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    user_msg = (
        f"Page: {page.get('content_id') or page.get('url', 'unknown')}\n"
        f"priority_score: {page['priority_score']:.3f}\n"
        f"days_since_last_update: {page['days_since_last_update']}\n"
        f"content_age_days: {page['content_age_days']}\n"
        f"ctr: {page['ctr']}\n"
        f"engagement_rate: {page['engagement_rate']}\n"
        f"search_volume: {page['search_volume']}\n"
        f"avg_position: {page['avg_position']}\n\n"
        "Explain this triage result."
    )

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    return "".join(block.text for block in response.content if block.type == "text")
