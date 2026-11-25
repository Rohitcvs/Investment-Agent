"""
Guardrails: checks that keep the system safe and on track.

Input guardrail: reject bad tickers before any agent runs, so we never spend
tokens on junk input or pass untrusted text into prompts.

Untrusted-content guardrail: web search results are written by strangers, so
they are wrapped and marked as data before an agent sees them, and obvious
prompt-injection phrases are removed and logged.

Output guardrail: a report must contain a Buy/Hold/Sell recommendation before
it can go to a human for approval. This is plain code, so it always runs,
even when the critic (an LLM) says APPROVED.

The other guardrails live in the graph:
- each sub-agent has a step limit, so a tool loop can't run forever
- the critic reviews every draft, with a revision cap
- a human approves the report before it's saved (human-in-the-loop)
- every decision is written to the audit log
"""
import re

from observability import log_event

# 1-5 letters, optionally followed by a class or exchange suffix: AAPL, BRK.B, BRK-B, SHOP.TO
TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}([.\-][A-Z]{1,2})?$")


def validate_ticker(ticker: str) -> tuple[bool, str]:
    cleaned = (ticker or "").strip().upper()
    if not cleaned:
        return False, "Ticker is empty."
    if not TICKER_PATTERN.match(cleaned):
        return False, f"'{ticker}' doesn't look like a stock ticker (expected something like AAPL or BRK.B)."
    return True, cleaned


# --- Untrusted content (prompt injection) ------------------------------------

UNTRUSTED_OPEN = "<untrusted_content"
UNTRUSTED_CLOSE = "</untrusted_content>"

# Phrases that try to give the model new instructions. This list can't catch every
# attack, which is why it is only one layer: the agents also have read-only tools,
# a human approves every report, and nothing is saved without that approval.
INJECTION_PATTERN = re.compile(
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+instructions"
    r"|disregard\s+(all\s+|any\s+)?(the\s+|your\s+)?(previous\s+|prior\s+)?(instructions|rules|system prompt)"
    r"|forget\s+(all\s+|your\s+)?(previous\s+)?instructions"
    r"|you\s+are\s+now\s+"
    r"|new\s+instructions\s*:"
    r"|reveal\s+(your\s+|the\s+)?system\s+prompt",
    re.IGNORECASE,
)
_WRAPPER_TAGS = re.compile(r"</?\s*untrusted_content[^>]*>", re.IGNORECASE)


def wrap_untrusted(text: str, source: str) -> str:
    """
    Mark outside text as data before it reaches a prompt.

    - Removes any wrapper tags inside the text, so it can't "close" the wrapper early.
    - Replaces known injection phrases and logs that it happened.
    - Wraps the result in tags the research skill is told never to take orders from.
    """
    cleaned = _WRAPPER_TAGS.sub("", text or "")
    cleaned, hits = INJECTION_PATTERN.subn("[removed: possible prompt injection]", cleaned)
    if hits:
        log_event("prompt_injection_suspected", source=source, matches=hits)
    return f'{UNTRUSTED_OPEN} source="{source}">\n{cleaned}\n{UNTRUSTED_CLOSE}'


# --- Output checks -----------------------------------------------------------

_RECOMMENDATION = re.compile(r"\b(strong\s+)?(buy|hold|sell)\b", re.IGNORECASE)


def extract_recommendation(report: str | None) -> str | None:
    """The last Buy/Hold/Sell mentioned, since reports end with the recommendation."""
    if not report:
        return None
    matches = _RECOMMENDATION.findall(report)
    return matches[-1][1].capitalize() if matches else None


def check_report(report: str | None) -> list[str]:
    """Problems that must be fixed before a person is asked to approve the report."""
    issues = []
    if not report or not report.strip():
        issues.append("The report is empty.")
    elif extract_recommendation(report) is None:
        issues.append("The report has no clear Buy, Hold or Sell recommendation.")
    return issues
