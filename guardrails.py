"""
Guardrails: checks that keep the system safe and on track.

Input guardrail: reject bad tickers before any agent runs, so we never spend
tokens on junk input or pass untrusted text into prompts.

The other guardrails live in the graph:
- each sub-agent has a step limit, so a tool loop can't run forever
- the critic reviews every draft, with a revision cap
- a human approves the report before it's saved (human-in-the-loop)
- every decision is written to the audit log
"""
import re

# 1-5 letters, optionally followed by a class or exchange suffix: AAPL, BRK.B, BRK-B, SHOP.TO
TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}([.\-][A-Z]{1,2})?$")


def validate_ticker(ticker: str) -> tuple[bool, str]:
    cleaned = (ticker or "").strip().upper()
    if not cleaned:
        return False, "Ticker is empty."
    if not TICKER_PATTERN.match(cleaned):
        return False, f"'{ticker}' doesn't look like a stock ticker (expected something like AAPL or BRK.B)."
    return True, cleaned
