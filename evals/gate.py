"""
Deployment gate: a release only ships if its eval results meet the thresholds.

    python -m evals.gate evals/results/run-....json

Exits with status 1 (blocking the release) if any threshold is missed. The
release flow for a prompt or model change is:

1. Register the change as a new skill version (skills.py).
2. Run the eval set: python -m evals.run_evals --baseline <last good run>
3. Run this gate on the new results. If it fails, the change doesn't ship.
4. After release, watch GET /metrics closely (hypercare). If failure or
   approval rates get worse, roll back by pinning the old version:
   SKILL_VERSIONS="writer=1.0.0".
"""
import json
import os
import sys

THRESHOLDS_FILE = os.path.join(os.path.dirname(__file__), "thresholds.json")


def check(summary: dict, thresholds: dict) -> list[str]:
    """Return a list of failures; an empty list means the release can ship."""
    failures = []
    for rule, limit in thresholds.items():
        kind, metric = rule.split("_", 1)  # "min_completion_rate" -> ("min", "completion_rate")
        value = summary.get(metric)
        if value is None:
            failures.append(f"{metric}: missing from results")
        elif kind == "min" and value < limit:
            failures.append(f"{metric}: {value} is below the minimum of {limit}")
        elif kind == "max" and value > limit:
            failures.append(f"{metric}: {value} is above the maximum of {limit}")
    return failures


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("Usage: python -m evals.gate <results.json>")
        return 2
    with open(argv[0]) as f:
        summary = json.load(f)["summary"]
    with open(THRESHOLDS_FILE) as f:
        thresholds = json.load(f)

    failures = check(summary, thresholds)
    if failures:
        print("Deployment gate FAILED:")
        print("\n".join(f"  - {line}" for line in failures))
        return 1
    print("Deployment gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
