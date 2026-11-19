"""
Per-customer context: the structured facts every agent reads before it acts.

Each customer has a small, validated profile (risk tolerance, time horizon,
notes). The graph loads it once at the start of a run and passes it to the
writer and critic, so reports are tailored to the customer and the critic can
check the fit. Keeping it structured and validated (instead of free text in a
prompt) means agents always see the same, trustworthy context.

Stored in SQLite so it survives restarts.
"""
import json
import sqlite3
import threading
from datetime import datetime, timezone

ALLOWED_KEYS = {"risk_tolerance", "time_horizon", "notes"}
RISK_LEVELS = {"low", "medium", "high"}
HORIZONS = {"short", "medium", "long"}


def validate_context(context: dict) -> dict:
    unknown = set(context) - ALLOWED_KEYS
    if unknown:
        raise ValueError(f"Unknown context fields: {sorted(unknown)}. Allowed: {sorted(ALLOWED_KEYS)}")
    if "risk_tolerance" in context and context["risk_tolerance"] not in RISK_LEVELS:
        raise ValueError(f"risk_tolerance must be one of {sorted(RISK_LEVELS)}")
    if "time_horizon" in context and context["time_horizon"] not in HORIZONS:
        raise ValueError(f"time_horizon must be one of {sorted(HORIZONS)}")
    if len(str(context.get("notes", ""))) > 500:
        raise ValueError("notes must be 500 characters or fewer")
    return context


def context_as_prompt(context: dict | None) -> str:
    """Render the context for a prompt, or an empty string if there is none."""
    if not context:
        return ""
    lines = [f"- {key.replace('_', ' ')}: {value}" for key, value in sorted(context.items())]
    return "## Customer context\n" + "\n".join(lines) + "\n"


class CustomerContextStore:
    def __init__(self, path: str = ":memory:"):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS customer_context ("
                "customer_id TEXT PRIMARY KEY, context TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )
            self._conn.commit()

    def get(self, customer_id: str) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT context FROM customer_context WHERE customer_id = ?", (customer_id,)
            ).fetchone()
        return json.loads(row[0]) if row else {}

    def set(self, customer_id: str, context: dict) -> dict:
        context = validate_context(dict(context))
        with self._lock:
            self._conn.execute(
                "INSERT INTO customer_context (customer_id, context, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(customer_id) DO UPDATE SET context = excluded.context, updated_at = excluded.updated_at",
                (customer_id, json.dumps(context), datetime.now(timezone.utc).isoformat()),
            )
            self._conn.commit()
        return context
