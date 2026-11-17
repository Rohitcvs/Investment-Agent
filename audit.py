"""
Audit log: an append-only record of every decision the system makes.

One JSON object per line (JSONL), so it's easy to search, filter by ticker or
trace_id, and load into other tools. It answers "what did the system decide,
when, and why?" after the fact.
"""
import json
import os
from datetime import datetime, timezone

from observability import trace_id_var


class AuditLog:
    def __init__(self, path: str):
        self.path = path

    def record(self, event: str, ticker: str | None = None, **details) -> dict:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id_var.get(),
            "ticker": ticker,
            "event": event,
            "details": details,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return entry

    def entries(self, ticker: str | None = None, limit: int = 100) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if ticker:
            rows = [r for r in rows if r.get("ticker") == ticker.upper()]
        return rows[-limit:]
