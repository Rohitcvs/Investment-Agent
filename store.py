"""
Persistent run tracking and checkpointing.

Two things need to survive a restart:

1. The graph's checkpoints, so a run paused for human approval can still be
   approved after the server restarts. `sqlite_checkpointer` swaps LangGraph's
   in-memory checkpointer for a SQLite-backed one (production would use Postgres).
2. A record of every run (status, customer, trace ID, cost, latency), so queued
   and failed runs can be looked up and recent runs can be monitored after a
   release. That's the RunStore, which also powers the /metrics endpoint.
"""
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

from langgraph.checkpoint.sqlite import SqliteSaver

FINISHED = {"saved", "rejected", "rejected_input", "failed"}


def sqlite_checkpointer(path: str) -> SqliteSaver:
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    COLUMNS = ("run_id", "customer_id", "ticker", "status", "trace_id", "error",
               "duration_ms", "cost_usd", "created_at", "updated_at")

    def __init__(self, path: str = ":memory:"):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS runs ("
                "run_id TEXT PRIMARY KEY, customer_id TEXT, ticker TEXT, status TEXT, trace_id TEXT, "
                "error TEXT, duration_ms INTEGER, cost_usd REAL, created_at TEXT, updated_at TEXT)"
            )
            self._conn.commit()

    def create(self, run_id: str, ticker: str, customer_id: str, trace_id: str, status: str = "queued") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (run_id, customer_id, ticker, status, trace_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, customer_id, ticker, status, trace_id, _now(), _now()),
            )
            self._conn.commit()

    def update(self, run_id: str, **fields) -> None:
        fields = {k: v for k, v in fields.items() if k in self.COLUMNS}
        if not fields:
            return
        assignments = ", ".join(f"{k} = ?" for k in fields) + ", updated_at = ?"
        with self._lock:
            self._conn.execute(f"UPDATE runs SET {assignments} WHERE run_id = ?",
                               (*fields.values(), _now(), run_id))
            self._conn.commit()

    def get(self, run_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {', '.join(self.COLUMNS)} FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return dict(zip(self.COLUMNS, row)) if row else None

    def metrics(self, hours: float = 24) -> dict:
        """Health of recent runs: what you watch closely after a release (hypercare)."""
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, duration_ms, cost_usd FROM runs WHERE created_at >= ?", (since,)
            ).fetchall()
        by_status: dict[str, int] = {}
        for status, _, _ in rows:
            by_status[status] = by_status.get(status, 0) + 1
        finished = [r for r in rows if r[0] in FINISHED]
        decided = by_status.get("saved", 0) + by_status.get("rejected", 0)
        durations = [r[1] for r in finished if r[1] is not None]
        costs = [r[2] for r in finished if r[2] is not None]
        return {
            "window_hours": hours,
            "runs": len(rows),
            "by_status": by_status,
            "failure_rate": round(by_status.get("failed", 0) / len(finished), 3) if finished else None,
            "approval_rate": round(by_status.get("saved", 0) / decided, 3) if decided else None,
            "avg_duration_ms": round(sum(durations) / len(durations)) if durations else None,
            "total_cost_usd": round(sum(costs), 6),
        }
