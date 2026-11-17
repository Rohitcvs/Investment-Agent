"""
Observability: structured logs, a trace ID per run, and per-step usage tracking.

Every log line is JSON and carries the run's trace_id, so you can filter the
logs for one run and see each step in order. Each node also records how long
it took and how many tokens it used, which feeds the cost estimate.

For full visual traces (every prompt, tool call and response), set
LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env. LangChain sends traces
to LangSmith automatically, with no code changes.
"""
import functools
import json
import logging
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone

from langgraph.errors import GraphBubbleUp

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")

_logger = logging.getLogger("investment_agent")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


def new_trace_id() -> str:
    trace_id = uuid.uuid4().hex[:12]
    trace_id_var.set(trace_id)
    return trace_id


def log_event(event: str, **fields) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id_var.get(),
        "event": event,
        **fields,
    }
    _logger.info(json.dumps(record, default=str))


def usage_from_messages(messages) -> dict:
    """Sum token usage across AI messages (LangChain puts it in usage_metadata)."""
    totals = {"input_tokens": 0, "output_tokens": 0}
    for m in messages:
        usage = getattr(m, "usage_metadata", None) or {}
        totals["input_tokens"] += usage.get("input_tokens", 0)
        totals["output_tokens"] += usage.get("output_tokens", 0)
    return totals


def estimate_cost(usage_records: list[dict], prices: dict) -> float:
    """Estimated USD cost from usage records like {"model": ..., "input_tokens": ..., "output_tokens": ...}."""
    total = 0.0
    for r in usage_records:
        price_in, price_out = prices.get(r.get("model"), (0.0, 0.0))
        total += r.get("input_tokens", 0) / 1e6 * price_in + r.get("output_tokens", 0) / 1e6 * price_out
    return round(total, 6)


def traced_node(name: str, model_name: str | None = None):
    """
    Wrap a graph node: log its start and end, time it, and record usage.

    The node may return a "_messages" key holding the LLM messages it produced;
    this wrapper turns them into a usage record and removes the key from the state update.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state):
            start = time.perf_counter()
            log_event("node_started", node=name, ticker=state.get("ticker"))
            try:
                update = fn(state)
            except GraphBubbleUp:
                # Not an error: LangGraph uses this to pause the graph (e.g. waiting for human approval).
                log_event("node_paused", node=name)
                raise
            except Exception as exc:
                log_event("node_failed", node=name, error=repr(exc),
                          duration_ms=round((time.perf_counter() - start) * 1000))
                raise
            duration_ms = round((time.perf_counter() - start) * 1000)
            messages = update.pop("_messages", [])
            record = {"node": name, "model": model_name, "duration_ms": duration_ms, **usage_from_messages(messages)}
            log_event("node_finished", **record)
            update.setdefault("usage", []).append(record)
            return update
        return wrapper

    return decorator
