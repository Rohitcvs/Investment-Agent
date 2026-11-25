"""
The single entry point for running an analysis.

The API, the Streamlit app and the evaluation runner all go through this
class, so they behave the same way: each run gets an ID and a trace ID, runs
until it finishes or pauses for approval, and can be resumed with a decision.

Runs can be started two ways:
- start():  synchronous, returns when the run finishes or pauses (simple scripts, evals).
- submit(): asynchronous, returns a run ID immediately and runs on a worker thread,
  so an API request never has to wait minutes for LLM calls. Production would put
  these jobs on a real queue (SQS, Celery, etc.); the interface stays the same.

Every run is also recorded in a RunStore, so its status, trace ID, cost and
latency survive restarts and feed the /metrics endpoint.
"""
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from langgraph.types import Command

from graph import initial_state
from observability import estimate_cost, log_event, new_trace_id, trace_id_var
from settings import settings
from store import FINISHED, RunStore


class AnalysisService:
    def __init__(self, graph, prices: dict | None = None, runs: RunStore | None = None,
                 worker_threads: int | None = None):
        self.graph = graph
        self.prices = prices if prices is not None else settings.prices
        self.runs = runs or RunStore()
        self._executor = ThreadPoolExecutor(max_workers=worker_threads or settings.worker_threads,
                                            thread_name_prefix="analysis")

    def _config(self, run_id: str) -> dict:
        return {"configurable": {"thread_id": run_id}}

    def _new_run(self, ticker: str, customer_id: str, idempotency_key: str | None = None) -> tuple[str, bool]:
        """Create a run record. Returns (run_id, created); created is False for a repeated idempotency key."""
        new_id = uuid.uuid4().hex[:10]
        run_id = self.runs.create(new_id, ticker=(ticker or "").strip().upper(), customer_id=customer_id,
                                  trace_id=uuid.uuid4().hex[:12], idempotency_key=idempotency_key)
        if run_id != new_id:
            log_event("idempotent_replay", run_id=run_id, customer_id=customer_id)
        return run_id, run_id == new_id

    def _execute(self, run_id: str, payload) -> None:
        """Run (or resume) the graph, then record the outcome. Failures are recorded, never lost."""
        record = self.runs.get(run_id)
        trace_id_var.set(record["trace_id"])  # context vars don't cross threads, so set it here
        self.runs.update(run_id, status="running")
        log_event("run_executing", run_id=run_id, ticker=record["ticker"])
        try:
            self.graph.invoke(payload, config=self._config(run_id))
        except Exception as exc:  # noqa: BLE001  (record any failure instead of crashing the worker)
            log_event("run_failed", run_id=run_id, error=repr(exc))
            self.runs.update(run_id, status="failed", error=repr(exc))
            return
        run = self.get(run_id)
        self.runs.update(run_id, status=run["status"], error=run["error"],
                         duration_ms=run["metrics"]["total_duration_ms"],
                         cost_usd=run["metrics"]["estimated_cost_usd"])

    # --- Starting runs -------------------------------------------------------

    def start(self, ticker: str, customer_id: str = "default", idempotency_key: str | None = None) -> dict:
        """Run synchronously and return the result."""
        run_id, created = self._new_run(ticker, customer_id, idempotency_key)
        if created:  # a repeated key returns the original run instead of paying for a second one
            new_trace_id()
            self._execute(run_id, initial_state(ticker, customer_id))
        return self.get(run_id)

    def submit(self, ticker: str, customer_id: str = "default", idempotency_key: str | None = None) -> dict:
        """Queue the run on a background worker and return its ID right away."""
        run_id, created = self._new_run(ticker, customer_id, idempotency_key)
        if created:
            log_event("run_queued", run_id=run_id, ticker=ticker)
            self._executor.submit(self._execute, run_id, initial_state(ticker, customer_id))
        return self.get(run_id)

    def owner(self, run_id: str) -> str | None:
        """The customer a run belongs to (used by the API to keep customers' runs apart)."""
        record = self.runs.get(run_id)
        if record is None:
            raise KeyError(run_id)
        return record["customer_id"]

    def decide(self, run_id: str, approved: bool, reason: str | None = None, by: str = "reviewer") -> dict:
        if self.get(run_id)["status"] != "awaiting_approval":
            raise ValueError(f"Run {run_id} is not waiting for approval.")
        self._execute(run_id, Command(resume={"approved": approved, "reason": reason, "by": by}))
        return self.get(run_id)

    def wait(self, run_id: str, timeout: float = 30.0) -> dict:
        """Poll until a queued run finishes or pauses for approval (used by tests and scripts)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            run = self.get(run_id)
            if run["status"] in FINISHED or run["status"] == "awaiting_approval":
                return run
            time.sleep(0.05)
        raise TimeoutError(f"Run {run_id} did not finish within {timeout}s")

    # --- Reading runs --------------------------------------------------------

    def get(self, run_id: str) -> dict:
        record = self.runs.get(run_id)
        snapshot = self.graph.get_state(self._config(run_id))
        state = snapshot.values
        if not state:
            if record is None:
                raise KeyError(run_id)
            # Queued, running, or failed before the graph saved anything.
            return {"run_id": run_id, "trace_id": record["trace_id"], "ticker": record["ticker"],
                    "customer_id": record["customer_id"], "status": record["status"],
                    "error": record["error"], "report": None}

        interrupts = [i.value for task in snapshot.tasks for i in task.interrupts]
        status = "awaiting_approval" if interrupts else state.get("status")
        if (record and record["status"] in ("queued", "running", "failed")
                and status not in FINISHED and status != "awaiting_approval"):
            status = record["status"]  # still executing (graph state not written yet), or crashed mid-run
        usage = state.get("usage", [])
        return {
            "run_id": run_id,
            "trace_id": record["trace_id"] if record else None,
            "ticker": state.get("ticker"),
            "customer_id": state.get("customer_id"),
            "status": status,
            "error": state.get("error") or (record or {}).get("error"),
            "report": state.get("final_report"),
            "research_data": state.get("research_data"),
            "technical_analysis": state.get("technical_analysis"),
            "revisions": state.get("revision_number", 0),
            "approval": state.get("approval"),
            "warnings": state.get("warnings", []),
            "skills": state.get("skills"),
            "log": state.get("log", []),
            "metrics": {
                "total_duration_ms": sum(u.get("duration_ms", 0) for u in usage),
                "input_tokens": sum(u.get("input_tokens", 0) for u in usage),
                "output_tokens": sum(u.get("output_tokens", 0) for u in usage),
                "estimated_cost_usd": estimate_cost(usage, self.prices),
                "steps": usage,
            },
        }

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
