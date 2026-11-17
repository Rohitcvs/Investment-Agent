"""
The single entry point for running an analysis.

The API, the Streamlit app and the evaluation runner all go through this
class, so they behave the same way: each run gets an ID and a trace ID, runs
until it finishes or pauses for approval, and can be resumed with a decision.
"""
import uuid

from langgraph.types import Command

from graph import initial_state
from observability import estimate_cost, log_event, new_trace_id, trace_id_var
from settings import settings


class AnalysisService:
    def __init__(self, graph, prices: dict | None = None):
        self.graph = graph
        self.prices = prices if prices is not None else settings.prices
        self._trace_ids: dict[str, str] = {}

    def _config(self, run_id: str) -> dict:
        return {"configurable": {"thread_id": run_id}}

    def start(self, ticker: str) -> dict:
        run_id = uuid.uuid4().hex[:10]
        self._trace_ids[run_id] = new_trace_id()
        log_event("run_started", run_id=run_id, ticker=ticker)
        self.graph.invoke(initial_state(ticker), config=self._config(run_id))
        return self.get(run_id)

    def decide(self, run_id: str, approved: bool, reason: str | None = None, by: str = "reviewer") -> dict:
        if self.get(run_id)["status"] != "awaiting_approval":
            raise ValueError(f"Run {run_id} is not waiting for approval.")
        trace_id_var.set(self._trace_ids.get(run_id, "-"))
        self.graph.invoke(Command(resume={"approved": approved, "reason": reason, "by": by}),
                          config=self._config(run_id))
        return self.get(run_id)

    def get(self, run_id: str) -> dict:
        snapshot = self.graph.get_state(self._config(run_id))
        state = snapshot.values
        if not state:
            raise KeyError(run_id)
        interrupts = [i.value for task in snapshot.tasks for i in task.interrupts]
        status = "awaiting_approval" if interrupts else state.get("status")
        usage = state.get("usage", [])
        return {
            "run_id": run_id,
            "trace_id": self._trace_ids.get(run_id),
            "ticker": state.get("ticker"),
            "status": status,
            "error": state.get("error"),
            "report": state.get("final_report"),
            "research_data": state.get("research_data"),
            "technical_analysis": state.get("technical_analysis"),
            "revisions": state.get("revision_number", 0),
            "approval": state.get("approval"),
            "log": state.get("log", []),
            "metrics": {
                "total_duration_ms": sum(u.get("duration_ms", 0) for u in usage),
                "input_tokens": sum(u.get("input_tokens", 0) for u in usage),
                "output_tokens": sum(u.get("output_tokens", 0) for u in usage),
                "estimated_cost_usd": estimate_cost(usage, self.prices),
                "steps": usage,
            },
        }
