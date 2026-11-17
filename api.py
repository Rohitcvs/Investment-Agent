"""
REST API for the investment agent (FastAPI).

Run it with:  uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs to try every endpoint.

Flow:
  POST /reports                     start an analysis -> usually "awaiting_approval"
  POST /reports/{run_id}/decision   approve or reject the draft
  GET  /reports/{run_id}            check status, the report, and cost/latency metrics
  GET  /audit                       see every decision the system made
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from audit import AuditLog
from guardrails import validate_ticker
from settings import settings


class ReportRequest(BaseModel):
    ticker: str


class Decision(BaseModel):
    approved: bool
    reason: str | None = None
    reviewer: str = "reviewer"


def create_app(service, audit: AuditLog) -> FastAPI:
    app = FastAPI(title="Investment Agent API")

    @app.get("/health")
    def health():
        return {"status": "ok", "require_approval": settings.require_approval,
                "models": {"main": settings.main_model, "critic": settings.critic_model}}

    @app.post("/reports")
    def create_report(req: ReportRequest):
        ok, message = validate_ticker(req.ticker)
        if not ok:  # fail fast with a clear 400 instead of starting a run
            raise HTTPException(status_code=400, detail=message)
        return service.start(req.ticker)

    @app.post("/reports/{run_id}/decision")
    def decide(run_id: str, decision: Decision):
        try:
            return service.decide(run_id, decision.approved, decision.reason, by=decision.reviewer)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found.")
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.get("/reports/{run_id}")
    def get_report(run_id: str):
        try:
            return service.get(run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found.")

    @app.get("/audit")
    def audit_log(ticker: str | None = None, limit: int = 100):
        return audit.entries(ticker=ticker, limit=limit)

    return app


def _default_app() -> FastAPI:
    from langchain_openai import ChatOpenAI

    from graph import build_graph
    from service import AnalysisService

    audit = AuditLog(settings.audit_file)
    graph = build_graph(
        ChatOpenAI(model=settings.main_model, temperature=0),
        critic_model=ChatOpenAI(model=settings.critic_model, temperature=0),
        audit=audit,
    )
    return create_app(AnalysisService(graph), audit)


try:
    app = _default_app()
except Exception:  # e.g. no OPENAI_API_KEY yet: let tests import create_app without failing
    app = None
