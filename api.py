"""
REST API for the investment agent (FastAPI).

Run it with:  uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs to try every endpoint.

Flow:
  PUT  /customers/{id}/context      set a customer's risk tolerance, horizon and notes
  POST /reports                     start an analysis (?async_mode=true returns 202 right away)
  GET  /reports/{run_id}            check status, the report, skill versions, cost and latency
  POST /reports/{run_id}/decision   approve or reject the draft
  GET  /audit                       every decision the system made (admin only)
  GET  /skills                      every registered skill and version
  GET  /metrics                     health of recent runs (admin only)

Security (see security.py):
  - Send X-API-Key. The key decides the customer, so customers only see their own runs and context.
  - POST /reports is rate limited per customer (429 + Retry-After when over the limit).
  - Send an Idempotency-Key header with POST /reports so a retried request can't start a second run.
"""
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field

from audit import AuditLog
from context import CustomerContextStore
from guardrails import validate_ticker
from security import ApiKeyAuth, Principal, RateLimiter
from settings import settings
from skills import list_skills


class ReportRequest(BaseModel):
    ticker: str
    customer_id: str = Field(default="default", pattern=r"^[A-Za-z0-9_\-]{1,64}$")


class Decision(BaseModel):
    approved: bool
    reason: str | None = None
    reviewer: str = "reviewer"


class CustomerContext(BaseModel):
    risk_tolerance: str | None = None
    time_horizon: str | None = None
    notes: str | None = None


def create_app(service, audit: AuditLog, context_store: CustomerContextStore | None = None,
               auth: ApiKeyAuth | None = None, rate_limiter: RateLimiter | None = None) -> FastAPI:
    app = FastAPI(title="Investment Agent API")
    context_store = context_store or CustomerContextStore()
    auth = auth or ApiKeyAuth(settings.api_keys, settings.admin_api_keys)
    rate_limiter = rate_limiter or RateLimiter(settings.rate_limit_per_minute)

    def caller(x_api_key: str | None = Header(default=None)) -> Principal:
        principal = auth.authenticate(x_api_key)
        if principal is None:
            raise HTTPException(status_code=401, detail="Missing or invalid API key.",
                                headers={"WWW-Authenticate": "ApiKey"})
        return principal

    def admin_only(principal: Principal = Depends(caller)) -> Principal:
        if not principal.is_admin:
            raise HTTPException(status_code=403, detail="Admin access required.")
        return principal

    def own_run(run_id: str, principal: Principal) -> None:
        """404 (not 403) for another customer's run, so run IDs can't be probed."""
        try:
            owner = service.owner(run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found.")
        if not principal.can_access(owner):
            raise HTTPException(status_code=404, detail="Run not found.")

    @app.get("/health")
    def health():
        return {"status": "ok", "require_approval": settings.require_approval, "auth_enabled": auth.enabled,
                "models": {"main": settings.main_model, "critic": settings.critic_model}}

    @app.post("/reports")
    def create_report(req: ReportRequest, response: Response, async_mode: bool = False,
                      idempotency_key: str | None = Header(default=None, max_length=128),
                      principal: Principal = Depends(caller)):
        # A customer key always acts as its own customer; admins and dev mode may name one.
        customer_id = principal.customer_id or req.customer_id
        ok, message = validate_ticker(req.ticker)
        if not ok:  # fail fast with a clear 400 instead of starting a run
            raise HTTPException(status_code=400, detail=message)
        allowed, retry_after = rate_limiter.allow(customer_id)
        if not allowed:
            raise HTTPException(status_code=429, detail="Too many analyses started. Try again shortly.",
                                headers={"Retry-After": str(max(1, round(retry_after)))})
        if async_mode:
            response.status_code = 202  # accepted: poll GET /reports/{run_id} for the result
            return service.submit(req.ticker, customer_id, idempotency_key=idempotency_key)
        return service.start(req.ticker, customer_id, idempotency_key=idempotency_key)

    @app.post("/reports/{run_id}/decision")
    def decide(run_id: str, decision: Decision, principal: Principal = Depends(caller)):
        own_run(run_id, principal)
        try:
            return service.decide(run_id, decision.approved, decision.reason, by=decision.reviewer)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found.")
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.get("/reports/{run_id}")
    def get_report(run_id: str, principal: Principal = Depends(caller)):
        own_run(run_id, principal)
        try:
            return service.get(run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found.")

    @app.get("/customers/{customer_id}/context")
    def get_context(customer_id: str, principal: Principal = Depends(caller)):
        if not principal.can_access(customer_id):
            raise HTTPException(status_code=403, detail="You can only read your own context.")
        return context_store.get(customer_id)

    @app.put("/customers/{customer_id}/context")
    def set_context(customer_id: str, context: CustomerContext, principal: Principal = Depends(caller)):
        if not principal.can_access(customer_id):
            raise HTTPException(status_code=403, detail="You can only change your own context.")
        try:
            return context_store.set(customer_id, context.model_dump(exclude_none=True))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.get("/audit")
    def audit_log(ticker: str | None = None, limit: int = 100, _: Principal = Depends(admin_only)):
        return audit.entries(ticker=ticker, limit=limit)

    @app.get("/skills")
    def skills():
        return list_skills()

    @app.get("/metrics")
    def metrics(hours: float = 24, _: Principal = Depends(admin_only)):
        return service.runs.metrics(hours=hours)

    return app


def _default_app() -> FastAPI:
    from langchain_openai import ChatOpenAI

    from graph import build_graph
    from service import AnalysisService
    from store import RunStore, sqlite_checkpointer

    # Create the models first: without an API key this fails before any files are written.
    main_model = ChatOpenAI(model=settings.main_model, temperature=0)
    critic_model = ChatOpenAI(model=settings.critic_model, temperature=0)

    audit = AuditLog(settings.audit_file)
    context_store = CustomerContextStore(settings.database_file)
    graph = build_graph(
        main_model,
        critic_model=critic_model,
        audit=audit,
        context_store=context_store,
        checkpointer=sqlite_checkpointer(settings.database_file),  # paused runs survive restarts
    )
    service = AnalysisService(graph, runs=RunStore(settings.database_file))
    return create_app(service, audit, context_store)


try:
    app = _default_app()
except Exception:  # e.g. no OPENAI_API_KEY yet: let tests import create_app without failing
    app = None
