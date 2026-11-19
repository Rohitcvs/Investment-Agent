"""Production-scale layers: skills, customer context, persistence, async runs, metrics, deployment gate."""
import json

import pytest
from fastapi.testclient import TestClient

from api import create_app
from context import CustomerContextStore
from evals.gate import check, main as gate_main
from graph import initial_state
from service import AnalysisService
from skills import Skill, get_skill, list_skills, register
from store import RunStore, sqlite_checkpointer

CONFIG = {"configurable": {"thread_id": "test"}}


# --- Versioned skills ---------------------------------------------------------

def test_registry_returns_latest_version_unless_pinned():
    assert get_skill("writer", pins={}).version == "1.1.0"
    assert get_skill("writer", pins={"writer": "1.0.0"}).version == "1.0.0"   # rollback by pinning
    assert get_skill("writer", version="1.0.0", pins={}).version == "1.0.0"
    with pytest.raises(KeyError):
        get_skill("writer", version="9.9.9", pins={})
    with pytest.raises(ValueError):  # published versions are immutable
        register(Skill("writer", "1.0.0", "duplicate", "prompt"))
    assert {"name": "critic", "version": "1.0.0"}.items() <= list_skills()[0].items()


def test_run_records_skill_versions_in_state_and_audit(make_graph):
    graph, _, _, audit = make_graph(["APPROVED"], ["Draft: Hold."], skill_versions={"writer": "1.0.0"})
    result = graph.invoke(initial_state("AAPL"), CONFIG)

    assert result["skills"]["writer"] == "1.0.0"
    started = next(e for e in audit.entries() if e["event"] == "run_started")
    assert started["details"]["skills"]["writer"] == "1.0.0"


# --- Per-customer context -----------------------------------------------------

def test_context_is_validated():
    store = CustomerContextStore()
    assert store.get("nobody") == {}
    with pytest.raises(ValueError):
        store.set("acme", {"risk_tolerance": "extreme"})
    with pytest.raises(ValueError):
        store.set("acme", {"favorite_color": "blue"})


def test_writer_and_critic_see_the_customers_context(make_graph):
    store = CustomerContextStore()
    store.set("acme", {"risk_tolerance": "low", "time_horizon": "long"})
    graph, models, _, _ = make_graph(["APPROVED"], ["Draft: Hold."], context_store=store)
    result = graph.invoke(initial_state("AAPL", customer_id="acme"), CONFIG)

    assert result["customer_context"] == {"risk_tolerance": "low", "time_horizon": "long"}
    assert "risk tolerance: low" in models["writer"].prompts[0]
    assert "time horizon: long" in models["critic"].prompts[0]


# --- Persistence --------------------------------------------------------------

def test_paused_run_survives_a_restart(make_graph, tmp_path):
    db = str(tmp_path / "agent.db")
    graph, _, memory, audit = make_graph(["APPROVED"], ["Draft: Buy."], require_approval=True,
                                         checkpointer=sqlite_checkpointer(db))
    run = AnalysisService(graph, runs=RunStore(db)).start("AAPL")
    assert run["status"] == "awaiting_approval"

    # "Restart": build a brand-new graph and service on the same database file.
    restarted_graph, _, _, _ = make_graph([], [], require_approval=True, checkpointer=sqlite_checkpointer(db))
    restarted = AnalysisService(restarted_graph, runs=RunStore(db))
    assert restarted.get(run["run_id"])["status"] == "awaiting_approval"
    assert restarted.decide(run["run_id"], True, by="rohit")["status"] == "saved"
    assert restarted.runs.get(run["run_id"])["trace_id"] == run["trace_id"]


# --- Async processing and failure handling ------------------------------------

def test_submit_returns_immediately_then_runs_in_the_background(make_graph):
    graph, _, _, _ = make_graph(["APPROVED"], ["Draft: Buy."], require_approval=True)
    service = AnalysisService(graph)

    queued = service.submit("AAPL", customer_id="acme")
    assert queued["status"] in {"queued", "running", "awaiting_approval"}
    assert service.wait(queued["run_id"])["status"] == "awaiting_approval"
    assert service.decide(queued["run_id"], True)["status"] == "saved"
    assert service.runs.get(queued["run_id"])["status"] == "saved"


def test_a_crashing_run_is_recorded_as_failed(make_graph):
    graph, _, _, _ = make_graph(["APPROVED"], [])  # writer has no scripted reply, so it raises
    service = AnalysisService(graph)

    run = service.wait(service.submit("AAPL")["run_id"])
    assert run["status"] == "failed"
    assert run["error"]
    assert service.runs.metrics()["failure_rate"] == 1.0


# --- Hypercare metrics --------------------------------------------------------

def test_metrics_summarize_recent_runs():
    runs = RunStore()
    for run_id, status, ms, cost in [("a", "saved", 1000, 0.01), ("b", "rejected", 3000, 0.03),
                                     ("c", "failed", None, None), ("d", "awaiting_approval", None, None)]:
        runs.create(run_id, ticker="AAPL", customer_id="acme", trace_id=run_id)
        runs.update(run_id, status=status, duration_ms=ms, cost_usd=cost)

    metrics = runs.metrics()
    assert metrics["runs"] == 4
    assert metrics["by_status"] == {"saved": 1, "rejected": 1, "failed": 1, "awaiting_approval": 1}
    assert metrics["failure_rate"] == round(1 / 3, 3)   # of finished runs
    assert metrics["approval_rate"] == 0.5               # of runs a person decided on
    assert metrics["avg_duration_ms"] == 2000 and metrics["total_cost_usd"] == 0.04


# --- Deployment gate ----------------------------------------------------------

def test_gate_blocks_releases_that_miss_thresholds(tmp_path):
    thresholds = {"min_completion_rate": 0.9, "max_avg_cost_usd": 0.05}
    assert check({"completion_rate": 1.0, "avg_cost_usd": 0.01}, thresholds) == []
    failures = check({"completion_rate": 0.7, "avg_cost_usd": 0.09}, thresholds)
    assert len(failures) == 2 and "completion_rate" in failures[0]
    assert check({"completion_rate": 1.0}, thresholds) == ["avg_cost_usd: missing from results"]

    good = {"completion_rate": 1.0, "recommendation_rate": 1.0, "disclaimer_rate": 1.0,
            "avg_grounding_ratio": 0.95, "avg_cost_usd": 0.01, "avg_duration_ms": 30000}
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"summary": good}))
    assert gate_main([str(path)]) == 0
    path.write_text(json.dumps({"summary": {**good, "avg_grounding_ratio": 0.4}}))
    assert gate_main([str(path)]) == 1


# --- API ----------------------------------------------------------------------

def test_api_async_runs_context_skills_and_metrics(make_graph):
    store = CustomerContextStore()
    graph, models, _, audit = make_graph(["APPROVED"], ["Draft: Hold."], require_approval=True,
                                         context_store=store)
    service = AnalysisService(graph)
    client = TestClient(create_app(service, audit, store))

    assert client.put("/customers/acme/context", json={"risk_tolerance": "low"}).status_code == 200
    assert client.put("/customers/acme/context", json={"risk_tolerance": "yolo"}).status_code == 422
    assert client.get("/customers/acme/context").json() == {"risk_tolerance": "low"}

    response = client.post("/reports", params={"async_mode": True}, json={"ticker": "AAPL", "customer_id": "acme"})
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    service.wait(run_id)
    assert client.get(f"/reports/{run_id}").json()["status"] == "awaiting_approval"
    assert "risk tolerance: low" in models["writer"].prompts[0]

    assert any(s["name"] == "writer" for s in client.get("/skills").json())
    assert client.get("/metrics").json()["runs"] == 1
    assert client.post("/reports", json={"ticker": "AAPL", "customer_id": "bad id!"}).status_code == 422
