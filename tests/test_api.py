"""The REST API, end to end with scripted models."""
from fastapi.testclient import TestClient

from api import create_app
from service import AnalysisService


def test_api_full_approval_flow(make_graph):
    graph, _, _, audit = make_graph(["APPROVED"], ["Draft 1: Buy."], require_approval=True)
    client = TestClient(create_app(AnalysisService(graph), audit))

    assert client.get("/health").json()["status"] == "ok"

    run = client.post("/reports", json={"ticker": "AAPL"}).json()
    assert run["status"] == "awaiting_approval"

    done = client.post(f"/reports/{run['run_id']}/decision", json={"approved": True, "reviewer": "rohit"}).json()
    assert done["status"] == "saved"
    assert client.get(f"/reports/{run['run_id']}").json()["status"] == "saved"
    assert any(e["event"] == "report_saved" for e in client.get("/audit", params={"ticker": "AAPL"}).json())


def test_api_rejects_bad_input_and_unknown_runs(make_graph):
    graph, _, _, audit = make_graph(["APPROVED"], ["Draft"], require_approval=True)
    client = TestClient(create_app(AnalysisService(graph), audit))

    assert client.post("/reports", json={"ticker": "drop table"}).status_code == 400
    assert client.get("/reports/does-not-exist").status_code == 404
