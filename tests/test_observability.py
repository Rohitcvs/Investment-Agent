"""Observability and optimization: usage tracking, cost estimates, retries, and caching."""
import pandas as pd
import pytest

import tools
from observability import estimate_cost
from resilience import TTLCache, with_retry
from service import AnalysisService
from tests.conftest import usage
from langchain_core.messages import AIMessage


def test_each_step_records_timing_and_tokens(make_graph):
    writer = [AIMessage(content="Draft 1: Buy.", usage_metadata=usage(500, 300))]
    critic = [AIMessage(content="APPROVED", usage_metadata=usage(600, 2))]
    graph, _, _, _ = make_graph(critic, writer)
    run = AnalysisService(graph).start("AAPL")

    steps = {s["node"]: s for s in run["metrics"]["steps"]}
    assert set(steps) >= {"validate", "research", "technical", "writer", "critic", "approval", "memory"}
    assert steps["writer"]["input_tokens"] == 500 and steps["critic"]["model"] == "gpt-4o-mini"
    assert run["metrics"]["input_tokens"] == 100 + 80 + 500 + 600
    assert run["metrics"]["estimated_cost_usd"] > 0
    assert run["trace_id"]


def test_cheaper_critic_model_costs_less():
    prices = {"gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60)}
    record = {"input_tokens": 1_000_000, "output_tokens": 0}
    assert estimate_cost([{**record, "model": "gpt-4o-mini"}], prices) < \
           estimate_cost([{**record, "model": "gpt-4o"}], prices)


def test_retry_recovers_from_temporary_failures():
    attempts = []

    @with_retry(attempts=3, sleep=lambda s: None)
    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("temporary")
        return "ok"

    assert flaky() == "ok" and len(attempts) == 3


def test_retry_gives_up_after_max_attempts():
    @with_retry(attempts=2, sleep=lambda s: None)
    def always_fails():
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        always_fails()


def test_cache_expires_after_ttl():
    now = [0.0]
    cache = TTLCache(ttl_seconds=60, clock=lambda: now[0])
    calls = []
    compute = lambda: calls.append(1) or len(calls)

    assert cache.get_or_compute("k", compute) == 1
    assert cache.get_or_compute("k", compute) == 1    # cached
    now[0] = 61
    assert cache.get_or_compute("k", compute) == 2    # expired, recomputed
    assert (cache.hits, cache.misses) == (1, 2)


def test_both_price_tools_share_one_download(monkeypatch):
    downloads = []
    df = pd.DataFrame({"Open": [1.0, 2.0], "High": [1.5, 2.5], "Low": [0.5, 1.5],
                       "Close": [1.2, 2.2], "Volume": [100, 200]},
                      index=pd.to_datetime(["2026-01-02", "2026-01-03"]))
    monkeypatch.setattr(tools, "_download_history", lambda t: downloads.append(t) or df)
    tools._history_cache.clear()

    tools.StockDataTool().invoke({"ticker": "AAPL"})
    tools.get_price_history("AAPL")
    assert downloads == ["AAPL"]                       # downloaded once, then served from cache
