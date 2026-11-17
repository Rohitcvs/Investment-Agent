"""Evaluation scorers."""
from evals.scorers import evaluate_run, extract_recommendation, grounding_ratio, has_disclaimer, summarize
from tools import FinancialReportFormatter


def test_extract_recommendation_uses_the_final_call():
    assert extract_recommendation("Earlier analysts said Sell. Recommendation: **Buy**.") == "Buy"
    assert extract_recommendation("No call made.") is None


def test_grounding_ratio_flags_made_up_numbers():
    sources = ["RSI 61.2, latest close 189.50, 1-year return 12.4%"]
    assert grounding_ratio("RSI is 61.2 and the return was 12.4%.", sources) == 1.0
    assert grounding_ratio("RSI is 61.2 but the price target is 250.00.", sources) == 0.5
    assert grounding_ratio("Strong momentum overall.", sources) is None


def test_evaluate_and_summarize():
    report = FinancialReportFormatter().invoke({"data": "RSI is 61.2. Recommendation: Hold."})
    run = {"ticker": "AAPL", "status": "saved", "report": report, "revisions": 1,
           "research_data": "close 189.50", "technical_analysis": "RSI 61.2",
           "metrics": {"total_duration_ms": 1200, "input_tokens": 1000, "output_tokens": 200,
                       "estimated_cost_usd": 0.004}}
    result = evaluate_run(run)
    assert result["recommendation"] == "Hold" and has_disclaimer(report)
    assert result["grounding_ratio"] == 1.0

    summary = summarize([result, {**result, "status": "rejected", "recommendation": None}])
    assert summary["completion_rate"] == 0.5 and summary["recommendation_rate"] == 0.5
