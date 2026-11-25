"""
Scorers that measure report quality.

Cheap, deterministic checks run on every report. The LLM judge is optional,
costs tokens, and gives a fuller grounding score.
"""
import re

from pydantic import BaseModel, Field

from guardrails import extract_recommendation  # noqa: F401  (shared with the runtime output check)
from tools import DISCLAIMER

_FIGURE = re.compile(r"-?\d[\d,]*\.\d+%?|-?\d[\d,]*%")  # figures like 61.2, 1,234.50 or 12%


def has_disclaimer(report: str | None) -> bool:
    return bool(report) and DISCLAIMER in report


def _figures(text: str) -> set[str]:
    return {m.replace(",", "").rstrip("%") for m in _FIGURE.findall(text or "")}


def grounding_ratio(report: str | None, sources: list[str | None]) -> float | None:
    """
    Share of figures in the report that also appear in the agents' source data.

    A quick proxy for "did the writer use real numbers, or make them up?".
    Returns None when the report quotes no figures.
    """
    report_figures = _figures(report)
    if not report_figures:
        return None
    source_figures = set().union(*(_figures(s) for s in sources))
    return round(len(report_figures & source_figures) / len(report_figures), 3)


class JudgeVerdict(BaseModel):
    score: int = Field(ge=1, le=5, description="1 = mostly unsupported claims, 5 = every claim supported by the data")
    reason: str = Field(description="One or two sentences explaining the score")


JUDGE_PROMPT = (
    "You are grading a stock report for grounding. Compare every factual claim in the REPORT "
    "with the SOURCE DATA. Score 5 if every claim is supported, 1 if most claims are unsupported "
    "or contradicted. Judge only grounding, not writing style."
)


def llm_judge(model, report: str, sources: list[str | None]) -> dict:
    judge = model.with_structured_output(JudgeVerdict)
    source_text = "\n\n".join(s for s in sources if s)
    verdict = judge.invoke(f"{JUDGE_PROMPT}\n\nSOURCE DATA:\n{source_text}\n\nREPORT:\n{report}")
    return verdict.model_dump()


def evaluate_run(run: dict, judge_model=None) -> dict:
    sources = [run.get("research_data"), run.get("technical_analysis")]
    report = run.get("report")
    result = {
        "ticker": run.get("ticker"),
        "status": run.get("status"),
        "recommendation": extract_recommendation(report),
        "has_disclaimer": has_disclaimer(report),
        "grounding_ratio": grounding_ratio(report, sources),
        "revisions": run.get("revisions", 0),
        "duration_ms": run["metrics"]["total_duration_ms"],
        "tokens": run["metrics"]["input_tokens"] + run["metrics"]["output_tokens"],
        "estimated_cost_usd": run["metrics"]["estimated_cost_usd"],
    }
    if judge_model is not None and report:
        result["judge"] = llm_judge(judge_model, report, sources)
    return result


def summarize(results: list[dict]) -> dict:
    def avg(key):
        values = [r[key] for r in results if r.get(key) is not None]
        return round(sum(values) / len(values), 3) if values else None

    n = len(results) or 1
    judge_scores = [r["judge"]["score"] for r in results if r.get("judge")]
    return {
        "runs": len(results),
        "completion_rate": round(sum(r["status"] == "saved" for r in results) / n, 3),
        "recommendation_rate": round(sum(r["recommendation"] is not None for r in results) / n, 3),
        "disclaimer_rate": round(sum(r["has_disclaimer"] for r in results) / n, 3),
        "avg_grounding_ratio": avg("grounding_ratio"),
        "avg_revisions": avg("revisions"),
        "avg_duration_ms": avg("duration_ms"),
        "avg_tokens": avg("tokens"),
        "avg_cost_usd": avg("estimated_cost_usd"),
        "avg_judge_score": round(sum(judge_scores) / len(judge_scores), 2) if judge_scores else None,
    }
