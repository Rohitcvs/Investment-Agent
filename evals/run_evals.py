"""
Run the evaluation set and compare against a previous run.

    python -m evals.run_evals                       # deterministic checks only
    python -m evals.run_evals --judge               # add the LLM judge (costs more)
    python -m evals.run_evals --baseline evals/results/run-....json

Approval is turned off here so runs complete unattended, and memory/audit
write to separate eval files so they don't mix with real reports.
"""
import argparse
import json
import os
from datetime import datetime

from audit import AuditLog
from evals.scorers import evaluate_run, summarize
from tools import MemoryTool

HERE = os.path.dirname(__file__)


def compare(current: dict, baseline: dict) -> list[str]:
    lines = []
    for key, now in current.items():
        before = baseline.get(key)
        if isinstance(now, (int, float)) and isinstance(before, (int, float)):
            lines.append(f"{key:22} {before:>10} -> {now:>10}  ({now - before:+.3f})")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", help="Override the dataset tickers")
    parser.add_argument("--judge", action="store_true", help="Score grounding with an LLM judge")
    parser.add_argument("--baseline", help="A previous results file to compare against")
    args = parser.parse_args()

    from langchain_openai import ChatOpenAI

    from graph import build_graph
    from service import AnalysisService
    from settings import settings

    with open(os.path.join(HERE, "dataset.json")) as f:
        tickers = args.tickers or json.load(f)["tickers"]

    graph = build_graph(
        ChatOpenAI(model=settings.main_model, temperature=0),
        critic_model=ChatOpenAI(model=settings.critic_model, temperature=0),
        memory=MemoryTool(file_path=os.path.join(HERE, "eval_memory.json")),
        audit=AuditLog(os.path.join(HERE, "eval_audit.jsonl")),
        require_approval=False,
    )
    service = AnalysisService(graph)
    judge = ChatOpenAI(model=settings.critic_model, temperature=0) if args.judge else None

    results = []
    for ticker in tickers:
        print(f"Running {ticker}...")
        results.append(evaluate_run(service.start(ticker), judge_model=judge))

    summary = summarize(results)
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"run-{datetime.now():%Y%m%d-%H%M%S}.json")
    with open(path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key:22} {value}")
    if args.baseline:
        with open(args.baseline) as f:
            print("\nCompared with baseline:")
            print("\n".join(compare(summary, json.load(f)["summary"])))
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
