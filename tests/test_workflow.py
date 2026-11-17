"""The core workflow: data flow, the critic loop, memory, and parallel execution."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_core.messages import AIMessage

from graph import initial_state
from tools import FinancialReportFormatter, MemoryTool

CONFIG = {"configurable": {"thread_id": "test"}}


def test_agents_see_tool_results_and_writer_gets_real_data(make_graph):
    graph, models, _, _ = make_graph(["APPROVED"], ["Draft 1: Hold AAPL."])
    result = graph.invoke(initial_state("aapl"), CONFIG)

    assert "RSI_14=61.2" in models["technical"].prompts[1]   # the agent read its tool's output
    writer_prompt = models["writer"].prompts[0]
    assert "strong iPhone sales news" in writer_prompt         # writer got real summaries,
    assert "RSI 61.2 shows moderate momentum" in writer_prompt
    assert "tool_calls" not in writer_prompt                   # not tool-call requests
    assert "Draft 1: Hold AAPL." in result["final_report"]


def test_final_report_is_saved_to_memory(make_graph):
    graph, _, memory, _ = make_graph(["APPROVED"], ["Draft 1: Buy."])
    result = graph.invoke(initial_state("AAPL"), CONFIG)

    saved = json.loads(Path(memory.file_path).read_text())
    assert "Draft 1: Buy." in saved["AAPL"][0]
    assert result["status"] == "saved"


def test_critic_feedback_reaches_writer_with_previous_draft(make_graph):
    graph, models, _, _ = make_graph(["Mention the RSI value.", "APPROVED"],
                                     ["Draft 1: Hold.", "Draft 2: Hold, RSI is 61."])
    result = graph.invoke(initial_state("AAPL"), CONFIG)

    revision_prompt = models["writer"].prompts[1]
    assert "Mention the RSI value." in revision_prompt
    assert "Draft 1: Hold." in revision_prompt                 # revises instead of starting over
    assert "Draft 2" in result["final_report"]
    assert result["revision_number"] == 1


def test_revision_loop_stops_after_max_revisions(make_graph):
    graph, models, _, _ = make_graph(["Needs work."] * 4, [f"Draft {i}" for i in range(1, 5)])
    result = graph.invoke(initial_state("AAPL"), CONFIG)

    assert len(models["writer"].prompts) == 4                  # first draft + 3 revisions
    assert "Draft 4" in result["final_report"]
    assert any("revision limit" in line for line in result["log"])


def test_research_and_technical_run_in_parallel_before_writer(make_graph):
    graph, _, _, _ = make_graph(["APPROVED"], ["Draft 1"])
    steps = [list(u)[0] for u in graph.stream(initial_state("AAPL"), CONFIG, stream_mode="updates")]

    assert steps[0] == "validate"
    assert set(steps[1:3]) == {"research", "technical"}
    assert steps[3:] == ["writer", "critic", "approval", "memory"]


def test_approved_must_be_the_verdict_not_just_a_word(make_graph):
    graph, _, _, _ = make_graph(["This is not approved yet: add dates.", "APPROVED"], ["Draft 1", "Draft 2"])
    result = graph.invoke(initial_state("AAPL"), CONFIG)
    assert "Draft 2" in result["final_report"]


def test_formatter_escapes_dollar_signs_once():
    report = FinancialReportFormatter().invoke({"data": "Price $100, already escaped \\$5"})
    assert "\\$100" in report and "\\\\$5" not in report
    assert "not financial advice" in report


def test_concurrent_memory_saves_are_not_lost(tmp_path):
    memory = MemoryTool(file_path=str(tmp_path / "memory.json"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: memory.save("AAPL", f"report {i}"), range(50)))

    saved = json.loads(Path(memory.file_path).read_text())
    assert sorted(saved["AAPL"]) == sorted(f"report {i}" for i in range(50))
