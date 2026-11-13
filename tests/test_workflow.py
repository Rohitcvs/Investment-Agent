"""Tests for the workflow, using scripted fake models instead of real LLM calls."""
import json
import sys
from pathlib import Path

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph import MAX_REVISIONS, build_graph, initial_state  # noqa: E402
from tools import FinancialReportFormatter, MemoryTool  # noqa: E402


class FakeModel(GenericFakeChatModel):
    """Returns pre-scripted messages in order, and records every prompt it receives."""

    def __init__(self, replies, **kwargs):
        super().__init__(messages=iter(replies), **kwargs)
        object.__setattr__(self, "prompts", [])

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.prompts.append("\n".join(str(m.content) for m in messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@tool
def fake_indicators(ticker: str) -> str:
    """Fake technical indicator tool."""
    return f"{ticker}: RSI_14=61.2 MACD=1.4"


def call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def make_graph(tmp_path, critic_replies, writer_replies):
    memory = MemoryTool(file_path=str(tmp_path / "memory.json"))
    models = {
        "research": FakeModel([
            call("Memory_Tool", {"action": "retrieve", "ticker": "AAPL"}, "r1"),
            AIMessage(content="Research summary: no past reports; strong iPhone sales news."),
        ]),
        "technical": FakeModel([
            call("fake_indicators", {"ticker": "AAPL"}, "t1"),
            AIMessage(content="Technical summary: RSI 61 shows moderate momentum."),
        ]),
        "writer": FakeModel(writer_replies),
        "critic": FakeModel(critic_replies),
    }
    graph = build_graph(
        models["writer"],
        research_model=models["research"],
        technical_model=models["technical"],
        writer_model=models["writer"],
        critic_model=models["critic"],
        research_tools=[memory],
        technical_tools=[fake_indicators],
        memory=memory,
    )
    return graph, models, memory


def test_agents_see_tool_results_and_writer_gets_real_data(tmp_path):
    graph, models, _ = make_graph(tmp_path, ["APPROVED"], ["Draft 1: Hold AAPL."])
    result = graph.invoke(initial_state("aapl"))

    # Fix 3: the technical agent's second call included its tool's output (the ReAct loop).
    assert "RSI_14=61.2" in models["technical"].prompts[1]
    # Fix 1: the writer received the agents' actual summaries, not tool-call requests.
    writer_prompt = models["writer"].prompts[0]
    assert "strong iPhone sales news" in writer_prompt
    assert "RSI 61 shows moderate momentum" in writer_prompt
    assert "tool_calls" not in writer_prompt
    assert "Draft 1: Hold AAPL." in result["final_report"]


def test_final_report_is_saved_to_memory(tmp_path):
    graph, _, memory = make_graph(tmp_path, ["APPROVED"], ["Draft 1: Buy."])
    graph.invoke(initial_state("AAPL"))

    # Fix 2: the memory step actually writes the file.
    saved = json.loads(Path(memory.file_path).read_text())
    assert len(saved["AAPL"]) == 1
    assert "Draft 1: Buy." in saved["AAPL"][0]
    assert "Draft 1: Buy." in memory.retrieve("AAPL")


def test_critic_feedback_reaches_writer_with_previous_draft(tmp_path):
    graph, models, _ = make_graph(
        tmp_path,
        ["Mention the RSI value explicitly.", "APPROVED"],
        ["Draft 1: Hold.", "Draft 2: Hold, RSI is 61."],
    )
    result = graph.invoke(initial_state("AAPL"))

    revision_prompt = models["writer"].prompts[1]
    assert "Mention the RSI value explicitly." in revision_prompt
    assert "Draft 1: Hold." in revision_prompt          # revises instead of starting over
    assert "Draft 2" in result["final_report"]
    assert result["revision_number"] == 1


def test_revision_loop_stops_after_max_revisions(tmp_path):
    drafts = [f"Draft {i}" for i in range(1, MAX_REVISIONS + 2)]
    graph, models, _ = make_graph(tmp_path, ["Needs work."] * (MAX_REVISIONS + 1), drafts)
    result = graph.invoke(initial_state("AAPL"))

    assert len(models["writer"].prompts) == MAX_REVISIONS + 1  # first draft + 3 revisions
    assert f"Draft {MAX_REVISIONS + 1}" in result["final_report"]
    assert any("revision limit" in line for line in result["log"])


def test_research_and_technical_run_in_parallel_before_writer(tmp_path):
    graph, _, _ = make_graph(tmp_path, ["APPROVED"], ["Draft 1"])
    steps = [list(update)[0] for update in graph.stream(initial_state("AAPL"), stream_mode="updates")]

    assert set(steps[:2]) == {"research", "technical"}
    assert steps[2:] == ["writer", "critic", "memory"]


def test_formatter_escapes_dollar_signs_once():
    report = FinancialReportFormatter().invoke({"data": "Price $100, already escaped \\$5"})
    assert "\\$100" in report and "\\\\$5" not in report
    assert "not financial advice" in report


def test_approved_must_be_the_verdict_not_just_a_word(tmp_path):
    graph, _, _ = make_graph(
        tmp_path,
        ["This is not approved yet: add news dates.", "APPROVED"],
        ["Draft 1", "Draft 2"],
    )
    result = graph.invoke(initial_state("AAPL"))
    assert "Draft 2" in result["final_report"]
