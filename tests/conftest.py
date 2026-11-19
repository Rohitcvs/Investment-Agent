"""Shared test helpers: scripted fake models, so no test calls a real LLM or API."""
import sys
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit import AuditLog  # noqa: E402
from graph import build_graph  # noqa: E402
from tools import MemoryTool  # noqa: E402


class FakeModel(GenericFakeChatModel):
    """Returns pre-scripted messages in order, and records every prompt it receives."""

    def __init__(self, replies, name="fake-model", **kwargs):
        super().__init__(messages=iter(replies), **kwargs)
        object.__setattr__(self, "prompts", [])
        object.__setattr__(self, "model_name", name)

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


def usage(i, o):
    return {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}


@pytest.fixture
def make_graph(tmp_path):
    """Build the real graph with scripted models. Returns (graph, models, memory, audit)."""

    def _make(critic_replies, writer_replies, require_approval=False, research_replies=None, **graph_options):
        memory = MemoryTool(file_path=str(tmp_path / "memory.json"))
        audit = AuditLog(str(tmp_path / "audit.jsonl"))
        models = {
            "research": FakeModel(research_replies or [
                call("Memory_Tool", {"action": "retrieve", "ticker": "AAPL"}, "r1"),
                AIMessage(content="Research summary: no past reports; strong iPhone sales news. Price 189.50.",
                          usage_metadata=usage(100, 20)),
            ], name="gpt-4o"),
            "technical": FakeModel([
                call("fake_indicators", {"ticker": "AAPL"}, "t1"),
                AIMessage(content="Technical summary: RSI 61.2 shows moderate momentum.",
                          usage_metadata=usage(80, 15)),
            ], name="gpt-4o"),
            "writer": FakeModel(writer_replies, name="gpt-4o"),
            "critic": FakeModel(critic_replies, name="gpt-4o-mini"),
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
            audit=audit,
            require_approval=require_approval,
            max_revisions=3,
            **graph_options,
        )
        return graph, models, memory, audit

    return _make
