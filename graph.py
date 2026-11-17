"""
Multi-agent investment analysis workflow, built with LangGraph.

    START ─► validate ─┬─► research ──┐
              │        └─► technical ─┴─► writer ─► critic ─(approved / cap)─► approval ─(yes)─► memory ─► END
              │                             ▲          │                          │
           (invalid)                        └──────────┘                        (no)
              ▼                            (revise)                               ▼
             END                                                                 END

Layers covered:
- Guardrails: input validation, a step limit on each sub-agent, the critic with
  a revision cap, and human approval before anything is saved.
- Observability: every node is timed and logged with the run's trace ID, and its
  token usage is recorded for cost estimates.
- Audit: every decision (input rejected, critic verdict, approval, save) is logged.
- Production: price data is retried and cached (see tools.py).
- Optimization: the critic can use a cheaper model than the writer.
"""
import operator
from typing import Annotated, Optional, TypedDict

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from audit import AuditLog
from guardrails import validate_ticker
from observability import log_event, traced_node
from settings import settings
from tools import (
    EconomicResearcherTool,
    FinancialReportFormatter,
    MemoryTool,
    StockDataTool,
    TechnicalIndicatorTool,
)

RESEARCH_PROMPT = (
    "You are a research assistant gathering data on a stock. Use your tools to: "
    "1. Retrieve past reports from memory for long-term context (Memory_Tool, action='retrieve'). "
    "2. Fetch the last year of stock data. "
    "3. Research recent news and market sentiment, focusing strictly on sources from the last 6-12 months. "
    "Then write a consolidated summary of what you found. Only use facts your tools returned."
)

TECHNICAL_PROMPT = (
    "You are a technical analyst. Use your tool to calculate technical indicators for the stock, "
    "then explain what RSI, MACD, SMA-20 and the Bollinger Bands say about its recent trend and momentum. "
    "Only use the numbers your tool returned."
)

WRITER_PROMPT = (
    "You are a senior financial analyst. Write a clear analysis using ONLY the research and technical "
    "data provided, with a strong emphasis on the last 6-12 months. Synthesize both into a coherent "
    "narrative, highlight recent trends and news, and end with a recommendation: Buy, Hold, or Sell, "
    "with the reasons. Write in clean Markdown."
)

CRITIC_PROMPT = (
    "You are a meticulous financial editor reviewing a draft report. Check: "
    "1. Recency: does it focus on data and events from the last 6-12 months? "
    "2. Grounding: is every claim supported by the research and technical data? "
    "3. Clarity: is the reasoning clear, with a definite Buy/Hold/Sell recommendation? "
    "If the report needs no changes, reply with only the word APPROVED. "
    "Otherwise, give specific, actionable feedback for the writer."
)


class AgentState(TypedDict):
    ticker: str
    status: str                       # running, rejected_input, approved, rejected, saved
    error: Optional[str]
    research_data: Optional[str]
    technical_analysis: Optional[str]
    draft_report: Optional[str]
    criticism: Optional[str]
    final_report: Optional[str]
    revision_number: int
    approval: Optional[dict]          # {"approved": bool, "by": ..., "reason": ...}
    log: Annotated[list[str], operator.add]      # short human-readable trace
    usage: Annotated[list[dict], operator.add]   # per-node timing and token usage


def text_of(message: BaseMessage) -> str:
    """Message text, across LangChain versions (.text is a property in newer ones, a method in older)."""
    text = message.text
    return text if isinstance(text, str) else text()


def model_name(model: BaseChatModel) -> str | None:
    return getattr(model, "model_name", None) or getattr(model, "model", None)


def build_graph(
    model: BaseChatModel,
    *,
    research_model: BaseChatModel | None = None,
    technical_model: BaseChatModel | None = None,
    writer_model: BaseChatModel | None = None,
    critic_model: BaseChatModel | None = None,
    research_tools: list | None = None,
    technical_tools: list | None = None,
    memory: MemoryTool | None = None,
    audit: AuditLog | None = None,
    require_approval: bool | None = None,
    max_revisions: int | None = None,
    checkpointer=None,
):
    """Build the workflow. Every dependency can be swapped, which is how the tests run without real APIs."""
    memory = memory or MemoryTool(file_path=settings.memory_file)
    audit = audit or AuditLog(settings.audit_file)
    require_approval = settings.require_approval if require_approval is None else require_approval
    max_revisions = settings.max_revisions if max_revisions is None else max_revisions
    formatter = FinancialReportFormatter()

    research_llm = research_model or model
    technical_llm = technical_model or model
    writer_llm = writer_model or model
    critic_llm = critic_model or model
    agent_config = {"recursion_limit": settings.agent_step_limit}  # guardrail: caps each sub-agent's loop

    research_agent = create_agent(
        research_llm,
        research_tools if research_tools is not None else [StockDataTool(), EconomicResearcherTool(), memory],
        system_prompt=RESEARCH_PROMPT,
    )
    technical_agent = create_agent(
        technical_llm,
        technical_tools if technical_tools is not None else [TechnicalIndicatorTool()],
        system_prompt=TECHNICAL_PROMPT,
    )

    def run_sub_agent(agent, request: str, label: str) -> dict:
        """Run a ReAct sub-agent and return its summary, tools used, and messages (for usage)."""
        try:
            result = agent.invoke({"messages": [HumanMessage(request)]}, config=agent_config)
        except GraphRecursionError:
            log_event("step_limit_reached", agent=label, limit=settings.agent_step_limit)
            return {"summary": f"{label} stopped after {settings.agent_step_limit} steps without finishing.",
                    "tools": [], "messages": []}
        messages = result["messages"]
        return {"summary": text_of(messages[-1]),
                "tools": [m.name for m in messages if m.type == "tool"],
                "messages": messages}

    # --- Nodes -------------------------------------------------------------

    @traced_node("validate")
    def validate_node(state: AgentState) -> dict:
        audit.record("run_started", ticker=state["ticker"], require_approval=require_approval)
        ok, result = validate_ticker(state["ticker"])
        if not ok:
            audit.record("input_rejected", ticker=state["ticker"], reason=result)
            return {"status": "rejected_input", "error": result, "log": [f"validate: rejected ({result})"]}
        return {"ticker": result, "status": "running", "log": ["validate: ok"]}

    @traced_node("research", model_name(research_llm))
    def research_node(state: AgentState) -> dict:
        out = run_sub_agent(research_agent, f"Research the stock {state['ticker']}.", "Research agent")
        return {"research_data": out["summary"], "log": [f"research: used {out['tools']}"],
                "_messages": out["messages"]}

    @traced_node("technical", model_name(technical_llm))
    def technical_node(state: AgentState) -> dict:
        out = run_sub_agent(technical_agent, f"Analyze the stock {state['ticker']}.", "Technical agent")
        return {"technical_analysis": out["summary"], "log": [f"technical: used {out['tools']}"],
                "_messages": out["messages"]}

    @traced_node("writer", model_name(writer_llm))
    def writer_node(state: AgentState) -> dict:
        prompt = (
            f"Stock: {state['ticker']}\n\n"
            f"## Research data and news\n{state['research_data']}\n\n"
            f"## Technical analysis\n{state['technical_analysis']}\n"
        )
        if state.get("criticism"):
            # Revise the previous draft instead of starting over.
            prompt += (
                f"\n## Your previous draft\n{state['draft_report']}\n\n"
                f"## Editor feedback (revision {state['revision_number']} of {max_revisions})\n"
                f"{state['criticism']}\n\nRevise the draft to address every point."
            )
        response = writer_llm.invoke([SystemMessage(WRITER_PROMPT), HumanMessage(prompt)])
        report = formatter.invoke({"data": text_of(response)})
        return {"draft_report": report, "log": [f"writer: draft {state['revision_number'] + 1}"],
                "_messages": [response]}

    @traced_node("critic", model_name(critic_llm))
    def critic_node(state: AgentState) -> dict:
        prompt = (
            f"## Research data\n{state['research_data']}\n\n"
            f"## Technical analysis\n{state['technical_analysis']}\n\n"
            f"## Draft report\n{state['draft_report']}"
        )
        response = critic_llm.invoke([SystemMessage(CRITIC_PROMPT), HumanMessage(prompt)])
        verdict = text_of(response).strip()

        if verdict.upper().startswith("APPROVED"):
            audit.record("critic_decision", ticker=state["ticker"], decision="approved",
                         revisions=state["revision_number"])
            return {"final_report": state["draft_report"], "criticism": None,
                    "log": ["critic: approved"], "_messages": [response]}
        if state["revision_number"] >= max_revisions:
            audit.record("critic_decision", ticker=state["ticker"], decision="revision_limit_reached",
                         revisions=state["revision_number"])
            return {"final_report": state["draft_report"], "criticism": None,
                    "log": [f"critic: revision limit ({max_revisions}) reached, using latest draft"],
                    "_messages": [response]}
        audit.record("critic_decision", ticker=state["ticker"], decision="revise",
                     revisions=state["revision_number"], feedback=verdict[:500])
        return {"criticism": verdict, "revision_number": state["revision_number"] + 1,
                "log": ["critic: revision requested"], "_messages": [response]}

    @traced_node("approval")
    def approval_node(state: AgentState) -> dict:
        if not require_approval:
            decision = {"approved": True, "by": "auto", "reason": "approval not required in this environment"}
        else:
            # Human-in-the-loop: pause here until someone resumes the graph with a decision.
            # Nothing before this line has side effects, because the node re-runs when resumed.
            answer = interrupt({
                "ticker": state["ticker"],
                "report": state["final_report"],
                "question": "Approve this report before it's saved and published?",
            })
            decision = {"approved": bool(answer.get("approved")),
                        "by": answer.get("by", "reviewer"),
                        "reason": answer.get("reason")}
        audit.record("approval_decision", ticker=state["ticker"], **decision)
        status = "approved" if decision["approved"] else "rejected"
        return {"approval": decision, "status": status, "log": [f"approval: {status} by {decision['by']}"]}

    @traced_node("memory")
    def memory_node(state: AgentState) -> dict:
        # Saving must always happen once approved, so it's plain code, not an LLM's choice.
        result = memory.save(state["ticker"], state["final_report"])
        audit.record("report_saved", ticker=state["ticker"])
        return {"status": "saved", "log": [f"memory: {result}"]}

    # --- Routing -----------------------------------------------------------

    def route_after_validate(state: AgentState):
        if state["status"] == "rejected_input":
            return END
        return ["research", "technical"]  # fan out: both run in parallel

    def route_after_critic(state: AgentState) -> str:
        return "approval" if state.get("final_report") else "writer"

    def route_after_approval(state: AgentState) -> str:
        return "memory" if state["approval"]["approved"] else END

    # --- Graph -------------------------------------------------------------

    workflow = StateGraph(AgentState)
    workflow.add_node("validate", validate_node)
    workflow.add_node("research", research_node)
    workflow.add_node("technical", technical_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("approval", approval_node)
    workflow.add_node("memory", memory_node)

    workflow.add_edge(START, "validate")
    workflow.add_conditional_edges("validate", route_after_validate, ["research", "technical", END])
    workflow.add_edge(["research", "technical"], "writer")  # writer waits for BOTH branches
    workflow.add_edge("writer", "critic")
    workflow.add_conditional_edges("critic", route_after_critic, ["writer", "approval"])
    workflow.add_conditional_edges("approval", route_after_approval, ["memory", END])
    workflow.add_edge("memory", END)

    # A checkpointer saves the graph's state after every step. It's what lets a run pause for
    # approval and resume later, and lets us look up any run's status by its ID.
    # MemorySaver keeps it in memory; production would use a database-backed checkpointer.
    if checkpointer is None:
        checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def initial_state(ticker: str) -> AgentState:
    return {
        "ticker": (ticker or "").strip().upper(),
        "status": "running",
        "error": None,
        "research_data": None,
        "technical_analysis": None,
        "draft_report": None,
        "criticism": None,
        "final_report": None,
        "revision_number": 0,
        "approval": None,
        "log": [],
        "usage": [],
    }
