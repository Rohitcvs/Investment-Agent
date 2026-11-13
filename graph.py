"""
Multi-agent investment analysis workflow, built with LangGraph.

    START ─┬─► research ──┐
           └─► technical ─┴─► writer ─► critic ──(approved, or 3 revisions done)──► memory ─► END
                                 ▲          │
                                 └──────────┘  (needs revision)

- research and technical are each a ReAct sub-agent: the model calls tools,
  sees the results, and loops until it can write a summary.
- They run in parallel because neither depends on the other.
- writer and critic form a quality-control loop with up to 3 revisions.
- memory saves the final report so the next run for the ticker has context.
"""
import operator
from typing import Annotated, Optional, TypedDict

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from tools import (
    EconomicResearcherTool,
    FinancialReportFormatter,
    MemoryTool,
    StockDataTool,
    TechnicalIndicatorTool,
)

MAX_REVISIONS = 3

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
    research_data: Optional[str]
    technical_analysis: Optional[str]
    draft_report: Optional[str]
    criticism: Optional[str]
    final_report: Optional[str]
    revision_number: int
    log: Annotated[list[str], operator.add]  # short trace of what each agent did


def text_of(message: BaseMessage) -> str:
    """Message text, across LangChain versions (.text is a property in newer ones, a method in older)."""
    text = message.text
    return text if isinstance(text, str) else text()


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
):
    """Build the workflow. Each role can use its own model (e.g. a cheaper one for simpler agents)."""
    memory = memory or MemoryTool()
    formatter = FinancialReportFormatter()
    writer_llm = writer_model or model
    critic_llm = critic_model or model

    research_agent = create_agent(
        research_model or model,
        research_tools if research_tools is not None else [StockDataTool(), EconomicResearcherTool(), memory],
        system_prompt=RESEARCH_PROMPT,
    )
    technical_agent = create_agent(
        technical_model or model,
        technical_tools if technical_tools is not None else [TechnicalIndicatorTool()],
        system_prompt=TECHNICAL_PROMPT,
    )

    # --- Agent nodes -------------------------------------------------------

    def research_node(state: AgentState) -> dict:
        result = research_agent.invoke({"messages": [HumanMessage(f"Research the stock {state['ticker']}.")]})
        tools_used = [m.name for m in result["messages"] if m.type == "tool"]
        return {
            "research_data": text_of(result["messages"][-1]),
            "log": [f"research: used {tools_used}"],
        }

    def technical_node(state: AgentState) -> dict:
        result = technical_agent.invoke({"messages": [HumanMessage(f"Analyze the stock {state['ticker']}.")]})
        tools_used = [m.name for m in result["messages"] if m.type == "tool"]
        return {
            "technical_analysis": text_of(result["messages"][-1]),
            "log": [f"technical: used {tools_used}"],
        }

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
                f"## Editor feedback (revision {state['revision_number']} of {MAX_REVISIONS})\n"
                f"{state['criticism']}\n\nRevise the draft to address every point."
            )
        response = writer_llm.invoke([SystemMessage(WRITER_PROMPT), HumanMessage(prompt)])
        report = formatter.invoke({"data": text_of(response)})
        return {"draft_report": report, "log": [f"writer: draft {state['revision_number'] + 1}"]}

    def critic_node(state: AgentState) -> dict:
        prompt = (
            f"## Research data\n{state['research_data']}\n\n"
            f"## Technical analysis\n{state['technical_analysis']}\n\n"
            f"## Draft report\n{state['draft_report']}"
        )
        verdict = text_of(critic_llm.invoke([SystemMessage(CRITIC_PROMPT), HumanMessage(prompt)])).strip()

        if verdict.upper().startswith("APPROVED"):
            return {"final_report": state["draft_report"], "criticism": None, "log": ["critic: approved"]}
        if state["revision_number"] >= MAX_REVISIONS:
            return {"final_report": state["draft_report"], "criticism": None,
                    "log": [f"critic: revision limit ({MAX_REVISIONS}) reached, using latest draft"]}
        return {"criticism": verdict, "revision_number": state["revision_number"] + 1,
                "log": ["critic: revision requested"]}

    def memory_node(state: AgentState) -> dict:
        # Saving is deterministic, so it's a direct call rather than an LLM deciding to call a tool.
        result = memory.save(state["ticker"], state["final_report"])
        return {"log": [f"memory: {result}"]}

    def route_after_critic(state: AgentState) -> str:
        return "memory" if state.get("final_report") else "writer"

    # --- Graph -------------------------------------------------------------

    workflow = StateGraph(AgentState)
    workflow.add_node("research", research_node)
    workflow.add_node("technical", technical_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("memory", memory_node)

    workflow.add_edge(START, "research")       # two edges from START = parallel branches
    workflow.add_edge(START, "technical")
    workflow.add_edge(["research", "technical"], "writer")  # writer waits for BOTH branches
    workflow.add_edge("writer", "critic")
    workflow.add_conditional_edges("critic", route_after_critic, {"writer": "writer", "memory": "memory"})
    workflow.add_edge("memory", END)
    return workflow.compile()


def initial_state(ticker: str) -> AgentState:
    return {
        "ticker": ticker.strip().upper(),
        "research_data": None,
        "technical_analysis": None,
        "draft_report": None,
        "criticism": None,
        "final_report": None,
        "revision_number": 0,
        "log": [],
    }
