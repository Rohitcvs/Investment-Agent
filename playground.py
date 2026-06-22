import os
import streamlit as st
from typing import TypedDict, Annotated, Optional, List
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

# Import the tools from your tools file
from tools import StockDataTool, TechnicalIndicatorTool, EconomicResearcherTool, FinancialReportFormatter, MemoryTool

"""
This script creates an advanced, multi-agent Streamlit web application 
that analyzes stocks using a dynamic workflow with planning, reflection, and memory.
"""

# --- 1. Agent & Graph Setup ---

# Load API key securely
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY environment variable not set!")
    st.stop()

# Define the state for our graph
class AgentState(TypedDict):
    """Represents the state of our multi-agent system."""
    messages: Annotated[list[BaseMessage], operator.add]
    ticker: str
    research_data: Optional[str] = None
    technical_analysis: Optional[str] = None
    draft_report: Optional[str] = None
    criticism: Optional[str] = None
    final_report: Optional[str] = None
    revision_number: int

# Instantiate the tools
research_tools = [StockDataTool(), EconomicResearcherTool(), MemoryTool()]
technical_tools = [TechnicalIndicatorTool()]
formatting_tools = [FinancialReportFormatter()]
all_tools = research_tools + technical_tools + formatting_tools

# Create specialized tool nodes
research_tool_node = ToolNode(research_tools)
technical_tool_node = ToolNode(technical_tools)
formatting_tool_node = ToolNode(formatting_tools)

# Create specialized models for each agent's role
planner_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o")
research_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o").bind_tools(research_tools)
technical_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o").bind_tools(technical_tools)
writer_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o").bind_tools(formatting_tools)
critic_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o")
memory_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o").bind_tools([MemoryTool()])

# --- Agent Nodes ---

def research_agent_node(state: AgentState) -> dict:
    """Executes research tasks (data, news, memory)."""
    prompt = (
        f"You are a research assistant. Your goal is to gather all necessary data for {state['ticker']}. "
        "Use your tools to perform the following actions: "
        "1. Retrieve any past reports from memory for long-term context. "
        "2. Fetch new historical stock data for the last year. "
        "3. Research recent news and market sentiment, **focusing strictly on sources from the last 6-12 months.** "
        "Consolidate all findings into a single, comprehensive summary."
    )
    response = research_model.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}

def technical_analyst_node(state: AgentState) -> dict:
    """Executes technical analysis tasks."""
    prompt = (
        "You are a technical analyst. Your sole job is to calculate technical indicators for the stock. "
        f"Use your tool to get the technical analysis data for {state['ticker']}."
    )
    response = technical_model.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}

def report_writer_node(state: AgentState) -> dict:
    """Writes the draft financial report."""
    # Consolidate data from parallel branches
    research_summary = state.get('research_data', "No research data provided.")
    tech_analysis_summary = state.get('technical_analysis', "No technical analysis provided.")
    
    prompt = (
        "You are a senior financial analyst. Your task is to write a comprehensive analysis based on the provided data, with a strong emphasis on information from the last 6-12 months. "
        "First, synthesize the information from the research and technical analysis into a coherent narrative. Highlight recent trends and news. "
        "Then, provide a final recommendation (e.g., Buy, Hold, Sell) based on your analysis. "
        "Finally, use the 'Financial_Report_Formatter' tool to structure your entire analysis and recommendation into a professional report. "
        "**Important Formatting Rule:** Your entire output must be clean markdown. All dollar signs must be escaped (e.g., `\\$100` instead of `$100`). Do NOT include any notes or comments about this formatting rule in the final report itself.\n\n"
        f"## Raw Research Data & News:\n{research_summary}\n\n"
        f"## Raw Technical Analysis:\n{tech_analysis_summary}\n\n"
        f"## Previous Criticisms (Revision #{state['revision_number']}):\n{state.get('criticism', 'None')}\n\n"
        "Please provide a detailed, well-structured report."
    )
    response = writer_model.invoke([HumanMessage(content=prompt)])
        
    return {"messages": [response]}

def critic_node(state: AgentState) -> dict:
    """Critiques the draft report for quality and completeness."""
    # The formatted report is in the content of the last message (a ToolMessage)
    formatted_report = state['messages'][-1].content

    prompt = (
        "You are a meticulous and demanding financial editor. Your job is to review the following draft report with a critical eye. "
        "Check for the following: "
        "1. **Recency:** Does the analysis primarily focus on data and events from the last 6-12 months? "
        "2. **Clarity & Depth:** Is the analysis clear, deep, and are all claims supported by the provided data? "
        "3. **Formatting:** Is the report clean, professional markdown? Are there any characters that could be misinterpreted by a renderer (like un-escaped dollar signs)? "
        "The report must be of the highest professional standard. "
        "If the report is perfect and requires no changes, respond ONLY with the word 'APPROVED'. "
        "Otherwise, provide specific, actionable feedback for the writer to improve it. Be tough and demand excellence."
        f"\n\n---DRAFT REPORT---\n{formatted_report}"
    )
    response = critic_model.invoke([HumanMessage(content=prompt)])
    
    revision_number = state.get('revision_number', 0) + 1
    
    # If the report is approved, set the final report and clear criticism
    if "APPROVED" in response.content.upper():
        return {"criticism": None, "final_report": formatted_report, "messages": [response]}
    
    # If the revision limit is reached, save the current draft as the final report anyway
    if revision_number >= 3:
        return {"criticism": None, "final_report": formatted_report, "messages": [response]}
        
    # Otherwise, return criticism for another revision cycle
    return {"criticism": response.content, "draft_report": None, "messages": [response], "revision_number": revision_number}

def memory_node(state: AgentState) -> dict:
    """Saves the final report to memory."""
    prompt = (
        "You are the memory archivist. Your job is to save the final report. "
        f"Use your tool to save the following report for ticker {state['ticker']}:\n\n"
        f"{state['final_report']}"
    )
    response = memory_model.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}

# --- Graph Edges & Logic ---

def route_to_tools_or_aggregate(state: AgentState) -> str:
    """
    If the last message has tool calls, route to the appropriate tool node.
    Otherwise, aggregate the results.
    """
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        # Check which agent the tool call is for
        if any(tool.name in [t.name for t in research_tools] for tool in last_message.tool_calls):
            return "tools_research"
        if any(tool.name in [t.name for t in technical_tools] for tool in last_message.tool_calls):
            return "tools_technical"
    return "aggregate_results"

def aggregate_results_node(state: AgentState) -> dict:
    """Aggregates results from parallel branches."""
    # Find the research and technical analysis tool outputs in the messages
    research_data = ""
    technical_analysis = ""
    for msg in reversed(state['messages']):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # This is naive; a real implementation would track tool call IDs
            # For now, we assume the last tool-using AIMessage from each branch is what we want
            if not research_data and any(t['name'] in [rt.name for rt in research_tools] for t in msg.tool_calls):
                 research_data = str(msg.tool_calls) # Simplified aggregation
            if not technical_analysis and any(t['name'] in [tt.name for tt in technical_tools] for t in msg.tool_calls):
                 technical_analysis = str(msg.tool_calls) # Simplified aggregation
    
    return {"research_data": research_data, "technical_analysis": technical_analysis}


def should_continue_writing(state: AgentState) -> str:
    """Decides if there are tool calls to execute."""
    return "tools_formatting" if state['messages'][-1].tool_calls else "critic"

def should_correct_report(state: AgentState) -> str:
    """Decides if the report needs revision based on criticism."""
    # If there is criticism, loop back to the writer. Otherwise, save to memory.
    if state.get("criticism"):
        return "report_writer"
    return "memory"

# --- Build the Graph ---
workflow = StateGraph(AgentState)

# The graph now starts with two parallel branches for research and technical analysis
workflow.add_node("research_agent", research_agent_node)
workflow.add_node("technical_analyst", technical_analyst_node)
workflow.add_node("tools_research", research_tool_node)
workflow.add_node("tools_technical", technical_tool_node)
workflow.add_node("aggregate_results", aggregate_results_node)
workflow.add_node("report_writer", report_writer_node)
workflow.add_node("tools_formatting", formatting_tool_node)
workflow.add_node("critic", critic_node)
workflow.add_node("memory", memory_node)

# Set the entry points for the parallel branches
workflow.add_edge(START, "research_agent")
workflow.add_edge(START, "technical_analyst")

# Edges for the research branch
workflow.add_edge("research_agent", "tools_research")
workflow.add_edge("tools_research", "aggregate_results")

# Edges for the technical analysis branch
workflow.add_edge("technical_analyst", "tools_technical")
workflow.add_edge("tools_technical", "aggregate_results")

# The rest of the workflow is sequential
workflow.add_edge("aggregate_results", "report_writer")
workflow.add_conditional_edges("report_writer", should_continue_writing, {"tools_formatting": "tools_formatting", "critic": "critic"})
workflow.add_edge("tools_formatting", "critic")
workflow.add_conditional_edges("critic", should_correct_report, {"report_writer": "report_writer", "memory": "memory"})
workflow.add_edge("memory", END)

app = workflow.compile()

# --- 2. Streamlit Web App ---

st.set_page_config(page_title="Fast Multi-Agent Stock Analysis", layout="wide")
st.title("🤖 Fast Multi-Agent Stock Analysis")

st.info(
    "Enter a stock ticker. A team of AI agents will collaborate in parallel to produce a financial analysis. "
    "The process includes parallel research and technical analysis, report writing, and streamlined self-correction."
)

ticker_input = st.text_input("Enter Stock Ticker (e.g., AAPL, GOOGL):", "TSLA")

if st.button("Run Analysis"):
    if not ticker_input:
        st.warning("Please enter a stock ticker.")
        st.stop()

    with st.spinner(f"Executing multi-agent analysis for {ticker_input}..."):
        try:
            inputs = {"ticker": ticker_input, "messages": [], "revision_number": 0}
            
            st.write("---")
            st.subheader("Agent Execution Log")
            final_report_md = ""
            
            with st.expander("Follow the agent team's progress"):
                for s in app.stream(inputs, {"recursion_limit": 100}):
                    step_name = list(s.keys())[0]
                    step_output = s[step_name]
                    
                    st.write(f"**Agent/Node:** `{step_name}`")
                    st.json(step_output, expanded=False) # Set to False for cleaner logs
                    
                    if "final_report" in step_output and step_output["final_report"]:
                        final_report_md = step_output["final_report"]

            st.write("---")
            st.subheader("Final Report")
            if final_report_md:
                # Use st.markdown and tell it to treat the content as-is.
                # The prompts now enforce clean markdown, so this should be safe.
                st.markdown(final_report_md, unsafe_allow_html=False)
            else:
                st.error("The agent team could not generate a final report. Please review the execution log.")

        except Exception as e:
            st.error(f"An error occurred during the analysis: {e}")
            st.exception(e)
