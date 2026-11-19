"""
Streamlit front end for the multi-agent stock analysis workflow.

Run it with:  streamlit run playground.py

The run pauses after the critic approves a draft, and waits for you to
approve or reject it before anything is saved (human-in-the-loop).
"""
import os

import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI

from audit import AuditLog
from context import CustomerContextStore
from graph import build_graph
from service import AnalysisService
from settings import settings
from store import RunStore, sqlite_checkpointer

if not os.environ.get("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY environment variable not set! Add it to a .env file.")
    st.stop()


@st.cache_resource
def get_service() -> AnalysisService:
    # Cached so the graph and its checkpointer survive Streamlit reruns,
    # which is what lets a paused run be resumed after a button click.
    graph = build_graph(
        ChatOpenAI(model=settings.main_model, temperature=0),
        critic_model=ChatOpenAI(model=settings.critic_model, temperature=0),
        audit=AuditLog(settings.audit_file),
        context_store=CustomerContextStore(settings.database_file),
        checkpointer=sqlite_checkpointer(settings.database_file),
    )
    return AnalysisService(graph, runs=RunStore(settings.database_file))


def show_metrics(run: dict) -> None:
    m = run["metrics"]
    cols = st.columns(4)
    cols[0].metric("Total time", f"{m['total_duration_ms'] / 1000:.1f}s")
    cols[1].metric("Tokens", f"{m['input_tokens'] + m['output_tokens']:,}")
    cols[2].metric("Est. cost", f"${m['estimated_cost_usd']:.4f}")
    cols[3].metric("Revisions", run["revisions"])
    with st.expander("Step-by-step timing and tokens"):
        st.dataframe(pd.DataFrame(m["steps"]), hide_index=True)
    st.caption(f"Trace ID: `{run['trace_id']}` · Run ID: `{run['run_id']}`")


st.set_page_config(page_title="Multi-Agent Stock Analysis", layout="wide")
st.title("🤖 Multi-Agent Stock Analysis")
st.info(
    "A research agent and a technical analyst run in parallel, a writer drafts a report, "
    "a critic reviews it (up to 3 revisions), and you approve it before it's saved."
)

service = get_service()
ticker = st.text_input("Stock ticker (e.g., AAPL, MSFT):", "TSLA")

if st.button("Run Analysis"):
    with st.spinner(f"Running multi-agent analysis for {ticker.upper()}..."):
        try:
            st.session_state.run = service.start(ticker)
        except Exception as e:
            st.error(f"The analysis failed: {e}")

run = st.session_state.get("run")
if run:
    if run["status"] == "rejected_input":
        st.error(run["error"])
    else:
        show_metrics(run)
        with st.expander("Agent log"):
            for line in run["log"]:
                st.write(f"- {line}")

        if run["status"] == "awaiting_approval":
            st.subheader("Draft report: awaiting your approval")
            st.markdown(run["report"])
            reason = st.text_input("Reason (optional, recorded in the audit log)")
            approve, reject = st.columns(2)
            if approve.button("✅ Approve and save", use_container_width=True):
                st.session_state.run = service.decide(run["run_id"], True, reason or None, by="streamlit-user")
                st.rerun()
            if reject.button("❌ Reject", use_container_width=True):
                st.session_state.run = service.decide(run["run_id"], False, reason or None, by="streamlit-user")
                st.rerun()
        elif run["status"] == "saved":
            st.success("Approved and saved to memory.")
            st.markdown(run["report"])
        elif run["status"] == "rejected":
            st.warning("Report rejected. It was not saved.")
            st.markdown(run["report"])
