import os

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from graph import build_graph, initial_state

"""Streamlit front end for the multi-agent stock analysis workflow (see graph.py)."""

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY environment variable not set!")
    st.stop()


@st.cache_resource
def get_app():
    return build_graph(ChatOpenAI(model="gpt-4o", temperature=0))


st.set_page_config(page_title="Multi-Agent Stock Analysis", layout="wide")
st.title("🤖 Multi-Agent Stock Analysis")
st.info(
    "Enter a stock ticker. A team of AI agents researches it and computes technical indicators in parallel, "
    "then a writer drafts a report and a critic reviews it (up to 3 revisions) before it's saved to memory."
)

ticker_input = st.text_input("Enter Stock Ticker (e.g., AAPL, GOOGL):", "TSLA")

if st.button("Run Analysis"):
    if not ticker_input.strip():
        st.warning("Please enter a stock ticker.")
        st.stop()

    final_report_md = ""
    with st.spinner(f"Running multi-agent analysis for {ticker_input.upper()}..."):
        try:
            st.subheader("Agent Execution Log")
            with st.expander("Follow the agent team's progress"):
                for update in get_app().stream(initial_state(ticker_input), stream_mode="updates"):
                    for node, output in update.items():
                        st.write(f"**Agent:** `{node}`")
                        for line in output.get("log", []):
                            st.write(f"- {line}")
                        if output.get("final_report"):
                            final_report_md = output["final_report"]
        except Exception as e:
            st.error(f"An error occurred during the analysis: {e}")
            st.exception(e)

    st.subheader("Final Report")
    if final_report_md:
        st.markdown(final_report_md)
    else:
        st.error("The agent team could not generate a final report. Please review the execution log.")
