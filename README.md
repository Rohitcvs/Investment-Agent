# Multi-Agent Investment Analysis

A stock analysis tool where specialized LLM agents work together: two agents research a stock and compute technical indicators in parallel, a writer drafts a report with a Buy/Hold/Sell call, and a critic reviews it before it's saved to memory. Built with LangGraph, with a Streamlit front end.

It's a learning project, not investment advice.

## How it works

```
START ─┬─► Research agent ──┐
       └─► Technical agent ─┴─► Writer ─► Critic ──(approved, or 3 revisions done)──► Memory ─► END
                                  ▲          │
                                  └──────────┘ (needs revision)
```

| Agent | Job | Tools |
|---|---|---|
| Research | Past reports, 1-year price summary, recent news (last 6–12 months) | Memory_Tool, Stock_Data_Fetcher, Economic_Researcher |
| Technical Analyst | RSI, MACD, SMA-20, Bollinger Bands (20-day) | Technical_Indicator_Calculator |
| Writer | Synthesizes both into a report with a Buy/Hold/Sell call | — |
| Critic | Checks recency, grounding in the data, and clarity; approves or sends feedback | — |
| Memory | Saves the final report per ticker in `report_memory.json` | MemoryTool (direct call) |

1. **Research and Technical run in parallel.** Neither depends on the other, so both start from `START`, and the writer waits for both to finish.
2. **Each of them is a ReAct sub-agent.** The model calls a tool, sees the result, and keeps going until it can write a summary grounded in that data.
3. **The writer and critic form a quality-control loop.** If the critic doesn't reply `APPROVED`, the writer revises its previous draft using the feedback, up to 3 revisions. After that, the latest draft is used, so the loop can't run forever.
4. **Memory saves the final report**, and the research agent retrieves the most recent past reports on the next run for that ticker, giving it historical context.

## Project structure

| File | Purpose |
|---|---|
| `graph.py` | The LangGraph workflow: state, agents, edges |
| `tools.py` | Data, search, formatting and memory tools |
| `playground.py` | Streamlit UI with a live log of each agent's step |
| `tests/test_workflow.py` | Tests of the whole workflow using scripted fake models (no API key needed) |

## Design decisions

- **One job per agent, with only the tools it needs.** Focused prompts are more reliable, and each agent can be debugged on its own.
- **LLMs for judgment, plain code for guarantees.** Formatting, escaping `$` signs, and saving to memory always happen the same way, so they're code, not something a prompt asks the model to do.
- **Agents only use facts their tools returned**, and the critic checks that claims are supported by the data.
- **Each role can use a different model** (`build_graph(model, writer_model=..., ...)`), e.g. a cheaper one for simpler agents.
- **The UI is separate from the workflow**, so the workflow can be tested without Streamlit or real API calls.

## Setup and running

```
pip install -r requirements.txt
# create a .env file containing: OPENAI_API_KEY=your-key-here
streamlit run playground.py
```

Type a ticker (AAPL, GOOGL, TSLA...), click **Run Analysis**, and expand the log to watch each agent's steps.

Run the tests:

```
pytest -v
```

## Ideas for next steps

- Track how recommendations change across runs for the same ticker
- More data sources besides yfinance
- Tracing with LangSmith for observability
- An evaluation set of tickers to compare report quality across prompt or model changes
