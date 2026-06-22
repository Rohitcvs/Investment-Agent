# Multi-Agent Investment Analysis

A stock analysis tool that runs a few specialized LLM agents in parallel to pull together technical indicators, recent news, and a written summary with a Buy/Hold/Sell call at the end. Built on LangGraph with a Streamlit front end.

It's a learning project, not investment advice.

## Project structure

### tools.py

Core tools used by the AI agents to perform various analysis tasks:

**StockDataTool**
- Purpose: Fetches historical stock data for any given ticker symbol
- Data Source: Yahoo Finance (yfinance)
- Time Period: 1 year of historical data
- Output: Formatted DataFrame with OHLCV data

**TechnicalIndicatorTool**
- Purpose: Calculates various technical indicators for stock analysis
- Indicators Included: RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence), SMA (Simple Moving Average - 20 period), Bollinger Bands
- Library: pandas-ta for technical analysis calculations
- Output: Last 5 rows of data with calculated indicators

**EconomicResearcherTool**
- Purpose: Performs internet searches for financial news and market sentiment
- Search Engine: DuckDuckGo (via ddgs library)
- Focus: Recent news and analysis from the last 6-12 months
- Output: Formatted search results with titles, snippets, and URLs
- Sources: Prioritizes trusted financial sources over forums/blogs

**FinancialReportFormatter**
- Purpose: Formats collected data into professional financial reports
- Format: Clean Markdown with proper escaping
- Structure: Includes analysis summary and disclaimer
- Output: Professional-grade financial report

**FinancialReportEditor**
- Purpose: Acts as a lead editor to refine and improve report quality
- Role: Ensures consistency and directional clarity in analysis
- Output: Polished financial report with editorial improvements

**MemoryTool**
- Purpose: Provides persistent storage for financial reports
- Storage: Local JSON file (report_memory.json)
- Operations: Save and retrieve reports by ticker symbol
- Benefit: Enables long-term context and trend analysis

### playground.py

The main application file that orchestrates the multi-agent workflow using Streamlit.

**Architecture**
- Framework: LangGraph for agent workflow orchestration
- UI: Streamlit web interface
- AI Model: OpenAI GPT-4 for all agents
- State Management: TypedDict for tracking workflow state

**Specialized Agents**

Research Agent
- Retrieves historical reports from memory
- Fetches current stock data
- Searches for recent news and sentiment
- Consolidates findings into comprehensive summary

Technical Analyst
- Calculates technical indicators
- Provides technical analysis insights
- Focuses purely on chart-based analysis

Report Writer
- Synthesizes research and technical data
- Provides investment recommendations (Buy/Hold/Sell)
- Formats analysis into professional reports

Critic Agent
- Reviews report quality and completeness
- Ensures focus on recent data (6-12 months)
- Validates formatting and professional standards
- Triggers revision cycles if needed

Memory Agent
- Archives final reports for future reference
- Maintains historical context

**Workflow Process**
1. Parallel Execution: Research and Technical Analysis run simultaneously
2. Data Aggregation: Results are combined from both branches
3. Report Generation: Writer creates initial draft
4. Quality Control: Critic reviews and provides feedback
5. Revision Loop: Up to 3 revision cycles for quality improvement
6. Memory Storage: Final report is archived
7. Web Display: Results presented in clean Markdown format

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
```

## Running it

```bash
streamlit run playground.py
```

Type in a ticker (AAPL, GOOGL, TSLA, whatever), hit **Run Analysis**, and expand the agent log if you want to watch what each step is doing. The finished report shows up at the bottom.

## Things I want to add next

- More technical indicators
- Other data sources besides yfinance
- A way to compare reports across runs so I can track trends over time
- Maybe swap GPT-4 out for something cheaper on the simpler agents
