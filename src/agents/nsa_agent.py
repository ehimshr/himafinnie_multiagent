import json
import yfinance as yf
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.core.llm import llm
from src.utils.tools import tavily_tool, fetch_newsdata, extract_ticker_symbol
from src.workflow.state import FinancePlannerState

# System prompt for the News Synthesizer Agent (NSA)
nsa_system_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a highly skilled News Synthesizer Agent (NSA) specializing in financial markets.
Your goal is to provide users with a comprehensive, well-structured summary of financial news that includes market context, impact analysis, and sentiment insights.

You will be provided with news from NewsData.io, Tavily, and market data from yfinance.
Your task is to SYNTHESIZE this information into a human-readable report.
DO NOT output tool calls. Output only the final markdown report.

RESPONSE FORMAT:
## 📰 News Summary
## 🌍 Market Context
## 📉 Impact Analysis
- **Short-term:**
- **Mid-term:**
- **Long-term:**
## ⚠️ Risks & 🚀 Opportunities
- **Risks:**
- **Opportunities:**
## ⚖️ Sentiment & Outlook
- **Sentiment:**
- **Actionable Insight:** (Buy / Sell / Hold / Watch)

TONE:
Professional, analytical, and objective.
"""),
    MessagesPlaceholder(variable_name="messages"),
])

# Create NSA agent
# nsa_llm_with_tools = llm.bind_tools(nsa_tools if defined)
nsa_agent = nsa_system_prompt | llm | StrOutputParser()

# NSA Agent node functions
def nsa_agent_node(state: FinancePlannerState):
    """News Synthesizer Agent (NSA) node - summarizes financial news and provides context/sentiment"""
    messages = state["messages"]
    user_message = messages[-1].content

    # Step 1: Gather News using NewsData.io
    try:
        newsdata_res = fetch_newsdata(user_message)
        newsdata_summary = json.dumps(newsdata_res, indent=2)
    except Exception as e:
        newsdata_summary = f"NewsData.io error: {str(e)}"

    # Step 2: Gather News using Tavily (Web Search)
    try:
        tavily_query = f"financial news summary for {user_message}"
        tavily_response = tavily_tool.invoke({"query": tavily_query})
        tavily_summary = json.dumps(tavily_response, indent=2) if tavily_response else "No news found."
    except Exception as e:
        tavily_summary = f"Tavily error: {str(e)}"

    # Step 3: Extract Ticker and get Market Context using yfinance
    try:
        ticker_symbol = extract_ticker_symbol(user_message)
        if ticker_symbol and ticker_symbol != user_message:
            ticker = yf.Ticker(ticker_symbol)
            data = ticker.history(period="5d")
            if not data.empty:
                latest = data.iloc[-1]
                prev = data.iloc[-2] if len(data) > 1 else latest
                change = latest['Close'] - prev['Close']
                change_pct = (change / prev['Close']) * 100
                market_context = f"""Ticker: {ticker_symbol}
Current Price: ${latest['Close']:.2f}
Change: ${change:.2f} ({change_pct:.2f}%)
Recent Volume: {int(latest['Volume']):,}"""
            else:
                market_context = f"Ticker {ticker_symbol} found but no recent price data available."
        else:
            market_context = "No specific stock ticker identified for market context."
    except Exception as e:
        market_context = f"Error fetching market context: {str(e)}"

    # Step 4: Synthesis using context
    context_content = f"""You have gathered the following information for synthesis:

=== NEWS FROM NEWSDATA.IO ===
{newsdata_summary}

=== NEWS FROM TAVILY (WEB SEARCH) ===
{tavily_summary}

=== MARKET CONTEXT (yfinance) ===
{market_context}

Now synthesize this into a structured news report according to your system prompt instructions.
"""
    
    context_message = SystemMessage(content=context_content)
    augmented_messages = messages + [context_message]
    
    # We use nsa_agent (which is now synthesis-only, no tools bound)
    response = nsa_agent.invoke({"messages": augmented_messages})
    
    return {"messages": [response]}
