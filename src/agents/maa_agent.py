import json
import yfinance as yf
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.core.llm import llm
from src.rag.rag_engine import retriever_tool
from src.utils.tools import tavily_tool, fetch_ticker_data, extract_ticker_symbol
from src.workflow.state import FinancePlannerState

maa_tools = [fetch_ticker_data, tavily_tool]

# System prompt for the Market Analysis Agent (MAA)
maa_system_prompt = ChatPromptTemplate.from_messages([
    ("system", """  You are a Market Analysis Agent specializing in providing real-time market insights, fundamental and technical analysis for stocks, ETFs, and crypto assets.
SCOPE & BEHAVIOR RULES:
- Focus on providing factual market data, analysis, and insights.
- Do NOT provide personalized financial advice or specific buy/sell recommendations.
- Always cite sources for market data and analysis. 
KNOWLEDGE SOURCES (STRICT ORDER):
1. Real-time market data (e.g., via yfinance or similar APIs)
2. RAG context for fundamental and technical analysis concepts
3. TavilySearch for up-to-date market insights and sentiment
REASONING STYLE:
1. THOUGHT: Identify the specific market information or analysis requested.
2. ACTION:
    - Fetch real-time market data for the specified ticker or asset.
    - Use RAG context to provide fundamental and technical analysis.
    - Use TavilySearch to gather current market sentiment and news.
3. OBSERVATION: Synthesize data and insights from all sources.
4. RESPONSE: Provide a clear, structured market analysis with cited sources.
RESPONSE GUIDELINES:
- Use clear headings (e.g., Real-Time Data, Fundamental Analysis, Technical Analysis, Market
Insights).
- Include relevant metrics (e.g., P/E ratio, moving averages) and explain their significance.
- Always cite sources for data and insights.
WHEN UNCERTAIN:
- Say “I don’t have enough reliable information to provide a market analysis.”
TONE:
- Neutral, factual, and analytical.
- Aim to inform and educate users about market conditions and analysis techniques."""),
    MessagesPlaceholder(variable_name="messages"),
])

# Create MAA agent
maa_llm_with_tools = llm.bind_tools(maa_tools)
maa_agent = maa_system_prompt | maa_llm_with_tools | StrOutputParser() 

# MAA Agent node functions
def maa_agent_node(state: FinancePlannerState):
    """Market analysis agent node - provides comprehensive market insights by synthesizing real-time data, RAG knowledge, and Tavily insights"""
    messages = state["messages"]
    user_message = messages[-1].content

    # Step 1: Extract ticker symbol and fetch real-time data
    try:
        # Use smart ticker extraction (regex + LLM fallback)
        ticker_symbol = extract_ticker_symbol(user_message)
        
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(period="1d")
        if not data.empty:
            latest = data.iloc[-1]
            ticker_data = f"""Ticker: {ticker_symbol}
Date: {str(latest.name.date())}
Open: ${latest['Open']:.2f}
High: ${latest['High']:.2f}
Low: ${latest['Low']:.2f}
Close: ${latest['Close']:.2f}
Volume: {int(latest['Volume']):,}"""
        else:
            ticker_data = f"No data found for ticker '{ticker_symbol}'"
    except Exception as e:
        ticker_data = f"Error fetching ticker data: {str(e)}"

    # Step 2: Fetch RAG context for fundamental and technical analysis concepts
    try:
        # Query RAG for fundamental analysis concepts
        fundamental_query = f"fundamental analysis metrics ratios valuation {ticker_symbol}"
        fundamental_context = retriever_tool.invoke({"query": fundamental_query})
        
        # Query RAG for technical analysis concepts  
        technical_query = f"technical analysis indicators trends patterns"
        technical_context = retriever_tool.invoke({"query": technical_query})
        
        rag_context = f"FUNDAMENTAL ANALYSIS CONCEPTS:\n{fundamental_context}\n\nTECHNICAL ANALYSIS CONCEPTS:\n{technical_context}"
    except Exception as e:
        rag_context = f"RAG retrieval failed: {str(e)}"

    # Step 3: Fetch current market insights and news from Tavily
    try:
        tavily_query = f"{ticker_symbol} stock news analysis latest"
        tavily_response = tavily_tool.invoke({"query": tavily_query})
        tavily_insights = json.dumps(tavily_response, indent=2) if tavily_response else "No Tavily insights available"
    except Exception as e:
        tavily_insights = f"Tavily search failed: {str(e)}"

    # Step 4: Create context message for the LLM to synthesize
    context_message = SystemMessage(content=f"""You have the following information to synthesize:

=== REAL-TIME TICKER DATA ===
{ticker_data}

=== FUNDAMENTAL & TECHNICAL ANALYSIS KNOWLEDGE (from RAG) ===
{rag_context}

=== CURRENT MARKET INSIGHTS (from Tavily) ===
{tavily_insights}

Now synthesize this information to provide a comprehensive market analysis following your system prompt guidelines.""")

    # Step 5: Invoke the MAA agent with all context
    augmented_messages = messages + [context_message]
    response = maa_agent.invoke({"messages": augmented_messages})

    return {"messages": [response]}
