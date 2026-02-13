# Import everything we need
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

# Import necessary modules
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from typing import TypedDict, Annotated, List, Union
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import re
import requests
# Import Web Search modules
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import yfinance as yf
import PyPDF2

from langchain_classic.tools.retriever import create_retriever_tool

# Specific modules for our financial multi-agent system
from rag import retriever

import re

# Set NewsData.io API Key
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")


# set LLM 
llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    temperature=0.2,
    max_tokens=800 
)

# Set tools
tools = []

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="retriever_search_results",
    description="Use this tool to fetch relevant financial information from the internal knowledge base. Input should be a natural language query. Output will be a JSON string with relevant information.",
    document_separator='\n\n'
    )

tavily_tool = TavilySearch(max_results=1)
# tools = [retriever_tool]

@tool
def yfinance_tool(query: str) -> str:
    """Tool to fetch real-time market data for a given ticker using yfinance"""
    try:
        ticker = yf.Ticker(query)
        data = ticker.history(period="1d")
        if not data.empty:
            latest = data.iloc[-1]
            return json.dumps({
                "ticker": query,
                "date": str(latest.name.date()),
                "open": latest["Open"],
                "high": latest["High"],
                "low": latest["Low"],
                "close": latest["Close"],
                "volume": int(latest["Volume"])
            }, indent=2)
        else:
            return f"No data found for ticker '{query}'"
    except Exception as e:
        return f"Error fetching data for ticker '{query}': {str(e)}"    

@tool
# Update fetch_ticker_data to extract ticker symbol from user query
def fetch_ticker_data(query: str):
    """Extract ticker symbol from query and fetch real-time market data using yfinance"""
    try:
        # Extract ticker symbol using regex (e.g., AAPL, TSLA, etc.)
        match = re.search(r"\b[A-Z]{1,5}\b", query)
        if not match:
            return f"No valid ticker symbol found in query: '{query}'"

        ticker = match.group(0)
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if data.empty:
            return f"No data found for ticker: {ticker}"
        latest_data = data.iloc[-1]
        return {
            "ticker": ticker,
            "date": latest_data.name.strftime("%Y-%m-%d"),
            "open": latest_data["Open"],
            "high": latest_data["High"],
            "low": latest_data["Low"],
            "close": latest_data["Close"],
            "volume": int(latest_data["Volume"])
        }
    except Exception as e:
        return f"Error fetching data for query '{query}': {str(e)}"


@tool
def calculator_tool(expression: str) -> str:
    """A simple calculator for math expressions. Use this for tax calculations.
    Input should be a mathematical expression like '(500000 * 0.05) + (250000 * 0.1)'.
    """
    try:
        # Using a safer way to evaluate simple math expressions
        # In a production app, use a proper math parser
        import math
        allowed_names = {"__builtins__": None, "math": math}
        result = eval(expression, allowed_names)
        return f"Calculation Result: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"

@tool
def pdf_upload_reader_tool(query: str = "") -> str:
    """Reads and extracts text from all PDF files in the ./upload directory.
    Use this when the user mentions an uploaded file or a document in the upload folder.
    """
    upload_dir = "./upload"
    if not os.path.exists(upload_dir):
        return "The upload directory does not exist."
    
    pdf_files = [f for f in os.listdir(upload_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        return "No PDF files found in the ./upload directory."
    
    full_text = ""
    for pdf_file in pdf_files:
        file_path = os.path.join(upload_dir, pdf_file)
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = f"--- Content of {pdf_file} ---\n"
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                full_text += text + "\n"
        except Exception as e:
            full_text += f"Error reading {pdf_file}: {str(e)}\n"
            
    return full_text if full_text else "No text could be extracted from the PDFs."

@tool
def sip_calculator_tool(monthly_investment: float, expected_return: float, tenure_years: int) -> str:
    """Calculates the future value of a Systematic Investment Plan (SIP).
    - monthly_investment: Amount invested every month.
    - expected_return: Annual expected rate of return in percentage (e.g., 12 for 12%).
    - tenure_years: Investment period in years.
    """
    try:
        monthly_rate = (expected_return / 100) / 12
        months = tenure_years * 12
        future_value = monthly_investment * (((1 + monthly_rate)**months - 1) / monthly_rate) * (1 + monthly_rate)
        total_invested = monthly_investment * months
        wealth_gained = future_value - total_invested
        
        return json.dumps({
            "estimated_future_value": f"{future_value:,.2f}",
            "total_amount_invested": f"{total_invested:,.2f}",
            "wealth_gained": f"{wealth_gained:,.2f}"
        }, indent=2)
    except Exception as e:
        return f"Error in SIP calculation: {str(e)}"

@tool
def inflation_calculator_tool(current_cost: float, inflation_rate: float, years: int) -> str:
    """Calculates the future cost of an expense adjusted for inflation.
    - current_cost: Present-day cost of the goal.
    - inflation_rate: Expected annual inflation rate in percentage (e.g., 6 for 6%).
    - years: Number of years until the goal is reached.
    """
    try:
        future_cost = current_cost * (1 + (inflation_rate / 100))**years
        return json.dumps({
            "future_adjusted_cost": f"{future_cost:,.2f}",
            "inflation_applied": f"{inflation_rate}% yearly",
            "time_horizon": f"{years} years"
        }, indent=2)
    except Exception as e:
        return f"Error in inflation calculation: {str(e)}"

# Helper function for NewsData.io
def fetch_newsdata(query: str):
    """Fetch news articles from NewsData.io"""
    try:
        url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&q={query}&language=en"
        response = requests.get(url)
        data = response.json()
        if data.get("status") == "success":
            articles = data.get("results", [])
            summary = []
            for art in articles[:3]: # Top 3 articles
                summary.append({
                    "title": art.get("title"),
                    "link": art.get("link"),
                    "description": art.get("description"),
                    "source": art.get("source_id")
                })
            return summary
        return f"NewsData.io error: {data.get('results', {}).get('message', 'Unknown error')}"
    except Exception as e:
        return f"Fetch error: {str(e)}"


## Tools for FQAA agent - RAG retriever and TavilySearch
fqaa_tools = [retriever_tool, tavily_tool]

# maa_tools = [yfinance_tool, retriever_tool, tavily_tool]
maa_tools = [fetch_ticker_data, tavily_tool]

fqaa_system_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a highly knowledgeable Finance Q&A Agent specializing in financial education.
You help users understand finance concepts clearly, accurately, and responsibly.

SCOPE & BEHAVIOR RULES:
- ONLY answer finance-related questions (personal finance, investing concepts, markets, economics, accounting, taxation basics, fintech, crypto basics).
- If a question is NOT finance-related, politely decline and redirect to finance education.
- DO NOT provide personalized financial, legal, or tax advice.
- Do NOT recommend specific stocks, mutual funds, crypto assets, or give buy/sell signals.
- Always explain concepts in a beginner-friendly way first, then add advanced depth if helpful.

KNOWLEDGE SOURCES (STRICT ORDER):
1. First, use the provided CONTEXT (RAG – internal knowledge base).
2. If CONTEXT is insufficient, outdated, or missing:
   - Use TavilySearch to fetch up-to-date, factual financial information.
3. If reliable information cannot be found, clearly say so.

REASONING STYLE (ReAct):
Follow this internal process before answering:
1. THOUGHT: Identify what financial concept or principle is being asked.
2. ACTION: 
   - Use RAG context if available.
   - Use TavilySearch ONLY if current data, definitions, regulations, or market structure info is required.
3. OBSERVATION: Validate and synthesize information from sources.
4. RESPONSE: Provide a clear, structured, and educational answer.

RESPONSE GUIDELINES:
- Use simple language and real-world examples.
- Structure answers with headings and bullet points.
- Include formulas or frameworks when useful.
- Mention risks, assumptions, and limitations clearly.
- For complex topics, include a short summary at the end.

WHEN UNCERTAIN:
- Say “I don’t have enough reliable information to answer that accurately.”
- Do NOT guess or hallucinate.

TONE:
- Neutral, educational, and unbiased.
- Supportive to beginners, insightful for advanced users.

TOOLS AVAILABLE:
- RAG Context (internal financial knowledge base)
- TavilySearch (for current financial definitions, regulations, market mechanisms, or news)

Your goal is to improve financial literacy, not to influence financial decisions.
"""),
    MessagesPlaceholder(variable_name="messages"),
])

nsa_tools = [fetch_ticker_data, tavily_tool]

## Tools for TEA agent - RAG, Calculator, Tavily, and PDF Reader
tea_tools = [retriever_tool, calculator_tool, tavily_tool, pdf_upload_reader_tool]

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

# System prompt for the Tax Education Agent (TEA)
tea_system_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Tax Education Agent (TEA) specializing in tax laws, regulations, and calculations.
Your goal is to educate users on tax concepts, perform tax calculations, and provide the latest tax updates.

SCOPE & BEHAVIOR RULES:
- Use RAG context for academic/official tax concepts and laws.
- Use the Calculator tool for precise math/tax computations.
- Use Tavily for the latest tax updates or news not in the knowledge base.
- Use the PDF Reader tool if the user mentions uploaded documents or files in the upload folder.

SYNTHESIS GUIDELINES:
- Provide clear, structured explanations of tax concepts.
- Show the step-by-step breakdown of calculations.
- Mention the source of the tax rules (e.g., Income Tax Act, recent budget updates).
- Always include a disclaimer that you are an AI and not a professional tax advisor.

RESPONSE FORMAT:
## 📑 Tax Concept Explanation
## 🧮 Calculation Breakdown (if applicable)
## 📢 Latest Updates & Insights
## 📂 Document Analysis (if applicable)
## ⚠️ Disclaimer
"""),
    MessagesPlaceholder(variable_name="messages"),
])

# System prompt for the Goal Planning Agent (GPA)
gpa_system_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Goal Planning Agent (GPA) specializing in helping users achieve financial milestones and manage their monthly surplus.
Your goal is to provide realistic, data-driven, and actionable financial plans, including long-term goals (retirement, education) and monthly budget allocation.

SCOPE & BEHAVIOR RULES:
- Use the inflation_calculator_tool to adjust current costs to future values.
- Use the sip_calculator_tool to determine required monthly investments.
- Use RAG context for asset allocation strategies and financial planning principles.
- Use yfinance for market return assumptions if needed.
- Always explain the "Why" behind your recommendations.

MONTHLY SURPLUS ALLOCATION FRAMEWORK:
If a user provides their salary (x) and monthly expenses (y):
1. Calculate Surplus: Monthly Surplus = x - y.
2. Emergency Fund (Cash in Hand): Recommend keeping 3-6 months of expenses in a liquid savings account or liquid fund.
3. Insurance Strategy:
   - Term Insurance: Recommend a sum assured of 10-15x annual income.
   - Health Insurance: Recommend adequate coverage for the individual/family.
4. Investment Split: Suggest splitting the remaining surplus after insurance premiums:
   - Mutual Funds: For diversified, professionally managed growth.
   - Equity: For direct stock market exposure (higher risk/reward).

PLANNING FRAMEWORK:
1. Understand the goal or current surplus situation.
2. Adjust for inflation if a future goal is targetted.
3. Analyze the gap or allocation priorities.
4. Suggest a strategy (Asset allocation, Insurance coverage, Emergency fund).
5. Provide a clear Action Plan.

RESPONSE FORMAT:
## 🎯 Scenario Overview
## 💰 Financial Analysis (Surplus, Inflation-Adjusted Target)
## 🛡️ Risk Protection (Insurance & Emergency Fund)
## 📈 Suggested Investment Strategy (Mutual Funds, Equity, Cash)
## 📅 Action Plan
## 💡 Reasoning & Assumptions
"""),
    MessagesPlaceholder(variable_name="messages"),
])





# Create FQAA agent
## Bind tools to the LLM
fqaa_llm_with_tools = llm.bind_tools(fqaa_tools)
fqaa_agent = fqaa_system_prompt | fqaa_llm_with_tools | StrOutputParser()

# Create MAA agent
maa_llm_with_tools = llm.bind_tools(maa_tools)
maa_agent = maa_system_prompt | maa_llm_with_tools | StrOutputParser() 


# Create NSA agent
# nsa_llm_with_tools = llm.bind_tools(nsa_tools)
nsa_agent = nsa_system_prompt | llm | StrOutputParser()

# TEA also uses base LLM for final synthesis of gathered tax data
tea_agent = tea_system_prompt | llm | StrOutputParser()

# GPA also uses base LLM for final synthesis of goal planning data
gpa_agent = gpa_system_prompt | llm | StrOutputParser()


# Create Router fuhnction 
def create_router():
    """Creates a router for the three travel agents using LangGraph patterns"""

    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a routing expert for a 6 Multi Agent Financial Advisor system.

        Analyze the user's query and decide which specialist agent should handle it:

        Finance Q&A Agent (FQAA): Handles general financial education queries
        Portfolio Analysis Agent (PAA): Reviews and analyzes user portfolios
        Market Analysis Agent (MAA): Provides real-time market insights, ticker info.
        Goal Planning Agent (GPA): Assists with financial goal setting and planning
        News Synthesizer Agent (NSA): Summarizes and contextualizes financial news
        Tax Education Agent (TEA): Explains tax concepts and account types

        Respond with ONLY one word: FQAA, PAA, MAA, GPA, NSA, or TEA

        Examples:
        "What is a stock" → FQAA
        "Analyze my portfolio" → PAA
        "What's the current price of AAPL?" → MAA
        "What are some good retirement planning strategies?" → GPA
        "Summarize today's market news" → NSA
        "Explain how a 401(k) works" → TEA
        "What is the P/E ratio?" → FQAA
        "Provide an analysis of my stock holdings" → PAA
        "What's the current market sentiment on tech stocks?" → MAA
        "How should I allocate my investments for a 10-year goal?" → GPA
        "What are the key events affecting the markets today?" → NSA
        "How can I save on taxes with my investments?" → TEA"""),

        ("user", "Query: {query}")
    ])

    router_chain = router_prompt | llm | StrOutputParser()
    def route_query(state):
        """Router function for LangGraph - decides which agent to call next"""

        # Get the latest user message
        user_message = state["messages"][-1].content

        print(f"🧭 Router analyzing: '{user_message[:50]}...'")

        try:
            # Get LLM routing decision
            decision = router_chain.invoke({"query": user_message}).strip().upper()
            print(f"🎯 Router decision: {decision}")

            # post process decision to make sure it's accurate
            if decision not in ["FQAA", "PAA", "MAA", "GPA", "NSA", "TEA"]:
                decision = "FQAA"

            # Map to our agent node names
            agent_mapping = {
                "FQAA": "fqaa_agent",
                "PAA": "paa_agent",
                "MAA": "maa_agent",
                "GPA": "gpa_agent",
                "NSA": "nsa_agent",
                "TEA": "tea_agent"
            }

            next_agent = agent_mapping.get(decision, "fqaa_agent")
            print(f"🎯 Router decision: {decision} → {next_agent}")

            return next_agent

        except Exception as e:
            print(f"⚠️ Router error, defaulting to fqaa_agent: {e}")
            return "fqaa_agent"

    return route_query


# Create the router
router = create_router()
print("✅ Multi Agent Finance Router created for LangGraph!")

from typing import TypedDict, Annotated, List, Optional
import operator
from langchain_core.messages import BaseMessage

## Define state schema for the financial multi agent system
class FinancePlannerState(TypedDict):
    """Simple state schema for financial multi agent system"""

    # Conversation history - persisted with checkpoint memory
    messages: Annotated[List[BaseMessage], operator.add]

    # Agent routing
    next_agent: Optional[str]

    # Current user query
    user_query: Optional[str]



# FQAA Agent node functions
def fqaa_agent_node(state: FinancePlannerState):
    """Financial query analysis agent node"""
    messages = state["messages"]
    user_message = messages[-1].content if messages else ""

    sources = []
    try:
        docs = retriever.get_relevant_documents(user_message)
        for doc in docs:
            src = doc.metadata.get("source") or doc.metadata.get("url")
            if src:
                sources.append(src)
    except Exception:
        pass

    # Always fetch RAG context and inject it for the LLM.
    try:
        rag_context = retriever_tool.invoke({"query": user_message})
    except Exception as e:
        rag_context = f"RAG retrieval failed: {str(e)}"

    # Keep context bounded to avoid oversized tool payloads
    max_rag_chars = 4000
    if isinstance(rag_context, str) and len(rag_context) > max_rag_chars:
        rag_context = rag_context[:max_rag_chars] + "\n[truncated]"

    augmented_messages = messages + [
        SystemMessage(content=f"RAG_CONTEXT:\n{rag_context}")
    ]

    response = fqaa_agent.invoke({"messages": augmented_messages})

    # Handle tool calls if present
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_messages = []
        for tool_call in response.tool_calls:
            if tool_call['name'] == 'retriever_search_results':
                try:
                    tool_result = retriever_tool.invoke({"query": tool_call['args']['query']})
                except Exception as e:
                    tool_result = f"RAG retrieval failed: {str(e)}"
                tool_messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call['id']
                ))
            elif tool_call['name'] == 'tavily_search_results_json':
                try:
                    tool_result = tavily_tool.search(query=tool_call['args']['query'], max_results=2)
                    tool_result = json.dumps(tool_result, indent=2)
                    try:
                        tavily_data = json.loads(tool_result)
                        results = tavily_data.get("results", []) if isinstance(tavily_data, dict) else tavily_data
                        for item in results:
                            url = item.get("url") or item.get("source")
                            if url:
                                sources.append(url)
                    except Exception:
                        pass
                except Exception as e:
                    tool_result = f"Search failed: {str(e)}"

                tool_messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call['id']
                ))

        if tool_messages:
            all_messages = augmented_messages + [response] + tool_messages
            final_response = fqaa_agent.invoke({"messages": all_messages})
            if sources:
                sources_text = "\n".join(f"- {src}" for src in sorted(set(sources)))
                final_response = AIMessage(content=f"{final_response.content}\n\nSources:\n{sources_text}")
            return {"messages": [response] + tool_messages + [final_response]}

    if sources:
        sources_text = "\n".join(f"- {src}" for src in sorted(set(sources)))
        response = AIMessage(content=f"{response.content}\n\nSources:\n{sources_text}")

    return {"messages": [response]}

# Helper function to extract ticker symbol with multi-exchange support
def extract_ticker_symbol(query: str) -> str:
    """Extract ticker symbol from query and add appropriate exchange suffix (NSE/BSE for Indian stocks)"""
    
    # Step 1: Try regex for explicit ticker symbols (all caps, 1-5 letters)
    match = re.search(r"\b[A-Z]{1,10}\b", query)
    base_ticker = match.group(0) if match else None
    
    # Step 2: If no regex match, use LLM to extract ticker from company name
    if not base_ticker:
        try:
            ticker_extraction_prompt = f"""Extract the stock ticker symbol from this query. 
Query: "{query}"

For Indian stocks/ETFs, provide just the base ticker (e.g., for Vedanta say VEDL, for Adani Green say ADANIGREEN, for Silverbees say SILVERBEES).
For US stocks, provide the ticker (e.g., AAPL, TSLA, MSFT).
Respond with ONLY the ticker symbol. If you cannot identify a ticker, respond with 'UNKNOWN'.
Ticker:"""
            
            response = llm.invoke(ticker_extraction_prompt)
            base_ticker = response.content.strip().upper().replace(" ", "")
            
            if base_ticker == 'UNKNOWN':
                return query
        except Exception as e:
            print(f"LLM ticker extraction failed: {e}")
            return query
    
    # Step 3: Try to fetch data with different exchange suffixes
    # Priority: NSE (.NS) > BSE (.BO) > US (no suffix)
    suffixes_to_try = [
        ".NS",      # NSE (National Stock Exchange of India)
        ".BO",      # BSE (Bombay Stock Exchange)
        "",         # US stocks (no suffix)
    ]
    
    for suffix in suffixes_to_try:
        test_ticker = base_ticker + suffix
        try:
            # Quick validation: try to fetch 1 day of data
            ticker_obj = yf.Ticker(test_ticker)
            data = ticker_obj.history(period="1d")
            if not data.empty:
                print(f"✓ Found valid ticker: {test_ticker}")
                return test_ticker
        except Exception:
            continue
    
    # If nothing worked, return base ticker (will likely fail, but error handling exists)
    print(f"⚠ No valid ticker found, using: {base_ticker}")
    return base_ticker

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


# TEA Agent node functions
def tea_agent_node(state: FinancePlannerState):
    """Tax Education Agent (TEA) node - explains tax concepts, performs calculations, and reads documents"""
    messages = state["messages"]
    user_message = messages[-1].content

    # Step 1: Fetch Tax Concept from RAG
    try:
        rag_context = retriever_tool.invoke({"query": f"tax laws rules and concepts for {user_message}"})
    except Exception as e:
        rag_context = f"RAG retrieval failed: {str(e)}"

    # Step 2: Fetch Latest Tax Updates from Tavily
    try:
        tavily_query = f"latest tax updates or news for {user_message}"
        tavily_response = tavily_tool.invoke({"query": tavily_query})
        tavily_updates = json.dumps(tavily_response, indent=2) if tavily_response else "No latest updates found."
    except Exception as e:
        tavily_updates = f"Tavily error: {str(e)}"

    # Step 3: Check for PDF uploads
    try:
        pdf_content = pdf_upload_reader_tool.invoke({"query": ""})
    except Exception as e:
        pdf_content = f"PDF reading error: {str(e)}"

    # Step 4: Calculator Hint (Synthesis will use the tool logic)
    calc_hint = ""
    if any(op in user_message for op in ["+", "-", "*", "/", "%"]):
        calc_hint = "\nNote: If a calculation is requested, show the step-by-step breakdown using the Calculator tool logic."

    # Step 5: Synthesis using context
    context_content = f"""You have gathered the following information for synthesis:

=== TAX CONCEPTS (from RAG) ===
{rag_context}

=== LATEST TAX UPDATES (from Tavily) ===
{tavily_updates}

=== UPLOADED DOCUMENT CONTENT (if any) ===
{pdf_content}
{calc_hint}

Now synthesize this into a structured tax education report according to your system prompt instructions.
"""
    
    context_message = SystemMessage(content=context_content)
    augmented_messages = messages + [context_message]
    
    # We use tea_agent (synthesis-only)
    response = tea_agent.invoke({"messages": augmented_messages})
    
    return {"messages": [response]}


# GPA Agent node functions
def gpa_agent_node(state: FinancePlannerState):
    """Goal Planning Agent (GPA) node - assists with financial goal setting, planning, and surplus allocation"""
    messages = state["messages"]
    user_message = messages[-1].content

    # Step 1: Query RAG for asset allocation, planning principles, and insurance guidelines
    try:
        rag_query = f"financial planning asset allocation surplus management insurance guidelines {user_message}"
        rag_context = retriever_tool.invoke({"query": rag_query})
    except Exception as e:
        rag_context = f"RAG retrieval failed: {str(e)}"

    # Step 2: Calculator Guidance
    # The agent uses tools internally, but we provide a hint for structured thinking
    calc_hint = """GUIDANCE:
- If salary and expenses are provided, calculate surplus = (Salary - Expenses).
- Use inflation_calculator_tool for future goal costs.
- Use sip_calculator_tool for monthly investment requirements.
- Follow the 3-6 month rule for Emergency Funds.
"""

    # Step 3: Synthesis using context
    context_content = f"""You have gathered the following information for goal planning and surplus allocation:

=== PLANNING & ALLOCATION CONCEPTS (from RAG) ===
{rag_context}

=== CALCULATION & STRATEGY GUIDANCE ===
{calc_hint}

Now synthesize this into a structured financial plan according to your system prompt instructions.
Address surplus allocation specifically if salary and expenses are mentioned.
"""
    
    context_message = SystemMessage(content=context_content)
    augmented_messages = messages + [context_message]
    
    # We use gpa_agent (synthesis-only)
    response = gpa_agent.invoke({"messages": augmented_messages})
    
    return {"messages": [response]}


# Router Node

def router_node(state: FinancePlannerState):
    """Router node - determines which agent should handle the query"""
    user_message = state["messages"][-1].content
    next_agent = router(state)

    return {
        "next_agent": next_agent,
        "user_query": user_message
    }

# Conditional routing function

def route_to_agent(state: FinancePlannerState):
    """Conditional edge function - routes to appropriate agent based on router decision"""

    next_agent = state.get("next_agent")

    if next_agent == "fqaa_agent":
        return "fqaa_agent"
    elif next_agent == "paa_agent":
        return "paa_agent"
    elif next_agent == "maa_agent":
        return "maa_agent"
    elif next_agent == "gpa_agent":
        return "gpa_agent"
    elif next_agent == "nsa_agent":
        return "nsa_agent"
    elif next_agent == "tea_agent":
        return "tea_agent"
    else:
        # Default fallback
        return "fqaa_agent"


# Build Graph & Add Memory
from langgraph.graph import StateGraph, END
from typing import Literal
from langgraph.checkpoint.memory import InMemorySaver

# Build the complete financial planning graph
workflow = StateGraph(FinancePlannerState)

# Add all nodes to the graph
workflow.add_node("router", router_node)
workflow.add_node("fqaa_agent", fqaa_agent_node)
# workflow.add_node("paa_agent", paa_agent_node)
workflow.add_node("maa_agent", maa_agent_node)
workflow.add_node("nsa_agent", nsa_agent_node)
workflow.add_node("gpa_agent", gpa_agent_node)
workflow.add_node("tea_agent", tea_agent_node)

# Set entry point - always start with router
workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    route_to_agent,
    {
        "fqaa_agent": "fqaa_agent",
        "maa_agent": "maa_agent",
        "nsa_agent": "nsa_agent",
        "gpa_agent": "gpa_agent",
        "tea_agent": "tea_agent"
    }
)


# Add edges from each agent back to END
workflow.add_edge("fqaa_agent", END)
workflow.add_edge("maa_agent", END)
workflow.add_edge("nsa_agent", END)
workflow.add_edge("gpa_agent", END)
workflow.add_edge("tea_agent", END)

# The checkpointer is a mechanism to save the state of the graph after each step completes.
checkpointer = InMemorySaver()

# Compile the graph
financial_planner = workflow.compile(checkpointer=checkpointer)

print("✅ Financial Planning Graph built successfully!")


## Grapgh Display (uncomment if running in Jupyter or IPython environment)
# from IPython.display import Image, display

# # Generate and display the graph
# graph_image = financial_planner.get_graph().draw_mermaid_png()
# display(Image(graph_image))


# Test Your Multi-Agent System
from langchain_core.messages import HumanMessage

def test_system(query, thread_id="test_thread"):
    """Test our multi-agent system"""
    print(f"🧑 User: {query}")

    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next_agent": ""
    }

    # Configuration for the checkpointer
    # The config object in your code tells the travel_planner application: "Run this workflow, save the progress using my checkpointer mechanism, and link all saved history to this specific thread_id."
    config = {"configurable": {"thread_id": thread_id}}

    # Run the system (use financial_planner)
    # We pass the config to invoke
    result = financial_planner.invoke(initial_state, config)

    # Update the response assignment to handle the case where result["messages"][-1] is a string
    response = result["messages"][-1] if isinstance(result["messages"][-1], str) else result["messages"][-1].content
    print(f"🤖 Assistant: {response}")
    print("-" * 50)


if __name__ == "__main__":
    # Test with different queries
    
    # FQAA Agent test
    # test_system("What is the stock ?", thread_id="test_thread_01")
    # test_system("What is Mutual Fund", thread_id="test_thread_001")
    
    # MAA Agent test - US Stocks
    # test_system("What is the current price of AAPL?", thread_id="test_thread_02")
    # test_system("What is the current price of TSLA?", thread_id="test_thread_03")
    # test_system("What is the current price of Apple?", thread_id="test_thread_04")
    
    # MAA Agent test - Indian Stocks/ETFs
    # test_system("What is the current price of vedanta?", thread_id="test_thread_indian_01")
    
    # NSA Agent test
    # test_system("Summarize today's financial news for Tesla and explain the impact.", thread_id="test_thread_nsa_01")
    # test_system("Summarize today's financial news for Silverbees and explain the impact.", thread_id="test_thread_nsa_02")  
    # test_system("Summarize today's financial news for Vedanta and explain the impact.", thread_id="test_thread_nsa_03")  
    
    # GPA Agent test
    # test_system("I want to save for my child's education in 15 years. Current cost is 20,00,000. How much should I save monthly assuming 10% return and 6% inflation?", thread_id="test_thread_gpa_01")
    
    test_system("My salary is 1,20,000 monthly and my monthly expenses are 50,000. Plan the best approach for the surplus: how much for cash in hand, mutual funds, equity, term insurance, and health insurance?", thread_id="test_thread_gpa_surplus_01")
    
    # TEA Agent test
    # test_system("Calculate the tax for 12,00,000 income under the new regime in India and explain the deduction.", thread_id="test_thread_tea_01")
    # test_system("Calculate the tax for file: pnl-JC0929.pdf in upload folder income under the new regime in India and explain the deduction.", thread_id="test_thread_tea_02")


