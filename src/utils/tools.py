import json
import os
import re
import requests
import yfinance as yf
import PyPDF2
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from src.core.llm import llm

# Set NewsData.io API Key
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")

tavily_tool = TavilySearch(max_results=1)

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
