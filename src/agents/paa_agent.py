from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from src.core.llm import llm
from src.core.mcp_server import mcp_server_session
from src.utils.tools import csv_excel_reader_tool, pdf_upload_reader_tool, fetch_ticker_data, tavily_tool
from src.workflow.state import FinancePlannerState
import json

# Standard Tools for PAA (Fallback if not using Kite MCP)
paa_tools = [fetch_ticker_data, tavily_tool]

# System prompt for the Portfolio Analysis Agent (PAA)
# paa_system_prompt = ChatPromptTemplate.from_messages([
#     ("system", """You are the Portfolio Analysis Agent (PAA), modeling your investment philosophy after legendary investors Benjamin Graham and Warren Buffett.
# Your goal is to analyze user portfolios with a focus on value investing, long-term growth, margin of safety, and proper diversification.

# SCOPE & BEHAVIOR RULES:
# - If analyzing uploaded data (PDF/CSV), evaluate that data directly.
# - If the user asks you to analyze their Zerodha or Kite portfolio, YOU MUST USE the tools provided to you (like view_holdings, view_positions) to fetch their live portfolio data before providing analysis.
# - Evaluate diversification (Sector, Asset Class).
# - Provide specific Buy/Sell/Hold/Average advice for individual stocks based on fundamental principles.
# - Use 'fetch_ticker_data' to get current prices if needed (if available).
# - Use 'tavily_tool' to get recent news impacting specific holdings.

# ANALYSIS FRAMEWORK:
# 1. **Diversification Check**: Is the portfolio too concentrated? Are there too many correlated assets?
# 2. **Stock Analysis (Buffett/Graham Style)**:
#    - Look for strong fundamentals (though you may not have full balance sheets, use proxy info or general knowledge).
#    - Advise "Hold" or "Buy" for quality companies with a moat.
#    - Advise "Sell" for speculative assets or deteriorating fundamentals.
#    - Advise "Average" if a quality stock is down but fundamentals remain strong.
# 3. **Actionable Advice**: Be direct. "Buy more of X", "Trim position in Y".

# RESPONSE FORMAT:
# ## 💼 Portfolio Overview
# - **Diversification Status**: (Good / Poor / Needs Improvement)
# - **Sector Allocation**:
# ## 📊 Individual Stock Analysis
# | Stock | Action (Buy/Sell/Hold/Avg) | Rationale (Buffett Style) |
# |-------|----------------------------|---------------------------|
# | ...   | ...                        | ...                       |
# ## 🧠 Strategic Advice
# - **Graham's Wisdom**:
# - **Buffett's Insight**:
# ## ⚠️ Risk Assessment
# """),
#     MessagesPlaceholder(variable_name="messages"),
# ])


paa_system_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the Portfolio Analysis Agent (PAA), 
Your goal is to analyze user portfolios with a focus on value investing, long-term growth, margin of safety, and proper diversification.

SCOPE & BEHAVIOR RULES:
- If analyzing uploaded data (PDF/CSV), evaluate that data directly.
- If the user asks you to analyze their Zerodha or Kite portfolio, YOU MUST USE the tools provided to you (like view_holdings, view_positions) to fetch their live portfolio data before providing analysis.
- Evaluate diversification (Sector, Asset Class).
- Provide specific Buy/Sell/Hold/Average advice for individual stocks based on fundamental principles.
- Use 'fetch_ticker_data' to get current prices if needed (if available).
- Use 'tavily_tool' to get recent news impacting specific holdings.

ANALYSIS FRAMEWORK:
1. **Diversification Check**: Is the portfolio too concentrated? Are there too many correlated assets?
"""),
    MessagesPlaceholder(variable_name="messages"),
])

# Define the standard standalone agent for manual document analysis
paa_llm_with_tools = llm.bind_tools(paa_tools)
paa_agent = paa_system_prompt | paa_llm_with_tools

# PAA Agent node functions
async def paa_agent_node(state: FinancePlannerState):
    """Portfolio Analysis Agent (PAA) node"""
    messages = state["messages"]
    
    # Determine mode based on user's last message
    last_user_message = [m.content for m in messages if getattr(m, 'type', '') == 'human' or m.__class__.__name__ == 'HumanMessage']
    user_query = last_user_message[-1].lower() if last_user_message else ""
    
    use_mcp = any(keyword in user_query for keyword in ["zerodha", "kite", "mcp"])

    if use_mcp:
        print("PAA Agent: Using Live Zerodha MCP flow...")
        # Flow A: Live MCP Data
        try:
             async with mcp_server_session() as mcp_tools:
                 # Combine standard tools with MCP tools
                 all_tools = paa_tools + mcp_tools
                 
                 # Create a React Agent that can loop to use Kite tools
                 agent = create_react_agent(llm, all_tools, prompt=paa_system_prompt)
                 
                 # Invoke the agent with the chat history
                 response = await agent.ainvoke({"messages": messages})
                 
                 # LangGraph's create_react_agent returns the full updated list of messages.
                 # We just want to return the last message (the AI's final answer)
                 final_message = response["messages"][-1]
                 return {"messages": [final_message]}
                 
        except Exception as e:
             print(f"PAA Agent MCP Error: {e}")
             return {"messages": [AIMessage(content=f"I encountered an error trying to connect to Zerodha Kite via MCP: {str(e)}")]}

    else:
        print("PAA Agent: Using standard uploaded document flow...")
        # Flow B: Uploaded PDF / CSV Data
        
        # Step 1: Read Portfolio Data (CSV/Excel)
        try:
            portfolio_content = csv_excel_reader_tool.invoke({"query": ""})
        except Exception as e:
            portfolio_content = f"Error reading portfolio files: {str(e)}"
    
        # Step 2: Read Text/PDF Data (if any)
        try:
            pdf_content = pdf_upload_reader_tool.invoke({"query": ""})
        except Exception as e:
            pdf_content = f"Error reading PDF files: {str(e)}"
    
        combined_data = f"=== CSV/EXCEL PORTFOLIO DATA ===\n{portfolio_content}\n\n=== PDF DOCUMENT CONTENT ===\n{pdf_content}\n"
    
        context_message = SystemMessage(content=f"Here is the user's uploaded portfolio data:\n{combined_data}\n\nPlease analyze this portfolio.")
        augmented_messages = messages + [context_message]
        
        print(f"PAA Agent: Analyzed portfolio data. Length: {len(combined_data)}")
    
        response = paa_agent.invoke({"messages": augmented_messages})
        
        # Handle manual tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"PAA Agent: Tool calls detected: {len(response.tool_calls)}")
            
            tool_messages = []
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                tool_id = tool_call['id']
                
                tool_output = ""
                if tool_name == 'fetch_ticker_data':
                    tool_output = str(fetch_ticker_data.invoke(tool_args))
                elif tool_name == 'tavily_search_results_json':
                    tool_output = str(tavily_tool.invoke(tool_args))
                else:
                    print(f"PAA Agent: Warning - Unhandled tool call: {tool_name}")
                    tool_output = f"Tool {tool_name} is not available or already executed. Please use the data provided in context."
                
                tool_messages.append(ToolMessage(
                    tool_call_id=tool_id,
                    content=tool_output,
                    name=tool_name
                ))
            
            if tool_messages:
                print("PAA Agent: Tools executed, generating final response.")
                final_aug_messages = (
                    augmented_messages 
                    + [response] 
                    + tool_messages
                    + [SystemMessage(content="All requested external data has been provided. Please now synthesize your final detailed analysis based on the portfolio and these tool outputs. Do NOT call any more tools.")]
                )
                final_response = paa_agent.invoke({"messages": final_aug_messages})
                return {"messages": [final_response]}
            else:
                if not response.content:
                     return {"messages": [AIMessage(content="I analyzed your portfolio but couldn't fetch additional external data. Here is my analysis based on the provided files:\n" + combined_data[:500] + "...")]}
    
        if not response.content:
            print("PAA Agent: Response content is empty!")
            return {"messages": [AIMessage(content="I'm sorry, I couldn't generate an analysis for your portfolio.")]}
    
        return {"messages": [response]}

