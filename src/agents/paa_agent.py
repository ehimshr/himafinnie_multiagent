from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.core.llm import llm
from src.utils.tools import csv_excel_reader_tool, pdf_upload_reader_tool, fetch_ticker_data, tavily_tool
from src.workflow.state import FinancePlannerState
import json

# Tools for PAA
# Tools for PAA
# Note: We do NOT include file reader tools here because the node pre-loads the data.
# We only give the agent tools to fetch *new* external data.
paa_tools = [fetch_ticker_data, tavily_tool]

# System prompt for the Portfolio Analysis Agent (PAA)
paa_system_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the Portfolio Analysis Agent (PAA), modeling your investment philosophy after legendary investors Benjamin Graham and Warren Buffett.
Your goal is to analyze user portfolios with a focus on value investing, long-term growth, margin of safety, and proper diversification.

SCOPE & BEHAVIOR RULES:
- Analyze uploaded portfolio data (CSV/Excel/PDF).
- Evaluate diversification (Sector, Asset Class).
- Provide specific Buy/Sell/Hold/Average advice for individual stocks based on fundamental principles.
- Use 'fetch_ticker_data' to get current prices if needed.
- Use 'tavily_tool' to get recent news impacting specific holdings.

ANALYSIS FRAMEWORK:
1. **Diversification Check**: Is the portfolio too concentrated? Are there too many correlated assets?
2. **Stock Analysis (Buffett/Graham Style)**:
   - Look for strong fundamentals (though you may not have full balance sheets, use proxy info or general knowledge).
   - Advise "Hold" or "Buy" for quality companies with a moat.
   - Advise "Sell" for speculative assets or deteriorating fundamentals.
   - Advise "Average" if a quality stock is down but fundamentals remain strong.
3. **Actionable Advice**: Be direct. "Buy more of X", "Trim position in Y".

RESPONSE FORMAT:
## 💼 Portfolio Overview
- **Diversification Status**: (Good / Poor / Needs Improvement)
- **Sector Allocation**:
## 📊 Individual Stock Analysis
| Stock | Action (Buy/Sell/Hold/Avg) | Rationale (Buffett Style) |
|-------|----------------------------|---------------------------|
| ...   | ...                        | ...                       |
## 🧠 Strategic Advice
- **Graham's Wisdom**:
- **Buffett's Insight**:
## ⚠️ Risk Assessment
"""),
    MessagesPlaceholder(variable_name="messages"),
])

# Define the Agent
paa_llm_with_tools = llm.bind_tools(paa_tools)
paa_agent = paa_system_prompt | paa_llm_with_tools

# PAA Agent node functions
def paa_agent_node(state: FinancePlannerState):
    """Portfolio Analysis Agent (PAA) node"""
    messages = state["messages"]
    
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

    combined_data = f"""=== CSV/EXCEL PORTFOLIO DATA ===
{portfolio_content}

=== PDF DOCUMENT CONTENT ===
{pdf_content}
"""

    context_message = SystemMessage(content=f"Here is the user's uploaded portfolio data:\n{combined_data}\n\nPlease analyze this portfolio.")
    augmented_messages = messages + [context_message]
    
    # Invoking the agent loop (single step for synthesis here, but in a real ReAct loop it would call tools)
    # Since we pre-fetched the file content, the agent can now use fetch_ticker_data or tavily if it needs MORE info.
    # However, for simplicity in this node, we will let the LLM synthesize the initial analysis. 
    # If the LLM wants to call tools, it will return tool calls.
    
    # For now, let's force a synthesis pass with the data we have, 
    # but strictly speaking, the agent might want to look up current prices for the stocks found in the CSV.
    # To enable that, we allow the agent to inspect the data and decide.
    
    print(f"PAA Agent: Analyzed portfolio data. Length: {len(combined_data)}")

    response = paa_agent.invoke({"messages": augmented_messages})
    
    # Handle tool calls (e.g., if it wants to fetch current prices for stocks in the list)
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
            
            # Create proper ToolMessage
            tool_messages.append(ToolMessage(
                tool_call_id=tool_id,
                content=tool_output,
                name=tool_name
            ))
        
        if tool_messages:
            print("PAA Agent: Tools executed, generating final response.")
            # Append AIMessage (with tool_calls) and then the ToolMessages
            # Add a directive to force synthesis and prevent further tool calls (since we only support 1 loop here)
            final_aug_messages = (
                augmented_messages 
                + [response] 
                + tool_messages
                + [SystemMessage(content="All requested external data has been provided. Please now synthesize your final detailed analysis based on the portfolio and these tool outputs. Do NOT call any more tools.")]
            )
            final_response = paa_agent.invoke({"messages": final_aug_messages})
            return {"messages": [final_response]}
        else:
            print("PAA Agent: Tool calls existed but no messages generated (should not happen).")
            # If for some reason we have calls but no messages, return initial response to avoid error,
            # but ideally we should have handled them.
            if not response.content:
                 return {"messages": [AIMessage(content="I analyzed your portfolio but couldn't fetch additional external data. Here is my analysis based on the provided files:\n" + combined_data[:500] + "...(rest of data)...")]}

    # If no tool calls, return the response
    if not response.content:
        print("PAA Agent: Response content is empty!")
        return {"messages": [AIMessage(content="I'm sorry, I couldn't generate an analysis for your portfolio. Please check if the file format is correct.")]}

    return {"messages": [response]}
