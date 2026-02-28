from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.core.llm import llm
from src.rag.rag_engine import retriever_tool
from src.utils.tools import sip_calculator_tool, inflation_calculator_tool
from src.workflow.state import FinancePlannerState

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

# GPA also uses base LLM for final synthesis of goal planning data
gpa_agent = gpa_system_prompt | llm | StrOutputParser()

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
