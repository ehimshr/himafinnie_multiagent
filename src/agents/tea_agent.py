import json
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.core.llm import llm
from src.rag.rag_engine import retriever_tool
from src.utils.tools import calculator_tool, tavily_tool, pdf_upload_reader_tool
from src.workflow.state import FinancePlannerState

## Tools for TEA agent - RAG, Calculator, Tavily, and PDF Reader
tea_tools = [retriever_tool, calculator_tool, tavily_tool, pdf_upload_reader_tool]

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

# TEA also uses base LLM for final synthesis of gathered tax data
tea_agent = tea_system_prompt | llm | StrOutputParser()

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
