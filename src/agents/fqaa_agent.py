from typing import List, Union
import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.core.llm import llm
from src.rag.rag_engine import retriever_tool, retriever
from src.utils.tools import tavily_tool
from src.workflow.state import FinancePlannerState

## Tools for FQAA agent - RAG retriever and TavilySearch
fqaa_tools = [retriever_tool, tavily_tool]

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

# Create FQAA agent
## Bind tools to the LLM
fqaa_llm_with_tools = llm.bind_tools(fqaa_tools)
fqaa_agent = fqaa_system_prompt | fqaa_llm_with_tools | StrOutputParser()

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
