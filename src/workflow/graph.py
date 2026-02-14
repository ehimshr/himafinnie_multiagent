from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.core.llm import llm
from src.workflow.state import FinancePlannerState
from src.agents.fqaa_agent import fqaa_agent_node
from src.agents.maa_agent import maa_agent_node
from src.agents.nsa_agent import nsa_agent_node
from src.agents.tea_agent import tea_agent_node
from src.agents.gpa_agent import gpa_agent_node


# Create Router Function 
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
                "PAA": "paa_agent", # PAA is not implemented yet in the original code
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

# Router Node
def router_node(state: FinancePlannerState):
    """Router node - determines which agent should handle the query"""
    # Create the router if not exists (but it is improved to be global or outside)
    # Re-using global router function
    
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


# Build the complete financial planning graph
workflow = StateGraph(FinancePlannerState)

# Add all nodes to the graph
workflow.add_node("router", router_node)
workflow.add_node("fqaa_agent", fqaa_agent_node)
# workflow.add_node("paa_agent", paa_agent_node) # Not implemented
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
        "tea_agent": "tea_agent",
        "paa_agent": "fqaa_agent" # Fallback since PAA is not implemented
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

if __name__ == "__main__":
    print("✅ Financial Planning Graph built successfully!")
    
    def test_system(query, thread_id="test_thread"):
        """Test our multi-agent system"""
        print(f"🧑 User: {query}")

        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "next_agent": ""
        }

        config = {"configurable": {"thread_id": thread_id}}

        # Run the system
        result = financial_planner.invoke(initial_state, config)

        response = result["messages"][-1] if isinstance(result["messages"][-1], str) else result["messages"][-1].content
        print(f"🤖 Assistant: {response}")
        print("-" * 50)

    test_system("My salary is 1,20,000 monthly and my monthly expenses are 50,000. Plan the best approach for the surplus: how much for cash in hand, mutual funds, equity, term insurance, and health insurance?", thread_id="test_thread_gpa_surplus_01")
