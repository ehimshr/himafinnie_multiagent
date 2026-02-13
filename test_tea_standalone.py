from main import financial_planner, FinancePlannerState
from langchain_core.messages import HumanMessage
import json

def test_tea_standalone():
    print("Testing TEA Agent Standalone...")
    
    # Test Query 1: Tax Concept & Calculation
    query = "Explain the capital gains tax in India and calculate the tax for a profit of 5,00,000 if it's long-term."
    print(f"\n🧑 User: {query}")
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next_agent": None,
        "user_query": query
    }
    
    config = {"configurable": {"thread_id": "test_tea_01"}}
    
    try:
        result = financial_planner.invoke(initial_state, config)
        response = result["messages"][-1].content
        print(f"🤖 Assistant: {response}")
    except Exception as e:
        print(f"❌ Error during TEA test: {e}")
    
    print("\n" + "="*50 + "\n")

    # Test Query 2: PDF Reading (if any)
    query = "Summary the tax document in my upload folder."
    print(f"\n🧑 User: {query}")
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next_agent": None,
        "user_query": query
    }
    
    config = {"configurable": {"thread_id": "test_tea_02"}}
    
    try:
        result = financial_planner.invoke(initial_state, config)
        response = result["messages"][-1].content
        print(f"🤖 Assistant: {response}")
    except Exception as e:
        print(f"❌ Error during TEA PDF test: {e}")

if __name__ == "__main__":
    test_tea_standalone()
