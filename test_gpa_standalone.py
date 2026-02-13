from main import financial_planner, FinancePlannerState
from langchain_core.messages import HumanMessage
import json

def test_gpa_standalone():
    print("Testing Goal Planning Agent (GPA) Standalone...")
    
    # Test Query: Child Education Savings
    query = "I want to save for my child's education in 15 years. Current cost is 20,00,000. How much should I save monthly assuming 10% return and 6% inflation?"
    print(f"\n🧑 User: {query}")
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next_agent": None,
        "user_query": query
    }
    
    config = {"configurable": {"thread_id": "test_gpa_01"}}
    
    try:
        result = financial_planner.invoke(initial_state, config)
        response = result["messages"][-1]
        response_content = response if isinstance(response, str) else response.content
        print(f"🤖 Assistant: {response_content}")
    except Exception as e:
        print(f"❌ Error during GPA test: {e}")
    
    print("\n" + "="*50 + "\n")

    # Test Query: Retirement Goal
    query = "Plan my retirement for 30 years from now. Current expenses are 50,000 monthly. I want to maintain this lifestyle. Assume 8% return and 7% inflation."
    print(f"\n🧑 User: {query}")
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next_agent": None,
        "user_query": query
    }
    
    config = {"configurable": {"thread_id": "test_gpa_02"}}
    
    try:
        result = financial_planner.invoke(initial_state, config)
        response = result["messages"][-1]
        response_content = response if isinstance(response, str) else response.content
        print(f"🤖 Assistant: {response_content}")
    except Exception as e:
        print(f"❌ Error during GPA Retirement test: {e}")

if __name__ == "__main__":
    test_gpa_standalone()
