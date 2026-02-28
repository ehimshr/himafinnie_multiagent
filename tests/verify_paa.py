from src.workflow.graph import router
from src.workflow.state import FinancePlannerState
from langchain_core.messages import HumanMessage

def test_paa_routing():
    queries = [
        "Analyze my portfolio",
        "Here is my stock list, please review it",
        "Review my uploaded portfolio CSV",
        "What should I do with my holdings?"
    ]
    
    print("Testing PAA Routing...")
    for q in queries:
        state = {"messages": [HumanMessage(content=q)]}
        next_agent = router(state)
        print(f"Query: '{q}' -> Agent: {next_agent}")
        if next_agent == "paa_agent":
            print("✅ PASS")
        else:
            print("❌ FAIL")

if __name__ == "__main__":
    test_paa_routing()
