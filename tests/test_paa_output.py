from src.agents.paa_agent import paa_agent_node
from src.workflow.state import FinancePlannerState
from langchain_core.messages import HumanMessage
import sys

def test_paa_output():
    print("🧪 Testing PAA Agent Output...")
    
    # Mock state
    state = {
        "messages": [HumanMessage(content="Analyze my uploaded portfolio CSV.")]
    }
    
    try:
        result = paa_agent_node(state)
        response = result["messages"][-1]
        content = response.content if hasattr(response, "content") else str(response)
        
        print("\n📝 Agent Response:")
        print("-" * 50)
        print(content)
        print("-" * 50)
        
        if content and len(content) > 50:
            print("✅ PAA Agent returned valid content.")
        else:
            print("❌ PAA Agent returned empty or too short content.")
            
    except Exception as e:
        print(f"❌ Error invoking PAA Agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_paa_output()
