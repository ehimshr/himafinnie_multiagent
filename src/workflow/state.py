from typing import TypedDict, Annotated, List, Optional
import operator
from langchain_core.messages import BaseMessage

## Define state schema for the financial multi agent system
class FinancePlannerState(TypedDict):
    """Simple state schema for financial multi agent system"""

    # Conversation history - persisted with checkpoint memory
    messages: Annotated[List[BaseMessage], operator.add]

    # Agent routing
    next_agent: Optional[str]

    # Current user query
    user_query: Optional[str]
