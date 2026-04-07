"""Classify query node for LangGraph."""
from src.agent.classifier import QueryType, classify_query
from src.agent.state import AgentState
from src.agent import progress


async def classify(state: AgentState) -> AgentState:
    """
    Classify the user's query and extract entities.

    Updates:
        - query_type: The classified query type
        - entities: Extracted entity references
        - plan: Execution plan steps
    """
    await progress.report(state.get("thread_ts", ""), "classify")
    last = state["messages"][-1] if state["messages"] else None
    user_message = (last.content if hasattr(last, "content") else last.get("content", "")) if last else ""

    result = classify_query(user_message)

    return {
        **state,
        "query_type": result["query_type"],
        "entities": result["entities"],
        "plan": result["plan"],
    }
