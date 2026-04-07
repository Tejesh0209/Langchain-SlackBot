"""RAG search node for LangGraph."""
from src.agent.state import AgentState
from tools.rag_tool import rag_tool


async def rag_search(state: AgentState) -> AgentState:
    """
    Perform RAG search on artifacts.

    For document queries that need full-text/semantic search.
    """
    last = state["messages"][-1] if state["messages"] else None
    user_message = (last.content if hasattr(last, "content") else last.get("content", "")) if last else ""
    entities = state.get("entities", [])

    # Build search query
    search_query = user_message
    if entities:
        # Append entity names to improve search
        search_query = f"{user_message} {' '.join(entities)}"

    # Perform hybrid search — higher alpha (0.75) favours dense/semantic search
    results = await rag_tool.search(query=search_query, limit=10, alpha=0.75)

    # Extract sources
    sources = []
    for r in results:
        if "artifact_id" in r:
            sources.append(f"Artifact:{r['artifact_id']} ({r.get('artifact_type', 'unknown')})")

    return {
        **state,
        "results": results,
        "sources": sources,
    }
