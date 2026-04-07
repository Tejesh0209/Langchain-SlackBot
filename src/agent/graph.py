"""LangGraph state machine for the Northstar Slack Bot."""
import uuid
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agent.classifier import QueryType
from src.agent.state import AgentState
from src.agent.nodes.classify import classify
from src.agent.nodes.sql_agent import sql_agent
from src.agent.nodes.rag_search import rag_search
from src.agent.nodes.multi_search import multi_search
from src.agent.nodes.evaluate import evaluate
from src.agent.nodes.generate import generate
from src.agent.nodes.format import format_slack
from src.agent.cache import query_cache
from src.config import settings


def route_query(state: AgentState) -> Literal["sql_agent", "rag_search", "multi_search"]:
    """Route to the appropriate tool based on query type."""
    query_type = state.get("query_type", "multi_hop")

    if query_type == QueryType.STRUCTURED:
        return "sql_agent"
    elif query_type == QueryType.DOCUMENT:
        return "rag_search"
    else:
        return "multi_search"


def should_evaluate(state: AgentState) -> Literal["evaluate", "generate"]:
    """Skip evaluation for simple structured queries — go straight to generate."""
    if state.get("query_type") == QueryType.STRUCTURED:
        return "generate"
    return "evaluate"


def should_retry(state: AgentState) -> Literal["reformulate", "generate"]:
    """Determine if we should retry or proceed to generation."""
    results = state.get("results", [])
    retry_count = state.get("retry_count", 0)

    # If no results or all errors, and we haven't exceeded retries
    if not results or all(r.get("error") for r in results):
        if retry_count < settings.max_retry_count:
            return "reformulate"
        else:
            return "generate"

    # If we have results, proceed to generation
    return "generate"


def reformulate_query(state: AgentState) -> AgentState:
    """
    Reformulate the query for retry.

    Increments retry_count and modifies the plan.
    """
    retry_count = state.get("retry_count", 0) + 1

    # Expand the search if first retry
    plan = state.get("plan", [])
    if retry_count == 1:
        # Broaden the search
        new_plan = [f"{p} (expanded)" for p in plan]
        if not new_plan:
            new_plan = ["Broaden search query", "Check additional data sources"]
    else:
        new_plan = plan

    return {
        **state,
        "retry_count": retry_count,
        "plan": new_plan,
    }


async def agent_node(state: AgentState) -> AgentState:
    """
    Main agent entry point that routes to appropriate search.

    This is a wrapper that follows the plan if multi-hop.
    """
    query_type = state.get("query_type", "multi_hop")

    if query_type == QueryType.STRUCTURED:
        return await sql_agent(state)
    elif query_type == QueryType.DOCUMENT:
        return await rag_search(state)
    else:
        return await multi_search(state)


def build_graph() -> StateGraph:
    """Build the LangGraph state machine."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify", classify)
    graph.add_node("agent", agent_node)
    graph.add_node("evaluate", evaluate)
    graph.add_node("reformulate", reformulate_query)
    graph.add_node("generate", generate)
    graph.add_node("format", format_slack)

    # Add edges
    graph.add_edge(START, "classify")

    # Conditional routing after classification
    graph.add_conditional_edges(
        "classify",
        route_query,
        {
            "sql_agent": "agent",
            "rag_search": "agent",
            "multi_search": "agent",
        }
    )

    # After agent: structured queries skip evaluate, others go through it
    graph.add_conditional_edges(
        "agent",
        should_evaluate,
        {"evaluate": "evaluate", "generate": "generate"},
    )

    # Conditional routing after evaluation
    graph.add_conditional_edges(
        "evaluate",
        should_retry,
        {
            "reformulate": "reformulate",
            "generate": "generate",
        }
    )

    # Retry loops back to agent
    graph.add_edge("reformulate", "agent")

    # Final steps
    graph.add_edge("generate", "format")
    graph.add_edge("format", END)

    return graph


def compile_graph():
    """Compile the graph with memory checkpointer."""
    checkpointer = MemorySaver()
    app = build_graph().compile(checkpointer=checkpointer)
    return app


# Singleton compiled graph
_app = None


def get_app():
    """Get or create the compiled LangGraph app."""
    global _app
    if _app is None:
        _app = compile_graph()
    return _app


async def run_agent(
    user_message: str,
    channel_id: str,
    thread_ts: str,
    thinking_msg_ts: str = "",
    conversation_history: list[dict] = [],
) -> dict:
    """
    Run the agent with the given message.

    Args:
        user_message: The user's question
        channel_id: Slack channel ID
        thread_ts: Slack thread timestamp
        thinking_msg_ts: Timestamp of the thinking message to update

    Returns:
        Final state after graph execution.
    """
    # Check cache first — skip the entire agent pipeline on a hit
    cached = query_cache.get(user_message)
    if cached is not None:
        return cached

    app = get_app()

    # Initial state
    messages = conversation_history + [{"role": "user", "content": user_message}]
    initial_state = AgentState(
        messages=messages,
        query_type="multi_hop",
        entities=[],
        plan=[],
        results=[],
        sources=[],
        retry_count=0,
        thinking_msg_ts=thinking_msg_ts,
        channel_id=channel_id,
        thread_ts=thread_ts,
    )

    # Thread config for checkpointer
    config = {
        "configurable": {
            "thread_id": thread_ts or str(uuid.uuid4()),
        }
    }

    # Run the graph
    final_state = await app.ainvoke(initial_state, config)

    # Store in cache for future identical queries
    query_cache.set(user_message, final_state)

    return final_state
