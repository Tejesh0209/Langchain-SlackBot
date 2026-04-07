"""Agent state definitions for LangGraph."""
from typing import Annotated, TypedDict

from langgraph.graph import add_messages


class AgentState(TypedDict):
    """State definition for the LangGraph agent."""

    messages: Annotated[list, add_messages]  # Conversation history
    query_type: str  # structured | document | multi_hop
    entities: list[str]  # Extracted entities
    plan: list[str]  # Execution plan steps
    results: list[dict]  # Retrieved results from tools
    sources: list[str]  # Source citations
    retry_count: int  # Self-reflection retry counter (max 2)
    thinking_msg_ts: str  # Slack message ts for progress updates
    channel_id: str  # Slack channel
    thread_ts: str  # Slack thread
