"""Query classification using LLM."""
import json
from enum import Enum

from langchain_openai import ChatOpenAI

from src.config import settings


class QueryType(str, Enum):
    """Query classification types."""

    STRUCTURED = "structured"  # SQL-only queries
    DOCUMENT = "document"  # Semantic search needed
    MULTI_HOP = "multi_hop"  # Both SQL + document + cross-referencing


CLASSIFICATION_PROMPT = """You are a query classifier for a Slack Q&A bot.

Classify the user's query into one of three categories:

1. **structured** - Can be answered with SQL alone (names, counts, values, dates, simple lookups)
   Examples: "How many customers do we have?", "What's BlueHarbor's account status?", "List products in the taxonomy category"

2. **document** - Needs full-text/semantic search in artifacts (call transcripts, Slack threads, implementation notes)
   Examples: "What did BlueHarbor say about the taxonomy rollout issue?", "Summarize the implementation challenges for Verdant Bay"

3. **multi_hop** - Needs both SQL + document search + cross-referencing across multiple sources
   Examples: "Which customer is most likely to defect to a competitor?", "What are the common implementation issues across North America West customers?"

Query: {query}

Respond with JSON only:
{{
    "query_type": "structured" | "document" | "multi_hop",
    "entities": ["customer_name", "product_name", ...],
    "plan": ["step 1", "step 2", ...],
    "reasoning": "brief explanation"
}}
"""


def classify_query(query: str, llm: ChatOpenAI | None = None) -> dict:
    """
    Classify a user query and extract entities.

    Args:
        query: The user's natural language query
        llm: Optional ChatOpenAI instance (for testing/mocking)

    Returns:
        Dict with query_type, entities, plan, and reasoning.
    """
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)

    prompt = CLASSIFICATION_PROMPT.format(query=query)

    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)

    # Parse JSON response
    try:
        result = json.loads(content)
        return {
            "query_type": result.get("query_type", QueryType.MULTI_HOP),
            "entities": result.get("entities", []),
            "plan": result.get("plan", []),
            "reasoning": result.get("reasoning", ""),
        }
    except json.JSONDecodeError:
        # Fallback to multi_hop if parsing fails
        return {
            "query_type": QueryType.MULTI_HOP,
            "entities": [],
            "plan": ["Process query as multi-hop"],
            "reasoning": "Failed to parse LLM response",
        }
