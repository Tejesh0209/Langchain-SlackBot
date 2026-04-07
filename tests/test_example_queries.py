"""Integration tests against example queries from the assignment.

These tests verify the agent can correctly answer the example queries.
Note: These require the full database and Weaviate to be set up.
"""
import pytest


# Example queries from the assignment
EXAMPLE_QUERIES = [
    # Easy queries (should take 1-3 tool calls)
    {
        "query": "What is BlueHarbor's taxonomy rollout issue?",
        "expected_topics": ["BlueHarbor", "taxonomy", "rollout"],
        "difficulty": "easy",
    },
    {
        "query": "What is Verdant Bay's live patch window?",
        "expected_topics": ["Verdant Bay", "patch", "window"],
        "difficulty": "easy",
    },
    {
        "query": "Tell me about MapleHarvest field mappings",
        "expected_topics": ["MapleHarvest", "field", "mapping"],
        "difficulty": "easy",
    },
    {
        "query": "What are Aureum's SCIM field conflicts?",
        "expected_topics": ["Aureum", "SCIM", "field", "conflict"],
        "difficulty": "easy",
    },
    # Hard queries (should take 3-6 tool calls)
    {
        "query": "Which customer is most likely to defect to a competitor?",
        "expected_topics": ["defect", "competitor", "customer"],
        "difficulty": "hard",
    },
    {
        "query": "What are the common implementation issues in North America West?",
        "expected_topics": ["North America West", "implementation", "issues"],
        "difficulty": "hard",
    },
    {
        "query": "What is the approval-bypass pattern in Canada?",
        "expected_topics": ["Canada", "approval", "bypass"],
        "difficulty": "hard",
    },
]


class TestQueryClassification:
    """Test that queries are classified correctly."""

    def test_easy_queries_classified(self):
        """Easy queries should be classified as document or structured."""
        from src.agent.classifier import classify_query, QueryType

        for eq in EXAMPLE_QUERIES:
            if eq["difficulty"] == "easy":
                result = classify_query(eq["query"])
                # Easy queries should not need multi_hop
                assert result["query_type"] in [QueryType.STRUCTURED, QueryType.DOCUMENT], \
                    f"Easy query misclassified: {eq['query']}"

    def test_hard_queries_classified(self):
        """Hard queries should be classified as multi_hop."""
        from src.agent.classifier import classify_query, QueryType

        for eq in EXAMPLE_QUERIES:
            if eq["difficulty"] == "hard":
                result = classify_query(eq["query"])
                # Hard queries should be multi_hop
                assert result["query_type"] == QueryType.MULTI_HOP, \
                    f"Hard query misclassified: {eq['query']}"


class TestExampleQueries:
    """Integration tests for example queries.

    These are marked as integration tests and require full setup.
    """

    @pytest.mark.integration
    @pytest.mark.parametrize("eq", EXAMPLE_QUERIES, ids=[eq["query"][:30] for eq in EXAMPLE_QUERIES])
    def test_query_produces_response(self, eq):
        """Test that each query produces a non-empty response."""
        import asyncio
        from src.agent.graph import run_agent

        # Skip if not in integration mode
        if not pytest.config.getoption("--integration", default=False):
            pytest.skip("Requires --integration flag")

        async def run():
            result = await run_agent(
                user_message=eq["query"],
                channel_id="test_channel",
                thread_ts="test_thread",
            )
            return result

        result = asyncio.run(run())
        messages = result.get("messages", [])

        assert len(messages) > 0, f"No response for query: {eq['query']}"
        final_message = messages[-1]["content"]
        assert len(final_message) > 0, f"Empty response for query: {eq['query']}"

    @pytest.mark.integration
    def test_structured_query_uses_sql(self):
        """Test that structured queries use the SQL tool."""
        import asyncio
        from src.agent.graph import run_agent

        if not pytest.config.getoption("--integration", default=False):
            pytest.skip("Requires --integration flag")

        async def run():
            result = await run_agent(
                user_message="How many customers do we have?",
                channel_id="test_channel",
                thread_ts="test_thread",
            )
            return result

        result = asyncio.run(run())

        # Should have SQL results
        assert len(result.get("results", [])) > 0
        # Sources should include SQL
        sources = result.get("sources", [])
        assert any("SQL" in s for s in sources)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: integration tests requiring full setup")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests",
    )
