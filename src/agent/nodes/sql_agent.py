"""SQL agent node for LangGraph."""
import json

from langchain_openai import ChatOpenAI

from src.agent.state import AgentState
from src.config import settings
from src.agent import progress
from tools.sql_tool import sql_tool


SQL_AGENT_PROMPT = """You are a SQL expert. Given a user question, generate a SQL query to answer it.

Rules:
- Only SELECT statements, no semicolons at the end
- Use proper SQLite syntax with LIMIT {row_limit}
- ALWAYS include in SELECT the columns you filter/sort on
- account_health values: 'at risk', 'expanding', 'healthy', 'recovering', 'watch list'
- NEVER filter implementations by customer name — always JOIN customers table:
  customers c JOIN implementations i ON c.customer_id = i.customer_id WHERE c.name LIKE '%CustomerName%'
- For COUNT queries use SELECT COUNT(*) AS total

Schema:
{schema}

User question: {question}

Generate only the SQL query, no explanation."""


async def sql_agent(state: AgentState) -> AgentState:
    """
    Execute SQL query based on the user's question.

    For structured queries that can be answered with SQL alone.
    """
    await progress.report(state.get("thread_ts", ""), "sql")
    last = state["messages"][-1] if state["messages"] else None
    user_message = (last.content if hasattr(last, "content") else last.get("content", "")) if last else ""
    entities = state.get("entities", [])

    # Get schema context
    schema = await sql_tool.get_schema()

    # Build prompt with schema and question
    prompt = SQL_AGENT_PROMPT.format(
        schema=schema,
        question=user_message,
        row_limit=settings.sql_row_limit,
    )

    # Generate SQL via LLM
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)
    response = llm.invoke(prompt)
    sql_query = response.content if hasattr(response, "content") else str(response)
    sql_query = sql_query.strip()
    # Strip markdown code fences if present
    if sql_query.startswith("```"):
        sql_query = sql_query.split("\n", 1)[-1]
        sql_query = sql_query.rsplit("```", 1)[0].strip()
    # Strip trailing semicolon (LLMs always add it, but validator blocks it)
    sql_query = sql_query.rstrip(";").strip()

    # Validate and execute
    from tools.sql_tool import validate_sql_query
    is_valid, error = validate_sql_query(sql_query)

    if not is_valid:
        return {
            **state,
            "results": [{"error": f"SQL validation failed: {error}"}],
            "sources": [],
        }

    # Execute query
    result = await sql_tool.execute(sql_query)

    if result.get("error"):
        return {
            **state,
            "results": [{"error": result["error"]}],
            "sources": [],
        }

    # Format results
    results = result["rows"]
    sources = [f"SQL: {sql_query}"]

    return {
        **state,
        "results": results,
        "sources": sources,
    }
