"""Multi-search node for LangGraph (SQL + RAG combined)."""
from langchain_openai import ChatOpenAI

from src.agent.state import AgentState
from src.agent import progress
from src.config import settings
from tools.sql_tool import sql_tool, get_schema_context, validate_sql_query
from tools.rag_tool import rag_tool


SQL_PROMPT = """You are a SQL expert. Generate a SQLite SELECT query to answer the user's question.

Rules:
- Only SELECT statements, no semicolons at the end
- Use proper SQLite syntax
- Include LIMIT clause (max {row_limit} rows)
- For "top N", "highest", "lowest" ranking questions: use LIMIT 5 and ORDER BY the relevant column
- Use JOINs when needed across tables
- ALWAYS include in SELECT the columns you filter on (e.g. if filtering by account_health, include account_health in SELECT)
- account_health values are lowercase: 'at risk', 'expanding', 'healthy', 'recovering', 'watch list'

{schema}

User question: {question}

Generate only the SQL query, no explanation.
"""


async def multi_search(state: AgentState) -> AgentState:
    """
    Perform combined SQL + RAG search.

    For multi-hop queries that need both structured data and documents.
    """
    await progress.report(state.get("thread_ts", ""), "search")
    last = state["messages"][-1] if state["messages"] else None
    user_message = (last.content if hasattr(last, "content") else last.get("content", "")) if last else ""
    entities = state.get("entities", [])

    all_results = []
    all_sources = []

    # Step 1: Always run LLM-generated SQL (not just entity-based)
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)
    prompt = SQL_PROMPT.format(
        schema=get_schema_context(),
        question=user_message,
        row_limit=settings.sql_row_limit,
    )
    response = await llm.ainvoke(prompt)
    sql_query = response.content if hasattr(response, "content") else str(response)
    sql_query = sql_query.strip()
    if sql_query.startswith("```"):
        sql_query = sql_query.split("\n", 1)[-1]
        sql_query = sql_query.rsplit("```", 1)[0].strip()
    sql_query = sql_query.rstrip(";").strip()

    is_valid, error = validate_sql_query(sql_query)
    if is_valid:
        sql_result = await sql_tool.execute(sql_query)
        if not sql_result.get("error") and sql_result.get("rows"):
            rows = sql_result["rows"]
            all_results.extend(rows)
            all_sources.append(f"SQL: {sql_query}")
    else:
        # Fallback: entity-based query if LLM SQL fails
        if entities:
            placeholders = ", ".join(["?" for _ in entities])
            fallback_sql = f"""
            SELECT c.name, c.account_health, c.country, c.region,
                   c.tech_stack_summary, c.notes, c.contract_value,
                   a.artifact_type, a.title, a.content_text, a.created_at
            FROM customers c
            LEFT JOIN artifacts a ON c.customer_id = a.customer_id
            WHERE c.name IN ({placeholders})
            LIMIT 20
            """
            sql_result = await sql_tool.execute(fallback_sql, tuple(entities))
            if not sql_result.get("error"):
                all_results.extend(sql_result.get("rows", []))
                all_sources.append("SQL query on customers and artifacts")

    # Step 2: RAG search (optional - skip gracefully if Weaviate is down)
    try:
        search_query = f"{user_message} {' '.join(entities)}" if entities else user_message

        # Pattern/cross-account queries need more chunks to find commonalities
        pattern_terms = {"pattern", "recurring", "common", "across", "multiple", "bypass", "trend", "similar"}
        query_words = set(search_query.lower().split())
        rag_limit = 15 if pattern_terms & query_words else 8

        rag_results = await rag_tool.search(query=search_query, limit=rag_limit, alpha=0.7)
        if rag_results and not rag_results[0].get("error"):
            all_results.extend(rag_results)
            for r in rag_results:
                if "artifact_id" in r:
                    all_sources.append(f"Artifact:{r['artifact_id']} ({r.get('artifact_type', 'unknown')})")
    except Exception:
        pass  # Weaviate may not be running; SQL results are sufficient

    return {
        **state,
        "results": all_results,
        "sources": all_sources,
    }
