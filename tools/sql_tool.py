"""Read-only SQL execution with guardrails."""
import re
from pathlib import Path
from typing import Any, Optional, Union

import aiosqlite

from src.config import settings


# Blocked SQL keywords and patterns
BLOCKED_KEYWORDS = [
    r"\bDROP\b",
    r"\bDELETE\b",
    r"\bUPDATE\b",
    r"\bINSERT\b",
    r"\bATTACH\b",
    r"\bCREATE\b",
    r"\bALTER\b",
    r"\bPRAGMA\b",
    r"\bsqlite_master\b",
    r"\bsqlite_temp\b",
    r"\b--\b",  # SQL comments
    r";\s*$",  # Trailing semicolon with more statements
]

# System tables to block
SYSTEM_TABLES = {
    "sqlite_master",
    "sqlite_temp_master",
    "sqlite_sequence",
}

# Column names that indicate system/sensitive data
SENSITIVE_COLUMNS = {
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
}


def validate_sql_query(query: str) -> tuple[bool, str]:
    """
    Validate a SQL query for safety.

    Returns (is_valid, error_message).
    """
    if not query or not query.strip():
        return False, "Empty query"

    # Normalize whitespace
    normalized = re.sub(r"\s+", " ", query.strip())

    # Check for blocked keywords
    for pattern in BLOCKED_KEYWORDS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return False, f"Blocked keyword or pattern detected: {pattern}"

    # Check for system tables in FROM/JOIN
    lower_query = normalized.lower()
    for sys_table in SYSTEM_TABLES:
        if sys_table in lower_query:
            return False, f"Access to system table blocked: {sys_table}"

    return True, ""


def get_schema_context() -> str:
    """
    Return schema context for the database.
    Used by the LLM to generate correct queries.
    """
    return """
    Schema:
    - customers (customer_id TEXT PK, name TEXT, industry, subindustry, region, country, size_band,
                 employee_count INTEGER, annual_revenue_band TEXT, account_health TEXT, crm_stage,
                 tech_stack_summary, notes)
      NOTE: annual_revenue_band is TEXT (e.g. '10M-50M'). account_health values: 'at risk', 'expanding', 'healthy', 'recovering', 'watch list'.
    - implementations (implementation_id TEXT PK, customer_id TEXT FK->customers.customer_id,
                       product_id, contract_value INTEGER, status, kickoff_date, go_live_date, scope_summary)
      NOTE: contract_value is numeric (e.g. 780000). To get contract values by customer NAME, always JOIN: customers c JOIN implementations i ON c.customer_id = i.customer_id
    - products (product_id TEXT PK, name TEXT, category, description, pricing_model)
    - artifacts (artifact_id TEXT PK, customer_id FK->customers.customer_id, artifact_type, title, content_text, summary, created_at)
    - competitors (competitor_id TEXT PK, name TEXT, segment, description, pricing_position)
    - employees (employee_id TEXT PK, full_name, title, department, region)
    - scenarios (scenario_id TEXT PK, industry, region, trigger_event, pain_point, scenario_summary)
    - company_profile (company_id TEXT PK, name TEXT, mission, ideal_customer_profile)

    IMPORTANT JOIN RULES:
    - NEVER filter by customer name on the implementations table — use JOIN with customers.name instead
    - Example: SELECT c.name, i.contract_value FROM customers c JOIN implementations i ON c.customer_id = i.customer_id WHERE c.name LIKE '%BlueHarbor%'
    """


class SQLTool:
    """Read-only SQL execution tool with guardrails."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.database_path

    async def execute(
        self, query: str, params: Optional[tuple] = None
    ) -> dict[str, Any]:
        """
        Execute a validated SELECT query and return results.

        Returns dict with 'rows', 'columns', and 'row_count'.
        """
        is_valid, error = validate_sql_query(query)
        if not is_valid:
            return {"error": error, "rows": [], "columns": [], "row_count": 0}

        # Apply row limit
        if not re.search(r"\bLIMIT\b", query, re.IGNORECASE):
            query = f"{query.rstrip(';')} LIMIT {settings.sql_row_limit}"

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []

                    return {
                        "rows": [dict(row) for row in rows],
                        "columns": columns,
                        "row_count": len(rows),
                    }
        except Exception as e:
            return {"error": str(e), "rows": [], "columns": [], "row_count": 0}

    async def _execute_internal(
        self, query: str, params: Optional[tuple] = None
    ) -> dict[str, Any]:
        """
        Internal execute that skips validation.
        Used for schema inspection only.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []

                    return {
                        "rows": [dict(row) for row in rows],
                        "columns": columns,
                        "row_count": len(rows),
                    }
        except Exception as e:
            return {"error": str(e), "rows": [], "columns": [], "row_count": 0}

    async def get_schema(self) -> str:
        """Return the database schema as a string."""
        # Use PRAGMA table_list which is not blocked by the pattern
        # but we use internal execute to bypass validation
        query = "PRAGMA table_list"
        result = await self._execute_internal(query)
        if result.get("error"):
            return f"Error getting schema: {result['error']}"

        schema_parts = []
        for row in result["rows"]:
            table_name = row["name"]
            table_type = row.get("type", "table")

            # Skip system tables
            if table_name.startswith("sqlite_"):
                continue

            # Get columns for this table
            col_query = f"PRAGMA table_info({table_name})"
            col_result = await self._execute_internal(col_query)
            if not col_result.get("error"):
                columns = [f"  {col['name']} {col['type']}" for col in col_result["rows"]]
                schema_parts.append(f"{table_type} {table_name} (\n" + ",\n".join(columns) + "\n)")

        return "\n\n".join(schema_parts)


# Singleton instance
sql_tool = SQLTool()
