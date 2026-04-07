"""SQLite FTS5 full-text search tool as fallback/supplement."""
import re
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from src.config import settings


class FTSTool:
    """SQLite FTS5 full-text search tool."""

    # FTS5 virtual table name
    FTS_TABLE = "artifacts_fts"

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.database_path

    async def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Search artifacts using FTS5.

        Args:
            query: FTS5 query string
            limit: Maximum number of results

        Returns:
            List of matching artifact dicts.
        """
        # Sanitize query - escape special FTS5 characters except AND/OR/NOT
        sanitized = self._sanitize_fts_query(query)

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row

                # Check if FTS table exists using PRAGMA (bypasses sqlite_master check)
                cursor = await db.execute("PRAGMA table_list")
                tables = await cursor.fetchall()
                table_names = [t['name'] for t in tables]
                if self.FTS_TABLE not in table_names:
                    return [{"error": f"FTS table '{self.FTS_TABLE}' does not exist"}]

                # Search using FTS5 - join with artifacts table to get content
                sql = f"""
                SELECT a.artifact_id, a.customer_id, a.artifact_type,
                       a.title, a.summary, a.content_text, a.created_at
                FROM {self.FTS_TABLE} f
                JOIN artifacts a ON f.rowid = a.rowid
                WHERE {self.FTS_TABLE} MATCH ?
                ORDER BY rank
                LIMIT ?
                """

                async with db.execute(sql, (sanitized, limit)) as cursor:
                    rows = await cursor.fetchall()
                    results = []
                    for row in rows:
                        r = dict(row)
                        r['content'] = r.get('content_text', '')  # Alias for compatibility
                        results.append(r)
                    return results

        except Exception as e:
            return [{"error": str(e)}]

    async def setup_fts(self) -> dict[str, Any]:
        """
        Create or recreate the FTS5 virtual table.
        Should be called during data ingestion.

        Returns:
            Dict with success/error status.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Drop existing FTS table if exists
                await db.execute(f"DROP TABLE IF EXISTS {self.FTS_TABLE}")

                # Create FTS5 table
                await db.execute(f"""
                    CREATE VIRTUAL TABLE {self.FTS_TABLE} USING fts5(
                        content,
                        content='artifacts',
                        content_rowid='id'
                    )
                """)

                # Populate from artifacts table
                await db.execute(f"""
                    INSERT INTO {self.FTS_TABLE}(rowid, content)
                    SELECT id, content FROM artifacts
                """)

                # Create index
                await db.execute(f"""
                    CREATE INDEX idx_{self.FTS_TABLE}_customer
                    ON {self.FTS_TABLE}(rowid)
                """)

                await db.commit()

                return {"success": True, "message": f"FTS table '{self.FTS_TABLE}' created"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _sanitize_fts_query(self, query: str) -> str:
        """
        Sanitize user input for FTS5 query.

        FTS5 special characters: " * ^ - + : ( )
        We preserve AND/OR/NOT and simple quoted strings.
        """
        # Simple sanitization - remove or escape problematic chars
        sanitized = re.sub(r'[*^:()]+', ' ', query)
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        return sanitized if sanitized else query


# Singleton instance
fts_tool = FTSTool()
