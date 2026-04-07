"""Tests for SQL validation."""
import pytest

from tools.sql_tool import validate_sql_query, BLOCKED_KEYWORDS, SYSTEM_TABLES


class TestSQLValidation:
    """Tests for SQL query validation."""

    def test_valid_select(self):
        is_valid, error = validate_sql_query("SELECT * FROM customers LIMIT 10")
        assert is_valid is True
        assert error == ""

    def test_valid_select_with_where(self):
        is_valid, error = validate_sql_query(
            "SELECT name, account_status FROM customers WHERE country = 'Canada' LIMIT 5"
        )
        assert is_valid is True

    def test_empty_query(self):
        is_valid, error = validate_sql_query("")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_whitespace_only(self):
        is_valid, error = validate_sql_query("   ")
        assert is_valid is False

    def test_drop_blocked(self):
        is_valid, error = validate_sql_query("DROP TABLE customers")
        assert is_valid is False
        assert "blocked" in error.lower()

    def test_delete_blocked(self):
        is_valid, error = validate_sql_query("DELETE FROM customers WHERE id = 1")
        assert is_valid is False
        assert "blocked" in error.lower()

    def test_update_blocked(self):
        is_valid, error = validate_sql_query("UPDATE customers SET name = 'hacked'")
        assert is_valid is False
        assert "blocked" in error.lower()

    def test_insert_blocked(self):
        is_valid, error = validate_sql_query("INSERT INTO customers (name) VALUES ('hacker')")
        assert is_valid is False
        assert "blocked" in error.lower()

    def test_attach_blocked(self):
        is_valid, error = validate_sql_query("ATTACH DATABASE '/tmp/hack.db' AS hack")
        assert is_valid is False
        assert "blocked" in error.lower()

    def test_system_table_blocked(self):
        is_valid, error = validate_sql_query("SELECT * FROM sqlite_master")
        assert is_valid is False
        # sqlite_master is blocked as a system table AND matches blocked pattern
        assert "blocked" in error.lower()

    def test_sqlite_sequence_blocked(self):
        is_valid, error = validate_sql_query("SELECT * FROM sqlite_sequence")
        assert is_valid is False
        assert "system table" in error.lower()

    def test_pragma_blocked(self):
        is_valid, error = validate_sql_query("PRAGMA table_info(customers)")
        assert is_valid is False
        assert "blocked" in error.lower()

    def test_sql_comment_blocked(self):
        is_valid, error = validate_sql_query("SELECT * FROM customers; -- DROP TABLE")
        assert is_valid is False

    def test_case_insensitive(self):
        """Blocked keywords should be caught regardless of case."""
        is_valid, error = validate_sql_query("drop table customers")
        assert is_valid is False

        is_valid, error = validate_sql_query("DELETE FROM customers")
        assert is_valid is False

        is_valid, error = validate_sql_query("SeLeCt * FrOm CuStOmErS")
        assert is_valid is True  # SELECT is not blocked

    def test_complex_valid_query(self):
        """Test that complex valid queries pass."""
        query = """
        SELECT c.name, c.account_status, COUNT(a.id) as artifact_count
        FROM customers c
        LEFT JOIN artifacts a ON c.id = a.customer_id
        WHERE c.region = 'North America West'
        GROUP BY c.id
        LIMIT 20
        """
        is_valid, error = validate_sql_query(query)
        assert is_valid is True
