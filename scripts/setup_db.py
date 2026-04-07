"""Database setup and verification script."""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.sql_tool import sql_tool


async def verify_database() -> bool:
    """
    Verify the SQLite database exists and has expected schema.

    Returns True if successful, False otherwise.
    """
    print(f"Checking database at: {sql_tool.db_path}")

    if not sql_tool.db_path.exists():
        print(f"ERROR: Database file not found: {sql_tool.db_path}")
        return False

    print("Database file exists.")

    # Get schema
    schema = await sql_tool.get_schema()
    print(f"\nSchema:\n{schema}")

    # Check for expected tables
    expected_tables = {"customers", "products", "implementations", "artifacts", "communications"}
    found_tables = set()

    for line in schema.split("\n"):
        if "table" in line.lower():
            # Extract table name
            parts = line.split()
            for i, part in enumerate(parts):
                if part.lower() == "table" and i + 1 < len(parts):
                    found_tables.add(parts[i + 1].strip("("))
                    break

    missing = expected_tables - found_tables
    if missing:
        print(f"\nWARNING: Missing expected tables: {missing}")
    else:
        print("\nAll expected tables found.")

    # Get row counts
    print("\nRow counts:")
    for table in sorted(found_tables):
        result = await sql_tool.execute(f"SELECT COUNT(*) as count FROM {table}")
        if not result.get("error"):
            print(f"  {table}: {result['rows'][0]['count']}")

    return True


async def setup_fts() -> bool:
    """
    Set up FTS5 virtual table for full-text search.

    Returns True if successful.
    """
    from tools.fts_tool import fts_tool

    print("\nSetting up FTS5...")
    result = await fts_tool.setup_fts()

    if result.get("success"):
        print(f"FTS setup successful: {result.get('message')}")
        return True
    else:
        print(f"FTS setup failed: {result.get('error')}")
        return False


async def main():
    """Main entry point."""
    print("=" * 60)
    print("Northstar Signal - Database Setup")
    print("=" * 60)

    db_ok = await verify_database()
    if not db_ok:
        return 1

    # Optionally setup FTS
    if "--fts" in sys.argv:
        fts_ok = await setup_fts()
        if not fts_ok:
            return 1

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
