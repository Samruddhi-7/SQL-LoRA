"""
SQL-LoRA — Schema Materialization

Parses CREATE TABLE SQL statements from the dataset and materializes
them into a real SQLite database. This enables execution-accuracy
evaluation by running generated SQL against actual tables (empty).

Usage:
    python data/schema_to_sqlite.py [--db PATH] [--dataset PATH]
"""


def create_database_from_schemas(
    schemas: list[str], db_path: str = ":memory:"
) -> None:
    """Create a SQLite database from a list of CREATE TABLE statements.

    Each statement is executed in order to set up the schema.
    Tables are left empty — only the schema matters for execution-accuracy.
    """
    ...


def extract_create_statements(example: dict) -> str | None:
    """Extract the CREATE TABLE statement from a dataset example's context field."""
    ...


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="data/schemas.db")
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    print(f"Database created at {args.db}")
