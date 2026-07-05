"""
SQL-LoRA — Schema Materialization

Parses CREATE TABLE SQL statements from the dataset and materializes
them into a real SQLite database. This enables execution-accuracy
evaluation by running generated SQL against actual tables (empty).

Usage:
    python data/schema_to_sqlite.py [--db PATH] [--drop-if-exists]
"""

import sqlite3
import tempfile


def create_database_from_schemas(
    schema_statements: list[str],
    db_path: str | None = None,
) -> tuple[sqlite3.Connection, str]:
    """Create a SQLite database from a list of CREATE TABLE statements.

    Parameters
    ----------
    schema_statements:
        One or more complete SQL DDL statements (e.g. CREATE TABLE ...).
    db_path:
        Path to the SQLite database file. If ``None`` a temporary file is
        created.  Pass ``:memory:`` for an in-memory database (useful for
        testing — data is lost when the connection closes).

    Returns
    -------
    (connection, path):
        A tuple of the open ``sqlite3.Connection`` and the path to the
        database file.
    """
    if db_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = tmp.name
        tmp.close()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = OFF")

    for stmt in schema_statements:
        stmt = stmt.strip()
        if not stmt:
            continue
        try:
            conn.executescript(stmt)
        except sqlite3.Error as e:
            print(f"Warning: could not execute schema statement:\n"
                  f"  {stmt[:200]}\n  Error: {e}")

    conn.commit()
    return conn, db_path


def extract_schema(context: str) -> list[str]:
    """Extract individual CREATE TABLE statements from a context string.

    The dataset's ``context`` field often contains one or more
    ``CREATE TABLE ...;`` statements.  This splits on semicolons and
    returns only statements that start with ``CREATE``.
    """
    statements = []
    for part in context.split(";"):
        part = part.strip()
        if part.upper().startswith("CREATE"):
            statements.append(part + ";")
    return statements


def table_names(conn: sqlite3.Connection) -> list[str]:
    """Return names of user tables in the database."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return [r[0] for r in rows]


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Materialize schemas")
    parser.add_argument("--db", type=str, default=None,
                        help="Path for the output SQLite db")
    parser.add_argument("--drop-if-exists", action="store_true",
                        help="Remove existing db file before creating")
    parser.add_argument("dataset_path", nargs="?",
                        default="data/eval.jsonl",
                        help="Path to a JSONL dataset file")
    args = parser.parse_args()

    # Load schemas from the dataset file (eval set)
    schemas: set[str] = set()
    with open(args.dataset_path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            schemas.add(ex["context"])

    stmts: list[str] = []
    for ctx in schemas:
        stmts.extend(extract_schema(ctx))

    db_path = args.db
    if args.drop_if_exists and db_path and db_path != ":memory:":
        import os as _os
        if _os.path.exists(db_path):
            _os.remove(db_path)

    conn, path = create_database_from_schemas(stmts, db_path)
    tables = table_names(conn)
    conn.close()

    print(f"Database created at {path}")
    print(f"Tables ({len(tables)}): {', '.join(tables)}")
