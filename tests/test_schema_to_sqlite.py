"""Unit tests for data.schema_to_sqlite."""

import sqlite3
from data.schema_to_sqlite import (
    create_database_from_schemas,
    extract_schema,
    table_names,
)


SAMPLE_CONTEXT = (
    "CREATE TABLE head (age INTEGER, name TEXT, department_id INTEGER);"
)
MULTI_TABLE_CONTEXT = (
    "CREATE TABLE player (player_id INTEGER, name TEXT); "
    "CREATE TABLE coach (coach_id INTEGER, name TEXT); "
    "CREATE TABLE player_coach (player_id INTEGER, coach_id INTEGER);"
)


def test_extract_schema_single():
    stmts = extract_schema(SAMPLE_CONTEXT)
    assert len(stmts) == 1
    assert stmts[0].upper().startswith("CREATE TABLE")


def test_extract_schema_multi():
    stmts = extract_schema(MULTI_TABLE_CONTEXT)
    assert len(stmts) == 3
    for s in stmts:
        assert s.upper().startswith("CREATE TABLE")


def test_create_database_in_memory():
    stmts = extract_schema(SAMPLE_CONTEXT)
    conn, path = create_database_from_schemas(stmts, ":memory:")
    assert path == ":memory:"
    tables = table_names(conn)
    assert tables == ["head"]
    conn.close()


def test_table_is_empty():
    stmts = extract_schema(SAMPLE_CONTEXT)
    conn, _ = create_database_from_schemas(stmts, ":memory:")
    row = conn.execute("SELECT COUNT(*) FROM head").fetchone()
    assert row[0] == 0
    conn.close()


def test_select_executes():
    """A basic SELECT against the materialized schema runs without error."""
    stmts = extract_schema(SAMPLE_CONTEXT)
    conn, _ = create_database_from_schemas(stmts, ":memory:")
    cur = conn.execute("SELECT name FROM head WHERE age > 30")
    assert cur.fetchall() == []
    conn.close()


def test_multi_table_join_executes():
    stmts = extract_schema(MULTI_TABLE_CONTEXT)
    conn, _ = create_database_from_schemas(stmts, ":memory:")
    sql = """
        SELECT p.name, c.name
        FROM player p
        JOIN player_coach pc ON p.player_id = pc.player_id
        JOIN coach c ON pc.coach_id = c.coach_id
    """
    cur = conn.execute(sql)
    assert cur.fetchall() == []
    conn.close()


def test_nested_select_executes():
    """Subqueries work against empty materialized tables."""
    stmts = extract_schema(SAMPLE_CONTEXT)
    conn, _ = create_database_from_schemas(stmts, ":memory:")
    sql = "SELECT name FROM head WHERE department_id IN (SELECT department_id FROM head)"
    cur = conn.execute(sql)
    assert cur.fetchall() == []
    conn.close()


def test_invalid_schema_does_not_crash():
    """Known-bad SQL should log a warning but not raise."""
    stmts = ["CREATE TABLE bad ();"]  # malformed — no columns
    conn, _ = create_database_from_schemas(stmts, ":memory:")
    conn.close()
    # We just verify no exception propagates to the caller
