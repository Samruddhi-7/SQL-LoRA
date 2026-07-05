"""Unit tests for evaluation.execution_accuracy.

This is the most important module in the project — tests cover:
- Identical queries
- Semantically equivalent queries (same results, different syntax)
- Column-order sensitivity (deliberate design choice)
- Genuinely wrong queries
- Invalid SQL (non-executable)
- ORDER BY preservation
- JOIN, GROUP BY, subqueries
- Synthetic data correctness
"""

import sqlite3

from evaluation.execution_accuracy import (
    _parse_create_table,
    _synthetic_value,
    populate_synthetic_data,
    execute_query,
    compare_result_sets,
    has_order_by,
    evaluate_example,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIMPLE_SCHEMA = "CREATE TABLE users (id INTEGER, name TEXT, age INTEGER);"


def _make_conn(schema: str = SIMPLE_SCHEMA) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(schema)
    conn.commit()
    return conn


# ===================================================================
# _parse_create_table
# ===================================================================

class TestParseCreateTable:
    def test_simple(self):
        name, cols = _parse_create_table(SIMPLE_SCHEMA)
        assert name == "users"
        assert cols == [("id", "INTEGER"), ("name", "TEXT"), ("age", "INTEGER")]

    def test_with_constraints(self):
        sql = "CREATE TABLE t (x INTEGER PRIMARY KEY, y TEXT NOT NULL, z REAL DEFAULT 0.0);"
        name, cols = _parse_create_table(sql)
        assert name == "t"
        assert cols == [("x", "INTEGER"), ("y", "TEXT"), ("z", "REAL")]

    def test_with_foreign_key(self):
        sql = (
            "CREATE TABLE child (id INTEGER, parent_id INTEGER, "
            "FOREIGN KEY (parent_id) REFERENCES parent(id));"
        )
        name, cols = _parse_create_table(sql)
        assert name == "child"
        assert cols == [("id", "INTEGER"), ("parent_id", "INTEGER")]

    def test_if_not_exists(self):
        sql = "CREATE TABLE IF NOT EXISTS t (a TEXT, b INTEGER);"
        name, cols = _parse_create_table(sql)
        assert name == "t"
        assert cols == [("a", "TEXT"), ("b", "INTEGER")]

    def test_quoted_table_name(self):
        sql = 'CREATE TABLE "my table" (x INTEGER);'
        name, cols = _parse_create_table(sql)
        assert name == "my table"
        assert cols == [("x", "INTEGER")]

    def test_multiple_tables(self):
        sql = "CREATE TABLE a (x INT); CREATE TABLE b (y TEXT);"
        name1, cols1 = _parse_create_table(sql[:sql.index(";") + 1])
        name2, cols2 = _parse_create_table(sql[sql.index(";") + 1:])
        assert name1 == "a" and cols1 == [("x", "INT")]
        assert name2 == "b" and cols2 == [("y", "TEXT")]


# ===================================================================
# _synthetic_value
# ===================================================================

class TestSyntheticValue:
    def test_integer_id(self):
        # ID columns get sequential values
        assert _synthetic_value("id", "INTEGER", 0, col_idx=0) == 1
        assert _synthetic_value("id", "INTEGER", 4, col_idx=0) == 5

    def test_integer_non_id(self):
        # Non-ID integers come from the rotating pool + col_idx
        val = _synthetic_value("age", "INTEGER", 0, col_idx=2)
        assert val == 5 + 2  # pool[0] + col_idx

    def test_real(self):
        val = _synthetic_value("score", "REAL", 0, col_idx=1)
        assert val == 5.0 + 1.0

    def test_text(self):
        val = _synthetic_value("name", "TEXT", 0, col_idx=1)
        assert val == "name_1"
        val2 = _synthetic_value("email", "VARCHAR(100)", 2, col_idx=2)
        assert val2 == "email_3"

    def test_boolean(self):
        assert _synthetic_value("flag", "BOOLEAN", 0, col_idx=2) == 5 + 2


# ===================================================================
# populate_synthetic_data
# ===================================================================

class TestPopulateSyntheticData:
    def test_rows_inserted(self):
        conn = _make_conn()
        populate_synthetic_data(conn, [SIMPLE_SCHEMA + ";"])
        rows = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        assert rows == 5

    def test_values_are_distinct(self):
        conn = _make_conn()
        populate_synthetic_data(conn, [SIMPLE_SCHEMA + ";"])
        ids = [r[0] for r in conn.execute("SELECT id FROM users").fetchall()]
        assert ids == [1, 2, 3, 4, 5]

    def test_multiple_tables(self):
        schema = (
            "CREATE TABLE a (x INTEGER); CREATE TABLE b (y TEXT);"
        )
        conn = _make_conn(schema)
        populate_synthetic_data(conn, ["CREATE TABLE a (x INTEGER);", "CREATE TABLE b (y TEXT);"])
        assert conn.execute("SELECT COUNT(*) FROM a").fetchone()[0] == 5
        assert conn.execute("SELECT COUNT(*) FROM b").fetchone()[0] == 5


# ===================================================================
# execute_query
# ===================================================================

class TestExecuteQuery:
    def test_valid_sql(self):
        conn = _make_conn()
        conn.execute("INSERT INTO users VALUES (1, 'alice', 30)")
        conn.commit()
        rows = execute_query("SELECT name FROM users", conn)
        assert rows == [("alice",)]

    def test_invalid_sql_returns_none(self):
        conn = _make_conn()
        rows = execute_query("SELECT x FROM nonexistent", conn)
        assert rows is None


# ===================================================================
# compare_result_sets
# ===================================================================

class TestCompareResultSets:
    def test_identical_unordered(self):
        assert compare_result_sets([(1,), (2,)], [(2,), (1,)], ordered=False) is True

    def test_identical_ordered(self):
        assert compare_result_sets([(1,), (2,)], [(1,), (2,)], ordered=True) is True

    def test_order_matters_when_ordered(self):
        assert compare_result_sets([(1,), (2,)], [(2,), (1,)], ordered=True) is False

    def test_different_sets(self):
        assert compare_result_sets([(1,)], [(2,)], ordered=False) is False

    def test_none_gold(self):
        assert compare_result_sets(None, [(1,)], ordered=False) is False

    def test_none_pred(self):
        assert compare_result_sets([(1,)], None, ordered=False) is False

    def test_both_none(self):
        assert compare_result_sets(None, None, ordered=False) is False


# ===================================================================
# has_order_by
# ===================================================================

class TestHasOrderBy:
    def test_with_order_by(self):
        assert has_order_by("SELECT x FROM t ORDER BY x") is True

    def test_without_order_by(self):
        assert has_order_by("SELECT x FROM t") is False

    def test_order_by_in_subquery(self):
        """ORDER BY inside a subquery should NOT affect the outer comparison."""
        sql = "SELECT x FROM (SELECT x FROM t ORDER BY x) AS sub"
        assert has_order_by(sql) is False


# ===================================================================
# evaluate_example — comprehensive
# ===================================================================

SCHEMA_USERS = "CREATE TABLE users (id INTEGER, name TEXT, age INTEGER);"
SCHEMA_MULTI = (
    "CREATE TABLE users (id INTEGER, name TEXT, dept_id INTEGER);"
    "CREATE TABLE depts (id INTEGER, name TEXT);"
)


class TestEvaluateExample:
    """Every test calls evaluate_example and checks result fields."""

    # ── Correct queries ───────────────────────────────────────────

    def test_exact_match(self):
        """Identical gold and pred → correct and executable."""
        r = evaluate_example(SCHEMA_USERS, "SELECT name FROM users", "SELECT name FROM users")
        assert r["executable"] is True
        assert r["correct"] is True

    def test_case_insensitive(self):
        """SQL keywords case should not matter."""
        r = evaluate_example(SCHEMA_USERS, "select name from users", "SELECT name FROM users")
        assert r["correct"] is True

    def test_semantic_equivalent_join(self):
        """An inner join written with JOIN vs implicit WHERE should match
        if they produce identical result sets on the synthetic data."""
        gold = (
            "SELECT u.name, d.name "
            "FROM users u JOIN depts d ON u.dept_id = d.id"
        )
        pred = (
            "SELECT u.name, d.name "
            "FROM users u, depts d WHERE u.dept_id = d.id"
        )
        r = evaluate_example(SCHEMA_MULTI, gold, pred)
        assert r["executable"] is True
        assert r["correct"] is True, "Semantically equivalent join styles should match"

    def test_where_equivalent(self):
        """Different but equivalent WHERE filters."""
        gold = "SELECT name FROM users WHERE age >= 30"
        pred = "SELECT name FROM users WHERE age > 29"
        r = evaluate_example(SCHEMA_USERS, gold, pred)
        assert r["correct"] is True

    # ── Known: column order sensitivity ───────────────────────────

    def test_column_order_mismatch(self):
        """Different column order in SELECT → not correct.
        This is a known deliberate limitation: column ordering is part
        of the SQL result signature."""
        gold = "SELECT name, age FROM users"
        pred = "SELECT age, name FROM users"
        r = evaluate_example(SCHEMA_USERS, gold, pred)
        assert r["correct"] is False, (
            "Column order differs — result tuples differ by design"
        )

    # ── Wrong queries ─────────────────────────────────────────────

    def test_wrong_table(self):
        """Referencing a non-existent table → not executable."""
        r = evaluate_example(SCHEMA_USERS, "SELECT name FROM users",
                             "SELECT name FROM employees")
        assert r["executable"] is False
        assert r["correct"] is False

    def test_wrong_column(self):
        """Referencing a non-existent column → not executable."""
        r = evaluate_example(SCHEMA_USERS, "SELECT name FROM users",
                             "SELECT email FROM users")
        assert r["executable"] is False
        assert r["correct"] is False

    def test_wrong_where(self):
        """WHERE clause that filters differently should produce a
        different result set."""
        r = evaluate_example(SCHEMA_USERS, "SELECT name FROM users WHERE age > 3",
                             "SELECT name FROM users WHERE age > 10")
        assert r["correct"] is False

    def test_missing_join(self):
        """Missing a JOIN entirely changes the result."""
        gold = (
            "SELECT u.name, d.name FROM users u "
            "JOIN depts d ON u.dept_id = d.id"
        )
        pred = "SELECT name, 'sales' FROM users"  # wrong: no join, hardcoded
        r = evaluate_example(SCHEMA_MULTI, gold, pred)
        assert r["correct"] is False

    # ── Aggregation / GROUP BY ────────────────────────────────────

    def test_aggregation_wrong_group_by(self):
        """Wrong GROUP BY column produces different results."""
        gold = "SELECT age, COUNT(*) FROM users GROUP BY age"
        # Pred selects the column it groups by — result tuples differ in type
        pred = "SELECT name, COUNT(*) FROM users GROUP BY name"
        r = evaluate_example(SCHEMA_USERS, gold, pred)
        assert r["correct"] is False

    def test_aggregation_correct(self):
        """Correct GROUP BY should match."""
        r = evaluate_example(SCHEMA_USERS,
                             "SELECT age, COUNT(*) FROM users GROUP BY age",
                             "SELECT age, COUNT(*) FROM users GROUP BY age")
        assert r["correct"] is True

    # ── ORDER BY ──────────────────────────────────────────────────

    def test_order_by_preserved(self):
        """When gold has ORDER BY, comparison is order-sensitive."""
        gold = "SELECT name FROM users ORDER BY name"
        pred_same_order = "SELECT name FROM users ORDER BY name"
        pred_diff_order = "SELECT name FROM users ORDER BY name DESC"
        r1 = evaluate_example(SCHEMA_USERS, gold, pred_same_order)
        r2 = evaluate_example(SCHEMA_USERS, gold, pred_diff_order)
        assert r1["correct"] is True
        assert r2["correct"] is False

    def test_no_order_by_unordered_comparison(self):
        """Without ORDER BY, comparison is order-independent."""
        gold = "SELECT name FROM users"
        pred = "SELECT name FROM users"
        r = evaluate_example(SCHEMA_USERS, gold, pred)
        assert r["correct"] is True

    # ── Subqueries ────────────────────────────────────────────────

    def test_subquery_correct(self):
        """Correctly written subquery should match."""
        gold = "SELECT name FROM users WHERE id IN (SELECT id FROM users WHERE age > 2)"
        pred = "SELECT name FROM users WHERE id IN (SELECT id FROM users WHERE age > 2)"
        r = evaluate_example(SCHEMA_USERS, gold, pred)
        assert r["correct"] is True

    def test_subquery_wrong(self):
        """Wrong subquery predicate → different results."""
        gold = "SELECT name FROM users WHERE id IN (SELECT id FROM users WHERE age > 2)"
        pred = "SELECT name FROM users WHERE id IN (SELECT id FROM users WHERE age > 10)"
        r = evaluate_example(SCHEMA_USERS, gold, pred)
        assert r["correct"] is False

    # ── Empty results ─────────────────────────────────────────────

    def test_both_empty_but_different_reason(self):
        """Both queries return empty sets for different reasons.
        This is a known false-positive risk — documented in the module
        docstring."""
        gold = "SELECT name FROM users WHERE age > 100"  # no rows match
        pred = "SELECT name FROM users WHERE name = 'nonexistent'"  # no rows match
        r = evaluate_example(SCHEMA_USERS, gold, pred)
        # Both return empty [] == [] → True (false positive)
        assert r["correct"] is True
        assert r["executable"] is True

    # ── Invalid SQL ──────────────────────────────────────────────

    def test_invalid_syntax(self):
        """Malformed SQL → not executable, not correct."""
        r = evaluate_example(SCHEMA_USERS, "SELECT name FROM users",
                             "SELECT name FORM users")
        assert r["executable"] is False
        assert r["correct"] is False
        assert r["error_message"] is not None

    def test_nonexistent_function(self):
        """Unknown function → not executable."""
        r = evaluate_example(SCHEMA_USERS, "SELECT name FROM users",
                             "SELECT FAKE_FUNC(name) FROM users")
        assert r["executable"] is False

    # ── Error message ─────────────────────────────────────────────

    def test_error_message_on_failure(self):
        """Error message should be populated when SQL fails."""
        r = evaluate_example(SCHEMA_USERS, "SELECT name FROM users",
                             "SELECT name FROM nonexistent")
        assert r["error_message"] is not None
        assert "nonexistent" in r["error_message"].lower()
