"""
SQL-LoRA — Execution-Accuracy Evaluation

Core evaluation harness: given a schema string, a gold SQL query, and a
predicted SQL query:

1. Materialize the schema in a fresh in-memory SQLite database
2. Populate tables with synthetic data (so empty-table false positives
   are minimized during result-set comparison)
3. Execute both queries and compare their result sets

A query is **executable** if it runs without raising a SQL error.
A query is **correct** if it is executable AND its result set matches
the gold query's result set (order-independent unless gold has ORDER BY).

Known blind spots of result-set comparison
-------------------------------------------
- Two differently-wrong queries can both return empty result sets and be
  counted as "correct" (a false positive).  Synthetic data reduces but
  does not eliminate this risk.
- ``SELECT a, b`` vs ``SELECT b, a`` will not match because column order
  is part of the result-set signature.  This is a deliberate choice:
  column ordering is part of correct SQL semantics.
- ``SELECT DISTINCT x`` vs ``SELECT x`` produce identical results when
  all x values are already distinct.  The synthetic data generator keeps
  values distinct per row to minimise this, but it is not guaranteed.
- Syntactically different but semantically equivalent queries (e.g. an
  inner join expressed as a subquery) will match if they produce the
  same result set on the synthetic data.  This is a feature, not a bug.
"""

import re
import sqlite3
from collections import defaultdict

from data.schema_to_sqlite import (
    create_database_from_schemas,
    extract_schema,
    table_names,
)

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

ROWS_PER_TABLE = 5

_SQL_TYPE_PATTERN = re.compile(
    r"^\s*(?P<name>\w+)\s+(?P<type>\w+(?:\s*\([^)]+\))?)",
    re.IGNORECASE,
)


def _parse_create_table(create_stmt: str) -> tuple[str, list[tuple[str, str]]]:
    """Parse a single CREATE TABLE statement.

    Returns (table_name, [(column_name, column_type), ...]).
    """
    name_match = re.search(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
        r"(?:\"([^\"]+)\"|(\w+))\s*\(",
        create_stmt,
        re.IGNORECASE | re.DOTALL,
    )
    if not name_match:
        return "", []
    table_name = name_match.group(1) or name_match.group(2)

    # Find the column definitions between the outer parentheses
    paren_depth = 0
    start_idx = None
    columns_text = ""
    for i, ch in enumerate(create_stmt):
        if ch == '(':
            if paren_depth == 0:
                start_idx = i + 1
            paren_depth += 1
        elif ch == ')':
            paren_depth -= 1
            if paren_depth == 0 and start_idx is not None:
                columns_text = create_stmt[start_idx:i]
                break

    columns: list[tuple[str, str]] = []
    for line in columns_text.split(","):
        line = line.strip()
        if not line:
            continue
        # Skip constraints (PRIMARY KEY, FOREIGN KEY, UNIQUE, CHECK, etc.)
        if line.upper().startswith(("PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "INDEX", "CONSTRAINT")):
            continue
        m = _SQL_TYPE_PATTERN.match(line)
        if m:
            col_name = m.group("name")
            col_type = m.group("type").upper().split("(")[0].strip()
            columns.append((col_name, col_type))

    return table_name, columns


def _is_id_column(name: str) -> bool:
    """Heuristic: return True if column looks like a primary/foreign key ID."""
    return name.lower() in ("id",) or name.lower().endswith("_id")


def _synthetic_value(
    col_name: str, col_type: str, row_idx: int, col_idx: int
) -> int | float | str:
    """Generate a synthetic value for a column.

    Design
    ------
    - ID columns (``id``, ``*_id``) receive unique sequential values so
      that JOINs on key columns produce meaningful matches.
    - Non-ID columns receive *repeated* values (at most 3 distinct per
      column) with a column-specific offset *col_idx*.  This ensures
      that ``GROUP BY col_a`` and ``GROUP BY col_b`` produce different
      result sets, letting the comparison catch wrong GROUP BY clauses.
    - TEXT values include the column name so that referencing the wrong
      column is detectable.
    """
    base_type = col_type.upper().split("(")[0].strip()
    if _is_id_column(col_name):
        val = row_idx + 1
        return val if "INT" in base_type else str(val)

    repeat_idx = row_idx % 4  # yields 0, 1, 2, 3, 0 for 5 rows — 4 distinct values
    if base_type in ("INTEGER", "INT", "BIGINT", "SMALLINT", "TINYINT", "BOOLEAN"):
        pool = [5, 15, 25, 35]
        return pool[repeat_idx] + col_idx
    elif base_type in ("REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL"):
        pool = [5.0, 15.0, 25.0, 35.0]
        return pool[repeat_idx] + float(col_idx)
    else:
        return f"{col_name}_{repeat_idx + 1}"


def populate_synthetic_data(conn: sqlite3.Connection, schema_statements: list[str]) -> None:
    """Insert a small number of synthetic rows into every table.

    Each column receives distinct, predictable values per row so that
    different JOIN / WHERE / GROUP BY clauses produce meaningfully
    different result sets during evaluation.
    """
    for stmt in schema_statements:
        stmt = stmt.strip()
        if not stmt:
            continue
        table_name, columns = _parse_create_table(stmt)
        if not table_name or not columns:
            continue
        try:
            _populate_table(conn, table_name, columns)
        except sqlite3.Error:
            pass  # table might not exist (schema conflict on same name)
    conn.commit()


def _populate_table(
    conn: sqlite3.Connection,
    table_name: str,
    columns: list[tuple[str, str]],
) -> None:
    col_names = [c[0] for c in columns]
    placeholders = ", ".join(["?"] * len(col_names))
    sql = f"INSERT INTO \"{table_name}\" ({', '.join(f'\"{n}\"' for n in col_names)}) VALUES ({placeholders})"

    for row_idx in range(ROWS_PER_TABLE):
        values = [
            _synthetic_value(col_name, col_type, row_idx, col_idx)
            for col_idx, (col_name, col_type) in enumerate(columns)
        ]
        conn.execute(sql, values)


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

def execute_query(sql: str, conn: sqlite3.Connection) -> list[tuple] | None:
    """Execute a SQL string against a connection.

    Returns a list of result tuples, or ``None`` if execution fails.
    """
    try:
        cursor = conn.execute(sql)
        return cursor.fetchall()
    except sqlite3.Error:
        return None


def has_order_by(sql: str) -> bool:
    """Check if a SQL query contains an ORDER BY clause (outside subqueries)."""
    # Remove parenthesized subqueries before checking
    cleaned = re.sub(r"\([^()]*\)", "", sql, flags=re.IGNORECASE)
    return bool(re.search(r"\bORDER\s+BY\b", cleaned, re.IGNORECASE))


# ---------------------------------------------------------------------------
# Result-set comparison
# ---------------------------------------------------------------------------

def compare_result_sets(
    gold_rows: list[tuple] | None,
    pred_rows: list[tuple] | None,
    ordered: bool = False,
) -> bool:
    """Compare two lists of result rows for exact match.

    Parameters
    ----------
    gold_rows:
        Result from the gold query, or ``None`` if it failed to execute.
    pred_rows:
        Result from the predicted query, or ``None``.
    ordered:
        If ``True`` (gold SQL has ORDER BY), compare as ordered lists.
        Otherwise compare as unordered sets (order-independent).

    Returns ``True`` iff the results match.
    """
    if gold_rows is None or pred_rows is None:
        return False

    if ordered:
        return gold_rows == pred_rows
    else:
        return set(gold_rows) == set(pred_rows)


# ---------------------------------------------------------------------------
# Per-example evaluation
# ---------------------------------------------------------------------------

EvalResult = dict[str, str | bool | None]


def evaluate_example(
    schema_context: str,
    gold_sql: str,
    pred_sql: str,
) -> EvalResult:
    """Evaluate a single (schema, gold SQL, predicted SQL) triple.

    Returns a dict with keys:
        - executable: bool — whether pred_sql ran without error
        - correct: bool — whether pred_sql result matches gold_sql result
        - generated_sql: str — the predicted SQL
        - gold_sql: str — the ground-truth SQL
        - error_message: str | None — SQL error if not executable
    """
    stmts = extract_schema(schema_context)
    conn, _ = create_database_from_schemas(stmts, ":memory:")
    populate_synthetic_data(conn, stmts)

    gold_rows = execute_query(gold_sql, conn)
    if gold_rows is None:
        # Gold SQL itself failed — this should not happen with a clean dataset
        gold_rows_set = ()
        ordered = False
    else:
        ordered = has_order_by(gold_sql)

    pred_rows = execute_query(pred_sql, conn)
    error_message: str | None = None
    executable = pred_rows is not None

    if not executable:
        try:
            conn.execute(pred_sql)
        except sqlite3.Error as e:
            error_message = str(e)

    correct = compare_result_sets(gold_rows, pred_rows, ordered=ordered)
    conn.close()

    return {
        "executable": executable,
        "correct": correct,
        "generated_sql": pred_sql,
        "gold_sql": gold_sql,
        "error_message": error_message,
    }


def evaluate_examples(
    examples: list[dict[str, str]],
) -> list[EvalResult]:
    """Evaluate a list of examples.

    Each example dict must have keys ``context`` (schema), ``answer`` (gold SQL),
    and ``generated`` (predicted SQL).
    """
    results = []
    for ex in examples:
        result = evaluate_example(
            schema_context=ex["context"],
            gold_sql=ex["answer"],
            pred_sql=ex["generated"],
        )
        results.append(result)
    return results
