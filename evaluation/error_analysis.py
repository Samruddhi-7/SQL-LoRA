"""
SQL-LoRA — Error Analysis

Categorizes failed SQL generations by error type to understand where
the model struggles most.

Error categories:
  - Syntax error (SQL fails to parse/execute)
  - Schema error (references nonexistent table/column)
  - Join error (wrong join condition or missing join)
  - Aggregation error (wrong GROUP BY, missing aggregation function)
  - Nested query error (subquery issues)
  - Logic error (executes but returns wrong results)
  - Other

Usage:
    python evaluation/error_analysis.py --results results/finetuned_results.json
"""

ERROR_CATEGORIES = {
    "syntax_error": "SQL fails to parse or execute",
    "schema_error": "References nonexistent table or column",
    "join_error": "Wrong join condition or missing join",
    "aggregation_error": "Wrong GROUP BY or missing aggregation",
    "nested_query_error": "Subquery issues",
    "logic_error": "Executes but returns wrong results",
    "other": "Does not fit above categories",
}


def categorize_error(
    gold_sql: str, pred_sql: str, error_msg: str | None
) -> str:
    """Return the error category for a (gold, pred) mismatch."""
    ...


def generate_report(results: list[dict]) -> dict:
    """Aggregate error categories into a report dict with counts and
    percentages, plus qualitative examples for each category."""
    ...


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    report = generate_report(data)
    print(json.dumps(report, indent=2))
