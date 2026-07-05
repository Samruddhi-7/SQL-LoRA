"""
SQL-LoRA — Zero-Shot Baseline Evaluation

Evaluates the untuned base model's text-to-SQL performance on the
held-out eval set using **execution-accuracy** (Phase 2).

Inference backend: Groq API (llama-3.1-8b-instant).
The exact model used for fine-tuning (unsloth/Llama-3.2-3B-Instruct-bnb-4bit)
was previously available on Groq as llama-3.2-3b-preview but has been
decommissioned.  llama-3.1-8b-instant is the closest Llama instruct model
currently available on Groq's free tier.  This is a *stronger* baseline
(8B > 3B parameters), so any improvement from fine-tuning is more
meaningful — the fine-tuned 3B model must beat a larger model.

Usage:
    export GROQ_API_KEY="gsk_..."
    python -m evaluation.run_baseline_eval [--num_samples N]

Results are saved to results/baseline_results.json.
"""

import json
import os
import re
import sys
import time

# Ensure project root is on sys.path (needed before any local imports)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from evaluation.execution_accuracy import evaluate_example


BASELINE_MODEL = "llama-3.1-8b-instant"
EVAL_PATH = os.path.join("data", "eval.jsonl")
RESULTS_PATH = os.path.join("results", "baseline_results.json")

SYSTEM_PROMPT = (
    "You are a SQL expert. Given a database schema and a natural language "
    "question, generate a single SQL query that answers the question. "
    "Output only the SQL query — no explanation, no markdown formatting."
)


def _build_messages(context: str, question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Schema:\n{context}\n\nQuestion: {question}\nSQL:",
        },
    ]


def _extract_sql(text: str) -> str:
    """Extract SQL from the model response, stripping markdown fences."""
    text = text.strip()
    # Remove markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    # Take only the first line or first statement (split on blank lines)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return ""
    return lines[0]


def load_eval_set(num_samples: int | None = None) -> list[dict]:
    """Load held-out evaluation examples from JSONL."""
    examples = []
    with open(EVAL_PATH, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    if num_samples is not None:
        examples = examples[:num_samples]
    return examples


def generate_sql_groq(client, context: str, question: str) -> str:
    """Call Groq API to generate SQL for a schema + question pair."""
    messages = _build_messages(context, question)
    try:
        completion = client.chat.completions.create(
            model=BASELINE_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=256,
            top_p=0.95,
        )
        raw = completion.choices[0].message.content or ""
        return _extract_sql(raw)
    except Exception as e:
        print(f"  API error: {e}", file=sys.stderr)
        return ""


def run_baseline_eval(num_samples: int | None = None) -> dict:
    """Run the full baseline evaluation and return aggregate results."""
    # ── Load eval set ────────────────────────────────────────────────
    examples = load_eval_set(num_samples)
    print(f"Loaded {len(examples)} evaluation examples")

    # ── Init Groq client ─────────────────────────────────────────────
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    from groq import Groq

    client = Groq(api_key=api_key)

    # ── Evaluate each example ────────────────────────────────────────
    per_example_results: list[dict] = []
    n_executable = 0
    n_correct = 0

    for i, ex in enumerate(examples):
        print(f"[{i + 1}/{len(examples)}] {ex['question'][:70]}...", end=" ")

        # Generate SQL
        generated = generate_sql_groq(client, ex["context"], ex["question"])
        if not generated:
            print("  [EMPTY]")
            per_example_results.append({
                "executable": False,
                "correct": False,
                "generated_sql": "",
                "gold_sql": ex["answer"],
                "error_message": "Empty response from model",
                "question": ex["question"],
                "context": ex["context"],
            })
            continue

        print(f"  SQL hint: {generated[:60]}...", end=" ")

        # Evaluate via execution accuracy
        eval_result = evaluate_example(
            schema_context=ex["context"],
            gold_sql=ex["answer"],
            pred_sql=generated,
        )
        eval_result["question"] = ex["question"]
        eval_result["context"] = ex["context"]

        per_example_results.append(eval_result)
        if eval_result["executable"]:
            n_executable += 1
        if eval_result["correct"]:
            n_correct += 1

        if eval_result["correct"]:
            status = "CORRECT"
        elif eval_result["executable"]:
            status = "EXECUTABLE"
        else:
            status = "FAILED"
        print(f" [{status}]")

        # Rate limit: Groq free tier allows ~30 RPM
        time.sleep(0.7)

    # ── Aggregate ────────────────────────────────────────────────────
    total = len(per_example_results)
    aggregate = {
        "model": BASELINE_MODEL,
        "total_examples": total,
        "executable": n_executable,
        "correct": n_correct,
        "execution_accuracy": round(n_correct / total, 4) if total else 0.0,
        "executable_rate": round(n_executable / total, 4) if total else 0.0,
    }
    return {"aggregate": aggregate, "per_example": per_example_results}


def print_summary(aggregate: dict) -> None:
    print("\n" + "=" * 55)
    print("  ZERO-SHOT BASELINE EVALUATION SUMMARY")
    print("=" * 55)
    print(f"  Model:               {aggregate['model']}")
    print(f"  Total examples:      {aggregate['total_examples']}")
    print(f"  Executable:          {aggregate['executable']} "
          f"({aggregate['executable_rate']:.1%})")
    print(f"  Correct:             {aggregate['correct']} "
          f"({aggregate['execution_accuracy']:.1%})")
    print("=" * 55)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run zero-shot baseline text-to-SQL evaluation"
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Number of eval examples to use (default: all 150)"
    )
    args = parser.parse_args()

    results = run_baseline_eval(num_samples=args.num_samples)
    print_summary(results["aggregate"])

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")
