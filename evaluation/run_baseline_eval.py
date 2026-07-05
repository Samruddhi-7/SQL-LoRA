"""
SQL-LoRA — Baseline Evaluation

Evaluates the untuned (base) model's zero-shot text-to-SQL performance
on the test set. Results are saved to results/baseline_results.json.

Usage:
    python evaluation/run_baseline_eval.py [--num_samples N]
"""


def load_base_model(model_id: str):
    """Load the base model in 4-bit for inference (no LoRA adapters)."""
    ...


def generate_sql(model, tokenizer, prompt: str) -> str:
    """Generate a SQL query from a natural language question + schema."""
    ...


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    print(f"Baseline evaluation on {args.num_samples} samples.")
