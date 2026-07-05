"""
SQL-LoRA — Fine-Tuned Evaluation

Loads the base model + QLoRA adapter from the Hugging Face Hub and
evaluates execution-accuracy on the test set. Results saved to
results/finetuned_results.json.

Usage:
    python evaluation/run_finetuned_eval.py [--num_samples N] [--adapter REPO_ID]
"""


def load_finetuned_model(base_model_id: str, adapter_repo: str):
    """Load base model in 4-bit and merge/attach LoRA adapter weights."""
    ...


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--adapter", type=str, default="")
    args = parser.parse_args()

    print(f"Fine-tuned evaluation on {args.num_samples} samples.")
