"""
SQL-LoRA — Interactive Demo CLI

Loads the fine-tuned model and lets the user input a schema + question
from the command line, then prints the generated SQL and its result.

Usage:
    python inference/demo.py [--adapter REPO_ID] [--base_model MODEL_ID]
"""


def format_prompt(schema: str, question: str) -> str:
    """Format a schema + question into the model's chat template."""
    ...


def run_demo(model, tokenizer):
    """Interactive REPL loop for querying the model."""
    ...


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default="")
    parser.add_argument("--base_model", type=str, default="")
    args = parser.parse_args()

    print("SQL-LoRA Interactive Demo")
    print("Type 'exit' to quit.")
