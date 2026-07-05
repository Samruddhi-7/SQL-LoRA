"""
SQL-LoRA — Fine-Tuned Model Evaluation

Evaluates the QLoRA fine-tuned model's text-to-SQL performance on the
held-out eval set using **execution-accuracy** (Phase 2).

The base model is loaded in 4-bit and the LoRA adapter is merged on top.
All inference happens locally (no API calls).

Usage:
    # From HF Hub (recommended)
    python -m evaluation.run_finetuned_eval --adapter your-username/sql-lora-llama3.2-3b

    # From local path
    python -m evaluation.run_finetuned_eval --adapter ./sql_lora_adapter

    # Subset
    python -m evaluation.run_finetuned_eval --adapter <name> --num_samples 10

Results are saved to results/finetuned_results.json.
"""

import json
import os
import re
import sys
import time

# Ensure project root is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from evaluation.execution_accuracy import evaluate_example


EVAL_PATH = os.path.join("data", "eval.jsonl")
RESULTS_PATH = os.path.join("results", "finetuned_results.json")


def _extract_sql(text: str) -> str:
    """Extract SQL from the model response, stripping markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
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


def build_prompt(context: str, question: str) -> str:
    """Build the same prompt format used during training."""
    return (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{context}\n\nQuestion: {question}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def load_model_and_tokenizer(adapter_path: str):
    """Load base model in 4-bit and attach LoRA adapter."""
    from peft import PeftModel
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct",
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def run_finetuned_eval(
    adapter_path: str,
    num_samples: int | None = None,
) -> dict:
    """Run the full fine-tuned evaluation and return aggregate results."""
    # ── Load eval set ────────────────────────────────────────────────
    examples = load_eval_set(num_samples)
    print(f"Loaded {len(examples)} evaluation examples")

    # ── Load model ───────────────────────────────────────────────────
    print(f"Loading model with adapter: {adapter_path}")
    model, tokenizer = load_model_and_tokenizer(adapter_path)
    print("Model loaded successfully")

    # ── Evaluate each example ────────────────────────────────────────
    per_example_results: list[dict] = []
    n_executable = 0
    n_correct = 0

    for i, ex in enumerate(examples):
        print(f"[{i + 1}/{len(examples)}] {ex['question'][:70]}...", end=" ")

        # Build prompt and generate
        prompt = build_prompt(ex["context"], ex["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.95,
            use_cache=True,
        )
        raw = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()
        generated = _extract_sql(raw)

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

        print(f"  SQL: {generated[:60]}...", end=" ")

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

    # ── Aggregate ────────────────────────────────────────────────────
    total = len(per_example_results)
    aggregate = {
        "model": adapter_path,
        "total_examples": total,
        "executable": n_executable,
        "correct": n_correct,
        "execution_accuracy": round(n_correct / total, 4) if total else 0.0,
        "executable_rate": round(n_executable / total, 4) if total else 0.0,
    }
    return {"aggregate": aggregate, "per_example": per_example_results}


def print_summary(aggregate: dict) -> None:
    print("\n" + "=" * 55)
    print("  FINE-TUNED MODEL EVALUATION SUMMARY")
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
        description="Run fine-tuned model text-to-SQL evaluation"
    )
    parser.add_argument(
        "--adapter", type=str, required=True,
        help="HF Hub repo ID or local path to LoRA adapter"
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Number of eval examples to use (default: all 150)"
    )
    args = parser.parse_args()

    results = run_finetuned_eval(
        adapter_path=args.adapter,
        num_samples=args.num_samples,
    )
    print_summary(results["aggregate"])

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")
