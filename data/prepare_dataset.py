"""
SQL-LoRA — Dataset Preparation

Downloads the b-mc2/sql-create-context dataset from Hugging Face,
reports complexity statistics (JOIN / GROUP BY / subquery counts),
drops duplicates and malformed entries, splits into training and
held-out evaluation sets with stratified complexity, and saves as
JSONL files for training and evaluation scripts.

Usage:
    python data/prepare_dataset.py
"""

import json
import os
import re
from collections import Counter

from datasets import load_dataset


TRAIN_SIZE = 1200
EVAL_SIZE = 150
DATASET_ID = "b-mc2/sql-create-context"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def complexity_label(sql: str) -> str:
    """Classify SQL complexity based on structural features.

    - 'simple': single-table SELECT with optional WHERE, no joins/aggregations/subqueries
    - 'medium': includes GROUP BY, aggregation functions, or DISTINCT (but no joins or subqueries)
    - 'complex': includes JOINs or subqueries (nested SELECTs)
    """
    joins = len(re.findall(r'\bJOIN\b', sql, re.IGNORECASE))
    groupbys = len(re.findall(r'\bGROUP BY\b', sql, re.IGNORECASE))
    subqueries = max(0, sql.lower().count('select') - 1)

    if joins > 0 or subqueries > 0:
        return 'complex'
    if groupbys > 0:
        return 'medium'
    return 'simple'


def compute_complexity_stats(examples: list[dict], label: str) -> dict:
    """Return per-complexity counts and percentage for a list of examples."""
    counts: dict[str, int] = Counter()
    for ex in examples:
        counts[ex["_complexity"]] += 1
    total = len(examples)
    return {
        "label": label,
        "total": total,
        "by_complexity": dict(counts),
        "pct_complex": round(counts.get("complex", 0) / total * 100, 2) if total else 0,
    }


def download_and_split() -> dict:
    """Download, clean, stratify, and return train/eval dicts."""
    raw = load_dataset(DATASET_ID, split="train")
    print(f"Downloaded {len(raw):,} total examples")

    # ── Clean ──────────────────────────────────────────────────────────
    seen: set[tuple[str, str]] = set()
    cleaned: list[dict] = []
    dropped_duplicates = 0
    dropped_malformed = 0

    for ex in raw:
        sql = ex["answer"]
        ctx = ex["context"]
        q = ex["question"]

        if not sql or not sql.strip():
            dropped_malformed += 1
            continue
        if not ctx or not ctx.strip():
            dropped_malformed += 1
            continue
        if not q or not q.strip():
            dropped_malformed += 1
            continue

        key = (q.strip(), ctx.strip())
        if key in seen:
            dropped_duplicates += 1
            continue
        seen.add(key)

        cleaned.append({
            "question": q.strip(),
            "context": ctx.strip(),
            "answer": sql.strip(),
            "_complexity": complexity_label(sql),
        })

    print(f"Cleaned: {len(cleaned):,} kept, "
          f"{dropped_duplicates} duplicates dropped, "
          f"{dropped_malformed} malformed dropped")

    # ── Stats ──────────────────────────────────────────────────────────
    stats = compute_complexity_stats(cleaned, "full_dataset")
    print(f"Complexity distribution (full): {stats['by_complexity']}")

    # ── Stratified split ───────────────────────────────────────────────
    by_cplx: dict[str, list[dict]] = {"simple": [], "medium": [], "complex": []}
    for ex in cleaned:
        by_cplx[ex["_complexity"]].append(ex)

    # Target: ~50 of each complexity in eval set (balanced), rest go to train
    eval_per_cplx = EVAL_SIZE // 3  # 50
    train_pool = []
    eval_set = []

    for cplx in ("simple", "medium", "complex"):
        pool = by_cplx[cplx]
        n_eval = min(eval_per_cplx, len(pool))
        n_train = min(TRAIN_SIZE // 3, len(pool) - n_eval)

        eval_set.extend(pool[:n_eval])
        train_pool.extend(pool[n_eval : n_eval + n_train])

    # Fill remaining train slots from whichever complexity has spare examples
    remaining = TRAIN_SIZE - len(train_pool)
    if remaining > 0:
        for cplx in ("simple", "medium", "complex"):
            pool = by_cplx[cplx]
            already_taken = (
                sum(1 for ex in train_pool if ex["_complexity"] == cplx)
                + sum(1 for ex in eval_set if ex["_complexity"] == cplx)
            )
            available = len(pool) - already_taken
            add = min(remaining, available)
            train_pool.extend(pool[already_taken : already_taken + add])
            remaining -= add
            if remaining <= 0:
                break

    # Remove internal complexity key before saving
    for ex in train_pool:
        del ex["_complexity"]
    for ex in eval_set:
        del ex["_complexity"]

    # ── Report ─────────────────────────────────────────────────────────
    print(f"\nTrain set: {len(train_pool)} examples")
    train_cplx = Counter()
    for ex in train_pool:
        train_cplx[complexity_label(ex["answer"])] += 1
    print(f"  By complexity: {dict(train_cplx)}")

    print(f"Eval set: {len(eval_set)} examples")
    eval_cplx = Counter()
    for ex in eval_set:
        eval_cplx[complexity_label(ex["answer"])] += 1
    print(f"  By complexity: {dict(eval_cplx)}")

    return {"train": train_pool, "eval": eval_set}


def save_jsonl(examples: list[dict], path: str) -> None:
    """Save a list of dicts as newline-delimited JSON."""
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved {len(examples):,} examples to {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare SQL-LoRA dataset")
    parser.add_argument("--train_size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--eval_size", type=int, default=EVAL_SIZE)
    args = parser.parse_args()

    TRAIN_SIZE = args.train_size
    EVAL_SIZE = args.eval_size

    result = download_and_split()
    save_jsonl(result["train"], os.path.join(OUTPUT_DIR, "train.jsonl"))
    save_jsonl(result["eval"], os.path.join(OUTPUT_DIR, "eval.jsonl"))
    print("Dataset preparation complete.")
