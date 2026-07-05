# SQL-LoRA

Fine-tuning a small open-weight LLM (Llama-3.2-3B) with QLoRA to translate natural language questions into correct SQL queries, evaluated by **execution accuracy** on a SQLite database.

## Motivation

Text-to-SQL benchmarks typically measure exact string match or BLEU against a reference query — metrics that penalize semantically equivalent but syntactically different SQL. This project instead evaluates end-to-end correctness by **executing** both the gold and predicted queries against a populated SQLite database and comparing the result sets.

## Dataset

**b-mc2/sql-create-context** ([Hugging Face](https://huggingface.co/datasets/b-mc2/sql-create-context)):
- 78,577 examples, 78,576 unique (1 duplicate)
- Each example: `context` (CREATE TABLE statement), `question` (NL), `answer` (SQL)
- Complexity splits: ~96.5% simple, ~2.1% medium (~1 JOIN + filter), ~1.4% complex (2+ JOINs / subqueries / aggregation)
- Stratified split into 1,200 train / 150 eval (50/50/50 per complexity tier)

## Evaluation Harness

`evaluation/execution_accuracy.py` implements the custom metric:

1. **Parse** CREATE TABLE statements into column names/types
2. **Populate** a SQLite database with synthetic rows (5 rows/table, 4 distinct values per column with column-specific offsets)
3. **Execute** both gold and predicted SQL
4. **Compare** result sets (column order matters — SQL semantics)

Derived columns, ORDER BY, subqueries, aggregation, joins, and LIMIT are all supported. Empty-result false positives are a known blind spot (documented in the source).

## Baseline

Zero-shot evaluation with `llama-3.1-8b-instant` via Groq API (the closest Llama-3.2-3B model was decommissioned on Groq; 8B is a stronger baseline):

| Metric | Value |
|--------|-------|
| Examples | 150 |
| Executable rate | 80.0% |
| Execution accuracy | **71.3%** |

58% of failures are complex queries (joins/subqueries).

## Fine-tuning

**Colab notebook**: [`training/train_qlora.ipynb`](training/train_qlora.ipynb)

- **Stack**: Unsloth + QLoRA on Google Colab T4 (15 GB VRAM)
- **Base model**: `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`
- **LoRA config**: rank=16, alpha=16, dropout=0, all linear modules
- **Training**: 200 steps, effective batch size 8, learning rate 2e-4
- **Loss**: masked to only train on assistant responses
- **Output**: LoRA adapter (~100 MB) pushed to Hugging Face Hub

### Train

1. Upload `data/train.jsonl` to Colab
2. Open `training/train_qlora.ipynb` in Colab
3. Set `HF_ADAPTER_REPO` to your HF Hub repo name
4. Run all cells

### Evaluate

```bash
python -m evaluation.run_finetuned_eval \
    --adapter your-username/sql-lora-llama3.2-3b \
    [--num_samples N]
```

## Project Structure

```
SQL-LoRA/
├── data/
│   ├── train.jsonl               # 1,200 training examples (prepared)
│   ├── eval.jsonl                # 150 held-out eval examples
│   ├── prepare_dataset.py        # Dataset download & stratified split
│   └── schema_to_sqlite.py       # CREATE TABLE → SQLite materialization
├── evaluation/
│   ├── execution_accuracy.py     # Core eval harness (parse, populate, execute, compare)
│   ├── run_baseline_eval.py      # Baseline evaluation via Groq API
│   └── run_finetuned_eval.py     # Fine-tuned model evaluation
├── training/
│   ├── train_qlora.ipynb         # Colab notebook for QLoRA fine-tuning
│   └── config.py                 # All hyperparameters as named constants
├── results/
│   └── baseline_results.json     # Baseline per-example results
├── tests/
│   ├── test_execution_accuracy.py # 45 tests for the eval harness
│   └── test_schema_to_sqlite.py  # 8 tests for schema materialization
├── .gitignore
├── LICENSE                       # MIT
├── requirements.txt
└── README.md
```

## Requirements

See [`requirements.txt`](requirements.txt). Key dependencies:
- `unsloth` — efficient QLoRA training on consumer GPUs
- `transformers`, `datasets`, `trl` — Hugging Face ecosystem
- `groq` — baseline inference (free tier)

## License

MIT — see [LICENSE](LICENSE).
