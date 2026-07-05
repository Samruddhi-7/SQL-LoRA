"""
SQL-LoRA — Hyperparameter Configuration

All tunable constants are defined here as named constants so the
training notebook and evaluation scripts can import them from one place.
"""

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
BASE_MODEL_ID: str = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
# Fallback: "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
# Verify exact model strings on https://huggingface.co/unsloth before training.

# ---------------------------------------------------------------------------
# QLoRA / LoRA hyperparameters
# ---------------------------------------------------------------------------
LORA_R: int = 16
LORA_ALPHA: int = 16  # best practice: alpha = r (not 2x rank)
LORA_DROPOUT: float = 0.0

TARGET_MODULES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
PER_DEVICE_TRAIN_BATCH_SIZE: int = 2
GRADIENT_ACCUMULATION_STEPS: int = 4
MAX_STEPS: int = 200
LEARNING_RATE: float = 2e-4
WARMUP_RATIO: float = 0.03
LOGGING_STEPS: int = 10
SAVE_STEPS: int = 50
EVAL_STEPS: int = 50

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_ID: str = "b-mc2/sql-create-context"
TRAIN_SPLIT_RATIO: float = 0.85
VAL_SPLIT_RATIO: float = 0.075
TEST_SPLIT_RATIO: float = 0.075
MAX_SEQ_LENGTH: int = 512

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS: int = 256
TEMPERATURE: float = 0.1
TOP_P: float = 0.95

# ---------------------------------------------------------------------------
# Hugging Face Hub
# ---------------------------------------------------------------------------
HF_ADAPTER_REPO: str = ""  # set to "your-username/sql-lora-llama3.2-3b" after training
