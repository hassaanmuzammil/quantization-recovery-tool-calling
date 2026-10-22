import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# ==========================================
# Configuration
# ==========================================

DATASET_NAME = "openhermes"  # "fineweb-edu", "c4", "openhermes"

NUM_EXAMPLES = 10_000
OUTPUT_DIR = Path("datasets")

# ==========================================
# Dataset configurations
# ==========================================

DATASETS = {
    "fineweb-edu": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "output": "fineweb_edu.jsonl",
    },
    "c4": {
        "path": "allenai/c4",
        "name": "en",
        "output": "allenai_c4.jsonl",
    },
    "openhermes": {
        "path": "teknium/openhermes",
        "name": None,
        "output": "openhermes_10k.jsonl",
    },
}

# ==========================================
# Load dataset
# ==========================================

config = DATASETS[DATASET_NAME]

print(f"Streaming {DATASET_NAME}...")

dataset = load_dataset(
    config["path"],
    name=config["name"],
    split="train",
    streaming=True,
)

# ==========================================
# Extract text
# ==========================================

def extract_text(example):
    if DATASET_NAME == "openhermes":
        return "\n\n".join(
            example[key].strip()
            for key in ("instruction", "input", "output")
            if example.get(key) and example[key].strip()
        )

    return example.get("text", "").strip()


# ==========================================
# Stream and save
# ==========================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / config["output"]

saved = 0

with output_path.open("w", encoding="utf-8") as f:
    with tqdm(
        total=NUM_EXAMPLES,
        desc=f"Processing {DATASET_NAME}",
        unit="rows",
    ) as progress:

        for example in dataset:
            text = extract_text(example)

            if not text:
                continue

            f.write(
                json.dumps(
                    {"text": text},
                    ensure_ascii=False,
                ) + "\n"
            )

            saved += 1
            progress.update(1)

            if saved >= NUM_EXAMPLES:
                break

print(f"\nSaved {saved} examples to {output_path}")
