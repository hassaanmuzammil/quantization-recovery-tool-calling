import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# ==========================================
# Configuration
# ==========================================

DATASET_NAME = "xlam"  # "fineweb-edu", "c4", "openhermes", "xlam"

NUM_EXAMPLES = 10_000
SEED = 42
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
    "xlam": {
        "path": "Salesforce/xlam-function-calling-60k",
        "name": None,
        "output": "xlam_function_calling_10k.jsonl",
    },
}

# ==========================================
# Load dataset
# ==========================================

config = DATASETS[DATASET_NAME]

print(f"Loading {DATASET_NAME}...")

dataset = load_dataset(
    config["path"],
    name=config["name"],
    split="train",
    streaming=(DATASET_NAME != "xlam"),
)

# ==========================================
# Random sampling for xLAM
# ==========================================

if DATASET_NAME == "xlam":
    dataset = dataset.shuffle(seed=SEED)

    # Take 10K random examples
    dataset = dataset.select(
        range(min(NUM_EXAMPLES, len(dataset)))
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

    if DATASET_NAME == "xlam":

        query = example.get("query", "")
        tools = example.get("tools", "[]")
        answers = example.get("answers", "[]")

        # xLAM stores tools and answers as JSON strings.
        if isinstance(tools, str):
            tools = json.loads(tools)

        if isinstance(answers, str):
            answers = json.loads(answers)

        text = (
            "Available tools:\n"
            + json.dumps(tools, ensure_ascii=False, indent=2)
            + "\n\nUser query:\n"
            + query.strip()
            + "\n\nTool calls:\n"
            + json.dumps(answers, ensure_ascii=False)
        )

        return text

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