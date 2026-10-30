import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# ==========================================
# Configuration
# ==========================================

DATASET_NAME = "xlam"  # "fineweb-edu", "c4", "openhermes", "xlam"

NUM_EXAMPLES = 10_000
SEED = 42
OUTPUT_DIR = Path("datasets")

# Tokenizer whose chat template is used to render xLAM tool calls.
CHAT_TEMPLATE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

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

    tokenizer = AutoTokenizer.from_pretrained(CHAT_TEMPLATE_MODEL)

# ==========================================
# xLAM -> JSON schema conversion
# ==========================================

def xlam_type_to_json_type(xlam_type):
    base = xlam_type.split(",")[0].strip().lower()

    if base in ("str", "string"):
        return "string"
    if base == "int":
        return "integer"
    if base == "float":
        return "number"
    if base == "bool":
        return "boolean"
    if base.startswith(("list", "tuple", "set")):
        return "array"
    if base.startswith("dict"):
        return "object"

    # Callable, free-form, or malformed types.
    return "string"


def xlam_tool_to_json_schema(tool):
    properties = {}
    required = []

    for name, param in tool.get("parameters", {}).items():
        xlam_type = param.get("type", "str")

        properties[name] = {
            "type": xlam_type_to_json_type(xlam_type),
            "description": param.get("description", ""),
        }

        if "optional" not in xlam_type.lower():
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }

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

        # Render in the model's native tool-calling format:
        # <tools>...</tools> in the system turn, <tool_call>...</tool_call>
        # in the assistant turn.
        messages = [
            {
                "role": "user", 
                "content": query.strip()
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": answer["name"],
                            "arguments": answer.get("arguments", {}),
                        },
                    }
                    for answer in answers
                ],
            },
        ]

        return tokenizer.apply_chat_template(
            messages,
            tools=[xlam_tool_to_json_schema(tool) for tool in tools],
            tokenize=False,
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