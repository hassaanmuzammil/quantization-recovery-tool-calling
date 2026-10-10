import json
from pathlib import Path

from lm_eval import evaluator

model_id = "Qwen/Qwen3-4B-Instruct-2507"

output_dir = Path("outputs/ifeval")
output_dir.mkdir(parents=True, exist_ok=True)

results = evaluator.simple_evaluate(
    model="hf",
    model_args=f"pretrained={model_id},dtype=bfloat16",
    tasks=["ifeval"],
    device="cuda:0",
    batch_size=4,
    log_samples=True,
    apply_chat_template=True,
)

with (output_dir / "results.json").open("w", encoding="utf-8") as file:
    json.dump(results, file, indent=2, default=str)
