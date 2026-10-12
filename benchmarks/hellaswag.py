import json
from datetime import datetime
from pathlib import Path

from lm_eval import evaluator

# model_id = "Qwen/Qwen3-4B-Instruct-2507"
model_id = "meta-llama/Llama-3.2-1B-Instruct"

output_dir = Path(__file__).resolve().parent / "outputs" / "hellaswag"
output_dir.mkdir(parents=True, exist_ok=True)

results = evaluator.simple_evaluate(
    model="hf",
    model_args=f"pretrained={model_id},dtype=bfloat16",
    tasks=["hellaswag"],
    device="cuda:0",
    batch_size=4,
    log_samples=True,
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with (output_dir / f"results_{timestamp}.json").open("w", encoding="utf-8") as file:
    json.dump(results, file, indent=2, default=str)
