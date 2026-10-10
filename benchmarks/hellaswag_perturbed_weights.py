import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

model_id = "Qwen/Qwen3-4B-Instruct-2507"

output_dir = Path("outputs/hellaswag_perturbed")
output_dir.mkdir(parents=True, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)

# Perturb K and V projection weights
with torch.no_grad():
    for name, module in model.named_modules():
        if name.endswith(("k_proj", "v_proj")):
            module.weight.add_(0.01 * torch.randn_like(module.weight))

model = HFLM(
    pretrained=model,
    batch_size=4,
)

results = evaluator.simple_evaluate(
    model=model,
    tasks=["hellaswag"],
    device="cuda:0",
    batch_size=4,
    log_samples=False,
)

with (output_dir / "results.json").open("w", encoding="utf-8") as file:
    json.dump(results, file, indent=2, default=str)