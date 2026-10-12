import json
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

# base_model_id = "Qwen/Qwen3-4B-Instruct-2507"
base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
degraded_model_path: Path | None = Path(
    "~/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.2-1B-Instruct-degraded"
).expanduser()

output_dir = Path(__file__).resolve().parent / "outputs" / "hellaswag_perturbed"
output_dir.mkdir(parents=True, exist_ok=True)

if degraded_model_path is not None:
    # load from path
    model = AutoModelForCausalLM.from_pretrained(
        degraded_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
else:
    # load original model and randomly perturb weights
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
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

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with (output_dir / f"results_{timestamp}.json").open("w", encoding="utf-8") as file:
    json.dump(results, file, indent=2, default=str)
