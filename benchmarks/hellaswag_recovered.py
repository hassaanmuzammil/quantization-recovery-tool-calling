"""Evaluate the degraded Llama checkpoint after applying the recovered LoRA adapter."""

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM

from src.adapters import add_lora_adapters


DEGRADED_MODEL = Path(
    "~/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.2-1B-Instruct-degraded"
).expanduser()
ADAPTER_DIR = Path("outputs/recover_lora/adapters")
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "hellaswag_recovered"


def load_recovered_model():
    adapter_config_path = ADAPTER_DIR / "adapter_config.json"
    adapter_weights_path = ADAPTER_DIR / "adapter_model.pt"

    with adapter_config_path.open(encoding="utf-8") as file:
        adapter_config = json.load(file)

    model = AutoModelForCausalLM.from_pretrained(
        DEGRADED_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    lora_config = SimpleNamespace(
        lora_targets=tuple(adapter_config["lora_targets"]),
        lora_rank=adapter_config["lora_rank"],
        lora_alpha=adapter_config["lora_alpha"],
    )
    add_lora_adapters(model, lora_config)

    adapter_state = torch.load(adapter_weights_path, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(adapter_state, strict=False)
    unexpected_adapter_keys = [
        key for key in unexpected_keys if key.endswith(("lora_a", "lora_b"))
    ]
    if unexpected_adapter_keys or not any(key.endswith("lora_a") for key in adapter_state):
        raise RuntimeError(
            "The adapter checkpoint did not load into the degraded model: "
            f"unexpected adapter keys={unexpected_adapter_keys}"
        )

    model.eval()
    return model


def main() -> None:
    model = HFLM(
        pretrained=load_recovered_model(),
        batch_size=4,
        device="cuda:0",
    )
    results = evaluator.simple_evaluate(
        model=model,
        tasks=["hellaswag"],
        batch_size=4,
        log_samples=False,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with (OUTPUT_DIR / f"results_{timestamp}.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, default=str)


if __name__ == "__main__":
    main()
