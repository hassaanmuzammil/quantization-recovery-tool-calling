"""Run original, degraded, or recovered models on a supported benchmark."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM

from src.adapters import add_lora_adapters


BENCHMARKS = ("hellaswag", "ifeval", "mmlu")
METHODS = ("original", "degraded", "recovered")
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_DEGRADED_MODEL = (
    "~/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.2-1B-Instruct-degraded"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Recover-LoRA benchmarks")
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--method", choices=METHODS, default="original")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--degraded-model", default=DEFAULT_DEGRADED_MODEL)
    parser.add_argument("--adapter-dir", default="outputs/recover_lora/adapters")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _model_shortname(model_name: str) -> str:
    if "Llama-3.2-1B-Instruct" in model_name:
        return "llama32_1b"
    if "Qwen3-4B-Instruct-2507" in model_name:
        return "qwen3_4b"
    return Path(model_name.rstrip("/")).name.replace(".", "_").replace("-", "_")


def _load_degraded_model(args: argparse.Namespace):
    return AutoModelForCausalLM.from_pretrained(
        Path(args.degraded_model).expanduser(),
        torch_dtype=_torch_dtype(args.dtype),
        device_map=args.device,
    )


def _load_recovered_model(args: argparse.Namespace):
    adapter_dir = Path(args.adapter_dir).expanduser()
    with (adapter_dir / "adapter_config.json").open(encoding="utf-8") as file:
        adapter_config = json.load(file)

    model = AutoModelForCausalLM.from_pretrained(
        Path(args.degraded_model).expanduser(),
        torch_dtype=_torch_dtype(args.dtype),
        device_map=args.device,
    )
    lora_config = SimpleNamespace(
        lora_targets=tuple(adapter_config["lora_targets"]),
        lora_rank=adapter_config["lora_rank"],
        lora_alpha=adapter_config["lora_alpha"],
    )
    add_lora_adapters(model, lora_config)

    adapter_state = torch.load(adapter_dir / "adapter_model.pt", map_location="cpu")
    _, unexpected_keys = model.load_state_dict(adapter_state, strict=False)
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
    args = _parser().parse_args()
    model_shortname = _model_shortname(args.model)

    task_kwargs = {
        "model": "hf",
        "tasks": [args.benchmark],
        "device": args.device,
        "batch_size": args.batch_size,
        "log_samples": args.method == "original",
    }
    if args.benchmark == "ifeval":
        task_kwargs["apply_chat_template"] = True

    if args.method == "original":
        task_kwargs["model_args"] = f"pretrained={args.model},dtype={args.dtype}"
        output_dir = Path(__file__).resolve().parent / "outputs" / f"{args.benchmark}_{model_shortname}"
    else:
        model = (
            _load_degraded_model(args)
            if args.method == "degraded"
            else _load_recovered_model(args)
        )
        task_kwargs["model"] = HFLM(
            pretrained=model,
            batch_size=args.batch_size,
            device=args.device,
        )
        output_dir = (
            Path(__file__).resolve().parent
            / "outputs"
            / f"{args.benchmark}_{model_shortname}_{args.method}"
        )

    results = evaluator.simple_evaluate(**task_kwargs)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with (output_dir / f"results_{timestamp}.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, default=str)


if __name__ == "__main__":
    main()
