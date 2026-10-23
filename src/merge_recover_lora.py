"""Merge a Recover-LoRA adapter into its degraded base model.

The Recover-LoRA training code stores adapters as ``adapter_model.pt`` with
custom ``lora_a``/``lora_b`` parameters.  This utility folds those parameters
into the base Linear weights and writes a normal Transformers checkpoint that
can be served by BFCL/SGLang.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import torch

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge a Recover-LoRA adapter into a degraded model"
    )
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    return parser


def _torch_dtype(name: str):
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _merge(module) -> None:
    try:
        from .adapters import LoRALinear
    except ImportError:  # Also support: python src/merge_recover_lora.py
        from src.adapters import LoRALinear

    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            with torch.no_grad():
                child.base.weight.add_(child.scale * (child.lora_b @ child.lora_a))
            setattr(module, name, child.base)
        else:
            _merge(child)


def _copy_tokenizer_files(base_model: Path, output_dir: Path) -> None:
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        source = base_model / filename
        if source.is_file():
            shutil.copy2(source, output_dir / filename)


def main() -> None:
    args = _parser().parse_args()
    import torch
    from transformers import AutoModelForCausalLM

    try:
        from .adapters import add_lora_adapters
    except ImportError:  # Also support: python src/merge_recover_lora.py
        from src.adapters import add_lora_adapters

    base_model = args.base_model.expanduser().resolve()
    adapter_dir = args.adapter_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    with (adapter_dir / "adapter_config.json").open(encoding="utf-8") as file:
        adapter_config = json.load(file)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=_torch_dtype(args.dtype),
        device_map="cpu",
    )

    lora_config = SimpleNamespace(
        lora_targets=tuple(adapter_config["lora_targets"]),
        lora_rank=adapter_config["lora_rank"],
        lora_alpha=adapter_config["lora_alpha"],
    )
    replaced = add_lora_adapters(model, lora_config)

    adapter_state = torch.load(
        adapter_dir / "adapter_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    missing, unexpected = model.load_state_dict(adapter_state, strict=False)
    unexpected_adapter = [
        key for key in unexpected if key.endswith(("lora_a", "lora_b"))
    ]
    if unexpected_adapter:
        raise RuntimeError(
            "Adapter parameters did not load into the base model: "
            f"{unexpected_adapter[:5]}"
        )

    _merge(model)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    _copy_tokenizer_files(base_model, output_dir)

    print(f"Wrapped {replaced} target Linear modules")
    if missing:
        print(f"Ignored missing non-adapter parameters: {len(missing)}")
    print(f"Merged model written to {output_dir}")


if __name__ == "__main__":
    main()
