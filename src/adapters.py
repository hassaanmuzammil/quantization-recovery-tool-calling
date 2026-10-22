from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .recover_config import RecoverLoRAConfig


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(hidden_states, self.lora_a), self.lora_b)
        return self.base(hidden_states) + self.scale * update


def _replace_targets(module: nn.Module, targets: set[str], rank: int, alpha: int) -> int:
    replaced = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child_name in targets:
            setattr(module, child_name, LoRALinear(child, rank, alpha))
            replaced += 1
        else:
            replaced += _replace_targets(child, targets, rank, alpha)
    return replaced


def add_lora_adapters(model: nn.Module, config: RecoverLoRAConfig) -> int:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if config.lora_targets is None:
        raise ValueError("LoRA targets must be resolved before adding adapters")
    replaced = _replace_targets(model, set(config.lora_targets), config.lora_rank, config.lora_alpha)
    if replaced == 0:
        raise ValueError(f"No target Linear modules found for {config.lora_targets}")
    return replaced


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name.endswith(("lora_a", "lora_b"))
    }


def save_adapter_checkpoint(model: nn.Module, tokenizer, checkpoint: Path, config: RecoverLoRAConfig) -> None:
    checkpoint.mkdir(parents=True, exist_ok=True)
    torch.save(adapter_state_dict(model), checkpoint / "adapter_model.pt")
    metadata = {
        "base_model": config.degraded_model,
        "teacher_model": config.teacher_model,
        "lora_targets": list(config.lora_targets),
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
    }
    (checkpoint / "adapter_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # Some CLI arguments (for example ``--log-dir``) are parsed as ``Path``
    # objects.  Serialize those values as strings in the JSON metadata.
    (checkpoint / "recover_lora_config.json").write_text(
        json.dumps(asdict(config), indent=2, default=str),
        encoding="utf-8",
    )
    tokenizer.save_pretrained(checkpoint)
