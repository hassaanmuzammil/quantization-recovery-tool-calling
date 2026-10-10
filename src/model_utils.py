from __future__ import annotations

import random

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def dtype_from_name(name: str) -> torch.dtype:
    values = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    try:
        return values[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_id: str, device: str, dtype: str) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype_from_name(dtype))
    return model.to(device)


def load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def perturb_kv_projections(model: nn.Module, noise_std: float, seed: int) -> nn.Module:
    set_seed(seed)
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name.endswith(("k_proj", "v_proj")):
                module.weight.add_(noise_std * torch.randn_like(module.weight))
    return model
