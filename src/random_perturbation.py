from __future__ import annotations

import torch
from torch import nn

from .model_utils import set_seed


def perturb_kv_projections(model: nn.Module, noise_std: float, seed: int) -> nn.Module:
    """Add seeded Gaussian noise to attention K/V projection weights."""
    set_seed(seed)
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name.endswith(("k_proj", "v_proj")):
                module.weight.add_(noise_std * torch.randn_like(module.weight))
    return model
