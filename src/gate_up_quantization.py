from __future__ import annotations

import torch
from torch import nn


def quantize_gate_up_projections(
    model: nn.Module, bits: int, group_size: int = 32
) -> nn.Module:
    """Apply groupwise affine fake quantization to MLP gate/up weights.

    Quantized values are dequantized back into the model's original dtype for
    recovery experiments; this does not create a packed low-bit model.
    """
    if bits not in (2, 4):
        raise ValueError("bits must be 2 or 4")
    if group_size < 1:
        raise ValueError("group_size must be positive")

    levels = (1 << bits) - 1
    quantized_layers = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear) or not name.endswith(("gate_proj", "up_proj")):
                continue
            quantized_layers += 1

            weight = module.weight.data
            source = weight.float()
            quantized = torch.empty_like(source)
            for start in range(0, source.shape[1], group_size):
                end = min(start + group_size, source.shape[1])
                group = source[:, start:end]
                minimum = group.amin(dim=1, keepdim=True)
                maximum = group.amax(dim=1, keepdim=True)
                scale = (maximum - minimum) / levels
                safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
                codes = torch.round((group - minimum) / safe_scale).clamp_(0, levels)
                quantized[:, start:end] = codes * safe_scale + minimum

            weight.copy_(quantized.to(dtype=weight.dtype))
    if quantized_layers == 0:
        raise ValueError("Model has no nn.Linear layers named gate_proj or up_proj")
    return model
