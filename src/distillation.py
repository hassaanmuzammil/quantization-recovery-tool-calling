from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


def read_texts(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line)["text"] for line in handle if line.strip()]


def batches(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def distillation_loss(teacher_logits: torch.Tensor, student_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    teacher_probs = teacher_logits.float().softmax(dim=-1)
    student_log_probs = student_logits.float().log_softmax(dim=-1)
    token_loss = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    return (token_loss * mask).sum() / mask.sum().clamp_min(1)


def accuracy_recovery_percentage(recovered: float, degraded: float, teacher: float) -> float:
    denominator = abs(degraded - teacher)
    return float("nan") if denominator == 0 else 100.0 * (recovered - degraded) / denominator
