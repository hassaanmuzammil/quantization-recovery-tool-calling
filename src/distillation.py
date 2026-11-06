from __future__ import annotations

import json
from pathlib import Path
import re
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


ASSISTANT_TURN = re.compile(r"<\|im_start\|>assistant\n(.*?<\|im_end\|>)", re.DOTALL)


def assistant_labels(texts: Sequence[str], offsets: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Return ``input_ids`` with every token outside an assistant turn set to -100.

    An assistant turn covers the response text and its closing ``<|im_end|>``,
    so the model is trained to emit the call and then stop.
    """
    labels = torch.full_like(input_ids, -100)
    for row, text in enumerate(texts):
        spans = [match.span(1) for match in ASSISTANT_TURN.finditer(text)]
        for column, (start, end) in enumerate(offsets[row].tolist()):
            if end > start and any(span_start <= start and end <= span_end for span_start, span_end in spans):
                labels[row, column] = input_ids[row, column]
    return labels


def sft_loss(student_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = student_logits[:, :-1].float()
    targets = labels[:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)


def accuracy_recovery_percentage(recovered: float, degraded: float, teacher: float) -> float:
    denominator = abs(degraded - teacher)
    return float("nan") if denominator == 0 else 100.0 * (recovered - degraded) / denominator
