from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import random

import torch
from transformers import get_linear_schedule_with_warmup

from .adapters import add_lora_adapters, save_adapter_checkpoint
from .distillation import batches, distillation_loss, read_texts
from .model_utils import load_model, load_tokenizer, set_seed
from .recover_config import RecoverLoRAConfig


def _optimizer_steps(config: RecoverLoRAConfig, batch_count: int) -> int:
    return config.epochs * math.ceil(batch_count / config.gradient_accumulation_steps)


def _split_texts(texts: list[str], validation_split: float, seed: int) -> tuple[list[str], list[str]]:
    if not 0.0 <= validation_split < 1.0:
        raise ValueError("validation_split must be in the range [0.0, 1.0)")
    if validation_split == 0.0:
        return texts, []

    validation_count = max(1, math.ceil(len(texts) * validation_split))
    if validation_count >= len(texts):
        raise ValueError("validation_split leaves no examples for training")

    shuffled = list(texts)
    random.Random(seed).shuffle(shuffled)
    return shuffled[validation_count:], shuffled[:validation_count]


@torch.no_grad()
def _validation_loss(model, teacher, tokenizer, texts, config: RecoverLoRAConfig) -> float:
    if not texts:
        return float("nan")

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in batches(texts, config.batch_size):
        encoded = tokenizer(
            list(batch),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_sequence_length,
        ).to(config.device)
        teacher_logits = teacher(**encoded).logits
        student_logits = model(**encoded).logits
        mask = encoded.attention_mask
        total_loss += distillation_loss(teacher_logits, student_logits, mask).item() * mask.sum().item()
        total_tokens += mask.sum().item()
    return total_loss / total_tokens


def _resolve_lora_targets(config: RecoverLoRAConfig) -> RecoverLoRAConfig:
    if config.lora_targets is not None:
        return config

    metadata_path = Path(config.degraded_model) / "degradation_config.json"
    method = "random-perturbation"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as handle:
            method = json.load(handle).get("method", method)

    targets_by_method = {
        "random-perturbation": ("k_proj", "v_proj"),
        "gate-up-quantization": ("gate_proj", "up_proj"),
    }
    try:
        targets = targets_by_method[method]
    except KeyError as exc:
        raise ValueError(f"Unsupported degradation method in {metadata_path}: {method}") from exc
    print(f"Using LoRA targets {targets} for degradation method {method}", flush=True)
    return replace(config, lora_targets=targets)


def train(config: RecoverLoRAConfig, synthetic_data: Path, checkpoint: Path) -> None:
    set_seed(config.seed)
    config = _resolve_lora_targets(config)
    tokenizer = load_tokenizer(config.teacher_model)
    teacher = load_model(config.teacher_model, config.device, config.dtype).eval()
    teacher.requires_grad_(False)
    student = load_model(config.degraded_model, config.device, config.dtype)
    add_lora_adapters(student, config)

    texts = read_texts(synthetic_data)
    if not texts:
        raise ValueError(f"Synthetic data is empty: {synthetic_data}")
    if config.validation_interval_steps <= 0:
        raise ValueError("validation_interval_steps must be positive")
    train_texts, validation_texts = _split_texts(texts, config.validation_split, config.seed)
    print(
        f"Using {len(train_texts)} training examples and "
        f"{len(validation_texts)} validation examples",
        flush=True,
    )
    batch_count = math.ceil(len(train_texts) / config.batch_size)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in student.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=_optimizer_steps(config, batch_count),
    )

    optimizer.zero_grad(set_to_none=True)
    total_steps = 0
    next_validation = config.validation_interval_steps
    for epoch in range(config.epochs):
        student.train()
        epoch_batches = list(batches(train_texts, config.batch_size))
        for group_start in range(0, len(epoch_batches), config.gradient_accumulation_steps):
            group = epoch_batches[group_start : group_start + config.gradient_accumulation_steps]
            group_size = len(group)
            group_loss = 0.0
            for batch in group:
                encoded = tokenizer(
                    list(batch),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.max_sequence_length,
                ).to(config.device)
                with torch.no_grad():
                    teacher_logits = teacher(**encoded).logits
                student_logits = student(**encoded).logits
                mask = encoded.attention_mask.to(student_logits.dtype)
                loss = distillation_loss(teacher_logits, student_logits, mask)
                group_loss += loss.item()
                (loss / group_size).backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            total_steps += 1
            print(
                f"epoch {epoch + 1}/{config.epochs} | "
                f"step {total_steps} | loss {group_loss / group_size:.6f}",
                flush=True,
            )

            if validation_texts and total_steps >= next_validation:
                validation = _validation_loss(student, teacher, tokenizer, validation_texts, config)
                print(
                    f"step {total_steps} | validation_loss {validation:.6f}",
                    flush=True,
                )
                while next_validation <= total_steps:
                    next_validation += config.validation_interval_steps

    save_adapter_checkpoint(student, tokenizer, checkpoint, config)
