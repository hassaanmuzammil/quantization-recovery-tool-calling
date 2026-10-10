from __future__ import annotations

import math
from pathlib import Path

import torch
from transformers import get_linear_schedule_with_warmup

from .adapters import add_lora_adapters, save_adapter_checkpoint
from .distillation import batches, distillation_loss, read_texts
from .model_utils import load_model, load_tokenizer, set_seed
from .recover_config import RecoverLoRAConfig


def _optimizer_steps(config: RecoverLoRAConfig, batch_count: int) -> int:
    return config.epochs * math.ceil(batch_count / config.gradient_accumulation_steps)


def train(config: RecoverLoRAConfig, synthetic_data: Path, checkpoint: Path) -> None:
    set_seed(config.seed)
    tokenizer = load_tokenizer(config.teacher_model)
    teacher = load_model(config.teacher_model, config.device, config.dtype).eval()
    teacher.requires_grad_(False)
    student = load_model(config.degraded_model, config.device, config.dtype)
    add_lora_adapters(student, config)

    texts = read_texts(synthetic_data)
    if not texts:
        raise ValueError(f"Synthetic data is empty: {synthetic_data}")
    batch_count = math.ceil(len(texts) / config.batch_size)
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
    for epoch in range(config.epochs):
        epoch_batches = list(batches(texts, config.batch_size))
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

    save_adapter_checkpoint(student, tokenizer, checkpoint, config)
