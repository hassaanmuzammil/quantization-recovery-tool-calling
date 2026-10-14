from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import random

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

from .adapters import add_lora_adapters, save_adapter_checkpoint
from .distillation import batches, distillation_loss, read_texts
from .model_utils import load_model, load_tokenizer, set_seed
from .recover_config import RecoverLoRAConfig


def _rng_state() -> dict:
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_training_checkpoint(
    model,
    tokenizer,
    checkpoint: Path,
    config: RecoverLoRAConfig,
    optimizer,
    scheduler,
    total_steps: int,
    next_epoch: int,
    next_group_start: int,
    next_validation: int,
) -> None:
    save_adapter_checkpoint(model, tokenizer, checkpoint, config)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "total_steps": total_steps,
            "next_epoch": next_epoch,
            "next_group_start": next_group_start,
            "next_validation": next_validation,
            "rng": _rng_state(),
        },
        checkpoint / "training_state.pt",
    )
    print(f"Saved resumable checkpoint: {checkpoint} (step {total_steps})", flush=True)


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
    log_dir = Path(config.log_dir) if config.log_dir else Path(config.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_text(
        "run/config",
        json.dumps(
            {
                "config": config.__dict__,
                "teacher_model": config.teacher_model,
                "degraded_model": config.degraded_model,
                "synthetic_data": str(synthetic_data),
                "checkpoint": str(checkpoint),
            },
            indent=2,
            default=str,
        ),
        0,
    )
    writer.add_text("run/models", f"Teacher: `{config.teacher_model}`\n\nStudent: `{config.degraded_model}`", 0)
    writer.add_text("run/paths", f"Synthetic data: `{synthetic_data}`\n\nCheckpoint: `{checkpoint}`", 0)
    degradation_metadata = Path(config.degraded_model) / "degradation_config.json"
    if degradation_metadata.exists():
        writer.add_text(
            "run/degradation",
            degradation_metadata.read_text(encoding="utf-8"),
            0,
        )
    writer.add_scalar("data/train_examples", len(train_texts), 0)
    writer.add_scalar("data/validation_examples", len(validation_texts), 0)
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
    writer.add_scalar("model/total_parameters", sum(parameter.numel() for parameter in student.parameters()), 0)
    writer.add_scalar(
        "model/trainable_parameters",
        sum(parameter.numel() for parameter in student.parameters() if parameter.requires_grad),
        0,
    )

    optimizer.zero_grad(set_to_none=True)
    total_steps = 0
    next_validation = config.validation_interval_steps
    start_epoch = 0
    start_group_start = 0
    if config.resume:
        resume_dir = checkpoint / "checkpoints" / "_latest"
        state_path = resume_dir / "training_state.pt"
        adapter_path = resume_dir / "adapter_model.pt"
        if not state_path.exists() or not adapter_path.exists():
            raise FileNotFoundError(
                f"Latest checkpoint must contain adapter_model.pt and training_state.pt: {resume_dir}"
            )
        adapter_state = torch.load(adapter_path, map_location="cpu", weights_only=True)
        student.load_state_dict(adapter_state, strict=False)
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        total_steps = state["total_steps"]
        start_epoch = state["next_epoch"]
        start_group_start = state["next_group_start"]
        next_validation = state["next_validation"]
        _restore_rng_state(state["rng"])
        print(f"Resumed training from {resume_dir} at step {total_steps}", flush=True)

    if config.save_every_steps < 0:
        raise ValueError("save_every_steps must be non-negative")

    for epoch in range(start_epoch, config.epochs):
        student.train()
        epoch_batches = list(batches(train_texts, config.batch_size))
        group_begin = start_group_start if epoch == start_epoch else 0
        for group_start in range(group_begin, len(epoch_batches), config.gradient_accumulation_steps):
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
            train_loss = group_loss / group_size
            writer.add_scalar("train/loss", train_loss, total_steps)
            writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], total_steps)
            writer.add_scalar("train/epoch", epoch + 1, total_steps)
            writer.flush()
            print(
                f"epoch {epoch + 1}/{config.epochs} | "
                f"step {total_steps} | loss {train_loss:.6f}",
                flush=True,
            )

            if validation_texts and total_steps >= next_validation:
                validation = _validation_loss(student, teacher, tokenizer, validation_texts, config)
                print(
                    f"step {total_steps} | validation_loss {validation:.6f}",
                    flush=True,
                )
                writer.add_scalar("validation/loss", validation, total_steps)
                writer.flush()
                while next_validation <= total_steps:
                    next_validation += config.validation_interval_steps

            if config.save_every_steps and total_steps % config.save_every_steps == 0:
                next_epoch = epoch
                next_group_start = group_start + config.gradient_accumulation_steps
                if next_group_start >= len(epoch_batches):
                    next_epoch += 1
                    next_group_start = 0
                periodic_checkpoint = checkpoint / "checkpoints" / f"_{total_steps}"
                _save_training_checkpoint(
                    student,
                    tokenizer,
                    periodic_checkpoint,
                    config,
                    optimizer,
                    scheduler,
                    total_steps,
                    next_epoch,
                    next_group_start,
                    next_validation,
                )
                _save_training_checkpoint(
                    student,
                    tokenizer,
                    checkpoint / "checkpoints" / "_latest",
                    config,
                    optimizer,
                    scheduler,
                    total_steps,
                    next_epoch,
                    next_group_start,
                    next_validation,
                )

    save_adapter_checkpoint(student, tokenizer, checkpoint, config)
    writer.close()
