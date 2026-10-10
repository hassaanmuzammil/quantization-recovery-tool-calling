from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RecoverLoRAConfig:
    teacher_model: str
    degraded_model: str
    output_dir: str = "outputs/recover_lora"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    lora_targets: tuple[str, ...] = ("k_proj", "v_proj")
    lora_rank: int = 64
    lora_alpha: int = 64
    learning_rate: float = 5e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 32
    warmup_steps: int = 80
    epochs: int = 3
    synthetic_samples: int = 120_000
    greedy_tokens: int = 5
    max_sequence_length: int = 256
    temperature: float = 1.0
    top_p: float = 0.9
    seed: int = 1234
    perturbation_std: float = 0.01
