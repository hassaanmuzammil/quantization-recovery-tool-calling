from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RecoverLoRAConfig:
    teacher_model: str
    degraded_model: str
    output_dir: str = "outputs/recover_lora"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    lora_targets: tuple[str, ...] | None = None
    lora_rank: int = 64
    lora_alpha: int = 64
    learning_rate: float = 5e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 32
    validation_split: float = 0.1
    validation_interval_steps: int = 100
    save_every_steps: int = 1000
    resume: bool = False
    warmup_steps: int = 80
    epochs: int = 3
    synthetic_samples: int = 120_000
    greedy_tokens: int = 5
    max_sequence_length: int = 2048
    temperature: float = 1.0
    top_p: float = 0.9
    seed: int = 1234
    perturbation_std: float = 0.01
    degradation_method: str = "random-perturbation"
    quantization_group_size: int = 32
    quantization_bits: int = 2
