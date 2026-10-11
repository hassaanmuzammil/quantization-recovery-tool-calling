from __future__ import annotations

import argparse
from pathlib import Path

from .recover_config import RecoverLoRAConfig
from .recover_lora import save_degraded_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a degraded Recover-LoRA student checkpoint")
    parser.add_argument("--teacher-model", required=True, type=Path)
    parser.add_argument("--output", default="outputs/degraded_model", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument(
        "--method",
        choices=("random-perturbation", "gate-up-quantization"),
        default="random-perturbation",
        help="How to degrade the teacher checkpoint",
    )
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--bits", type=int, choices=(2, 4), default=2)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    config = RecoverLoRAConfig(
        teacher_model=str(args.teacher_model),
        degraded_model=str(args.output),
        device=args.device,
        dtype=args.dtype,
        perturbation_std=args.noise_std,
        degradation_method=args.method,
        quantization_group_size=args.group_size,
        quantization_bits=args.bits,
        seed=args.seed,
    )
    save_degraded_model(config)
    print(f"Saved degraded model to {args.output}")


if __name__ == "__main__":
    main()
