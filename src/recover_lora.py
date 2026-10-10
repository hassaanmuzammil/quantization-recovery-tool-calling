"""CLI and backwards-compatible public imports for the Recover-LoRA pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .adapters import LoRALinear, add_lora_adapters
from .distillation import accuracy_recovery_percentage, distillation_loss
from .model_utils import load_model, load_tokenizer, perturb_kv_projections, set_seed
from .recover_config import RecoverLoRAConfig
from .synthetic_data import generate_hybrid_sample, generate_synthetic_data
from .training import train


def save_degraded_model(config: RecoverLoRAConfig) -> None:
    model = load_model(config.teacher_model, config.device, config.dtype)
    perturb_kv_projections(model, config.perturbation_std, config.seed)
    destination = Path(config.degraded_model)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    load_tokenizer(config.teacher_model).save_pretrained(destination)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recover-LoRA pipeline")
    parser.add_argument("--teacher-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--degraded-model", default="outputs/degraded_model")
    parser.add_argument("--output-dir", default="outputs/recover_lora")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--lora-targets", default="k_proj,v_proj")
    parser.add_argument("--synthetic-samples", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=1234)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-degraded")
    subparsers.add_parser("generate-synthetic")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--synthetic-data", required=True, type=Path)
    train_parser.add_argument("--checkpoint", type=Path)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    train_parser.add_argument("--epochs", type=int, default=3)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = RecoverLoRAConfig(
        teacher_model=args.teacher_model,
        degraded_model=args.degraded_model,
        output_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype,
        lora_targets=tuple(part.strip() for part in args.lora_targets.split(",") if part.strip()),
        synthetic_samples=args.synthetic_samples,
        seed=args.seed,
        batch_size=getattr(args, "batch_size", 1),
        gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 32),
        epochs=getattr(args, "epochs", 3),
    )
    if args.command == "prepare-degraded":
        save_degraded_model(config)
    elif args.command == "generate-synthetic":
        generate_synthetic_data(config, Path(config.output_dir) / "synthetic.jsonl")
    else:
        train(config, args.synthetic_data, args.checkpoint or Path(config.output_dir) / "adapter")


if __name__ == "__main__":
    main()
