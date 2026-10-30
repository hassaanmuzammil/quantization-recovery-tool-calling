"""CLI and backwards-compatible public imports for the Recover-LoRA pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, fields
from pathlib import Path

from .adapters import LoRALinear, add_lora_adapters
from .distillation import accuracy_recovery_percentage, distillation_loss
from .gate_up_quantization import quantize_gate_up_projections
from .model_utils import load_model, load_tokenizer, set_seed
from .random_perturbation import perturb_kv_projections
from .recover_config import RecoverLoRAConfig
from .synthetic_data import generate_hybrid_sample, generate_synthetic_data
from .training import _resolve_lora_targets, train


def save_degraded_model(config: RecoverLoRAConfig) -> None:
    model = load_model(config.teacher_model, config.device, config.dtype)
    if config.degradation_method == "random-perturbation":
        perturb_kv_projections(model, config.perturbation_std, config.seed)
    elif config.degradation_method == "gate-up-quantization":
        quantize_gate_up_projections(
            model,
            bits=config.quantization_bits,
            group_size=config.quantization_group_size,
        )
    else:
        raise ValueError(f"Unsupported degradation method: {config.degradation_method}")
    destination = Path(config.degraded_model)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)

    # Do not reserialize the tokenizer with the training environment's
    # Transformers version.  Newer versions can write Qwen's
    # ``extra_special_tokens`` as a list, while BFCL's environment expects the
    # older mapping format and fails with ``list has no attribute keys``.
    # Copy the tokenizer artifacts from the teacher checkpoint verbatim so the
    # degraded checkpoint stays compatible with both environments.
    teacher_path = Path(config.teacher_model)
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    )
    copied_tokenizer_files = 0
    for filename in tokenizer_files:
        source = teacher_path / filename
        target = destination / filename
        if source.is_file():
            shutil.copy2(source, target)
            copied_tokenizer_files += 1
    if copied_tokenizer_files == 0:
        # Hub model IDs and snapshot paths may not expose files directly as a
        # normal Path.  Fall back to Transformers' resolver in that case.
        load_tokenizer(config.teacher_model).save_pretrained(destination)
        tokenizer_config = destination / "tokenizer_config.json"
        if tokenizer_config.is_file():
            tokenizer_data = json.loads(tokenizer_config.read_text(encoding="utf-8"))
            if isinstance(tokenizer_data.get("extra_special_tokens"), list):
                tokenizer_data.pop("extra_special_tokens")
                tokenizer_config.write_text(
                    json.dumps(tokenizer_data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

    (destination / "degradation_config.json").write_text(
        json.dumps(
            {
                "method": config.degradation_method,
                "bits": config.quantization_bits,
                "group_size": config.quantization_group_size,
                "noise_std": config.perturbation_std,
                "seed": config.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _lora_targets(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parser() -> argparse.ArgumentParser:
    # Defaults live in RecoverLoRAConfig.  Every option that maps to a config
    # field uses ``argparse.SUPPRESS`` so it only overrides the config when it
    # is passed explicitly, and its ``dest`` matches the config field name.
    parser = argparse.ArgumentParser(description="Recover-LoRA pipeline", argument_default=argparse.SUPPRESS)
    parser.add_argument("--teacher-model", dest="teacher_model")
    parser.add_argument("--degraded-model", dest="degraded_model")
    parser.add_argument("--output-dir", dest="output_dir")
    parser.add_argument(
        "--log-dir",
        dest="log_dir",
        help="TensorBoard log directory (defaults to <output-dir>/logs)",
    )
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    parser.add_argument(
        "--lora-targets",
        dest="lora_targets",
        type=_lora_targets,
        help="Override adapter targets (otherwise inferred from degraded checkpoint metadata)",
    )
    parser.add_argument("--synthetic-samples", dest="synthetic_samples", type=int)
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--max-sequence-length", dest="max_sequence_length", type=int)
    parser.add_argument("--seed", type=int)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare-degraded", argument_default=argparse.SUPPRESS)
    prepare_parser.add_argument(
        "--method",
        dest="degradation_method",
        choices=("random-perturbation", "gate-up-quantization"),
        help="How to degrade the teacher checkpoint",
    )
    prepare_parser.add_argument("--noise-std", dest="perturbation_std", type=float)
    prepare_parser.add_argument("--group-size", dest="quantization_group_size", type=int)
    prepare_parser.add_argument("--bits", dest="quantization_bits", type=int, choices=(2, 4))
    generate_parser = subparsers.add_parser("generate-synthetic", argument_default=argparse.SUPPRESS)
    generate_parser.add_argument("--greedy-tokens", dest="greedy_tokens", type=int)
    generate_parser.add_argument("--temperature", type=float)
    generate_parser.add_argument("--top-p", dest="top_p", type=float)
    train_parser = subparsers.add_parser("train", argument_default=argparse.SUPPRESS)
    train_parser.add_argument("--synthetic-data", required=True, type=Path)
    train_parser.add_argument("--checkpoint", type=Path, default=None)
    train_parser.add_argument(
        "--log-dir",
        dest="log_dir",
        help="TensorBoard log directory (defaults to <output-dir>/logs)",
    )
    train_parser.add_argument("--lora-rank", dest="lora_rank", type=int)
    train_parser.add_argument("--lora-alpha", dest="lora_alpha", type=int)
    train_parser.add_argument("--learning-rate", dest="learning_rate", type=float)
    train_parser.add_argument("--warmup-steps", dest="warmup_steps", type=int)
    train_parser.add_argument("--gradient-accumulation-steps", dest="gradient_accumulation_steps", type=int)
    train_parser.add_argument("--validation-split", dest="validation_split", type=float)
    train_parser.add_argument("--validation-interval-steps", dest="validation_interval_steps", type=int)
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument(
        "--save-every-steps",
        dest="save_every_steps",
        type=int,
        help="Save a resumable checkpoint every N optimizer steps (0 disables periodic saves)",
    )
    train_parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        type=lambda value: value.lower() in {"1", "true", "yes", "on"},
        help="Resume automatically from checkpoint/_latest (also accepts --resume=True)",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    config_fields = {field.name for field in fields(RecoverLoRAConfig)}
    config = RecoverLoRAConfig(**{key: value for key, value in vars(args).items() if key in config_fields})
    if args.command == "train":
        config = _resolve_lora_targets(config)
    print(
        "\n========== EFFECTIVE TRAINING CONFIG ==========\n"
        + json.dumps(
            asdict(config),
            indent=4,
            sort_keys=True,
            default=str,
        )
        + "\n===============================================",
        flush=True,
    )
    if args.command == "prepare-degraded":
        save_degraded_model(config)
    elif args.command == "generate-synthetic":
        generate_synthetic_data(config, Path(config.output_dir) / "synthetic.jsonl")
    else:
        train(config, args.synthetic_data, args.checkpoint or Path(config.output_dir) / "adapter")


if __name__ == "__main__":
    main()
