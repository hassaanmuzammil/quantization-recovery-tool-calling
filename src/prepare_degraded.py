from __future__ import annotations

import argparse
from .recover_config import RecoverLoRAConfig
from .recover_lora import save_degraded_model


def main() -> None:
    # Defaults live in RecoverLoRAConfig; options only override when passed.
    parser = argparse.ArgumentParser(
        description="Create a degraded Recover-LoRA student checkpoint",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument("--teacher-model", dest="teacher_model", required=True)
    parser.add_argument("--output", dest="degraded_model")
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--noise-std", dest="perturbation_std", type=float)
    parser.add_argument(
        "--method",
        dest="degradation_method",
        choices=("random-perturbation", "gate-up-quantization"),
        help="How to degrade the teacher checkpoint",
    )
    parser.add_argument("--group-size", dest="quantization_group_size", type=int)
    parser.add_argument("--bits", dest="quantization_bits", type=int, choices=(2, 4))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = RecoverLoRAConfig(**vars(args))
    save_degraded_model(config)
    print(f"Saved degraded model to {config.degraded_model}")


if __name__ == "__main__":
    main()
