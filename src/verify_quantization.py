import argparse
import torch
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--bits", type=int, choices=[2, 4], required=True)
    parser.add_argument("--group-size", type=int, default=32)
    args = parser.parse_args()

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    max_levels = 2 ** args.bits
    gate_count = 0
    up_count = 0
    other_count = 0
    failures = []

    print("\nChecking weights...\n")

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(("gate_proj.weight", "up_proj.weight")):
                if name.endswith("gate_proj.weight"):
                    gate_count += 1
                else:
                    up_count += 1

                weight = param.detach()
                rows, cols = weight.shape

                max_unique = 0
                invalid_groups = 0
                total_groups = 0

                for start in range(0, cols, args.group_size):
                    end = min(start + args.group_size, cols)

                    # Each row has an independent quantization group
                    group = weight[:, start:end]

                    for row in group:
                        unique = torch.unique(row).numel()
                        max_unique = max(max_unique, unique)
                        total_groups += 1

                        if unique > max_levels:
                            invalid_groups += 1

                print(
                    f"{name}: "
                    f"dtype={weight.dtype}, "
                    f"max_unique={max_unique}, "
                    f"invalid_groups={invalid_groups}/{total_groups}"
                )

                if invalid_groups:
                    failures.append(name)

            else:
                other_count += 1
                if param.dtype != torch.bfloat16:
                    failures.append(f"Non-BF16: {name}")

    print("\n========== SUMMARY ==========")
    print(f"Bits: {args.bits}")
    print(f"Group size: {args.group_size}")
    print(f"Gate projections: {gate_count}")
    print(f"Up projections: {up_count}")
    print(f"Other BF16 parameters: {other_count}")
    print(f"Failures: {len(failures)}")

    if failures:
        print("\nFAIL: Unexpected quantization levels or dtypes.")
        print("First failures:", failures[:10])
    else:
        print("\nPASS: Weight structure is consistent with "
              f"{args.bits}-bit Gate/Up fake quantization "
              "and BF16 elsewhere.")


if __name__ == "__main__":
    main()
