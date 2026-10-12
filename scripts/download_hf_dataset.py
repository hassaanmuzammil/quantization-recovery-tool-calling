from datasets import load_dataset
import json

# dataset = load_dataset(
#     "HuggingFaceFW/fineweb-edu",
#     name="sample-10BT",
#     split="train",
#     streaming=True,
# )

dataset = load_dataset(
    "allenai/c4",
    "en",
    split="train",
    streaming=True,
)

with open("datasets/allenai_c4.jsonl", "w", encoding="utf-8") as f:
    for i, example in enumerate(dataset):
        if i >= 100_000:
            break

        f.write(
            json.dumps(
                {"text": example["text"]},
                ensure_ascii=False
            ) + "\n"
        )

        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/100000")
