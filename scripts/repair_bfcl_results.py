"""Repair the formatting of BFCL FC-mode outputs, as a diagnostic.

For each output, the first JSON object with a string ``name`` and a dict
``arguments`` is rewrapped as a single, correctly closed call:

    <tool_call>
    {"name": ..., "arguments": {...}}
    </tool_call>

The function name and arguments are never changed, and any extra keys
next to them are dropped.  Outputs without such an object are left as is.
This separates format damage (extra ``}``, missing ``</tool_call>``,
markdown JSON blocks, repetition after the call) from content damage.  The
repaired score is a diagnostic, not an official BFCL result.

Usage:
    python scripts/repair_bfcl_results.py \
        --result-dir benchmarks/gorilla/berkeley-function-call-leaderboard/result/qwen_q2_xlam_sft \
        --output-dir benchmarks/gorilla/berkeley-function-call-leaderboard/result/qwen_q2_xlam_sft_repaired

then run ``bfcl evaluate`` with ``--result-dir`` set to the output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def repair(text: str) -> str:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str) and isinstance(obj.get("arguments"), dict):
            call = {"name": obj["name"], "arguments": obj["arguments"]}
            return "<tool_call>\n" + json.dumps(call, ensure_ascii=False) + "\n</tool_call>"
    return text


def repair_file(source: Path, destination: Path) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = changed = 0
    with source.open(encoding="utf-8") as reader, destination.open("w", encoding="utf-8") as writer:
        for line in reader:
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record.get("result"), str):
                repaired = repair(record["result"])
                changed += repaired != record["result"]
                record["result"] = repaired
            total += 1
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
    return total, changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--result-dir", required=True, type=Path, help="BFCL result directory of one run")
    parser.add_argument("--output-dir", required=True, type=Path, help="Where to write the repaired copy")
    args = parser.parse_args()

    sources = sorted(args.result_dir.rglob("*_result.json"))
    if not sources:
        raise SystemExit(f"No *_result.json files under {args.result_dir}")
    for source in sources:
        destination = args.output_dir / source.relative_to(args.result_dir)
        total, changed = repair_file(source, destination)
        print(f"{source.relative_to(args.result_dir)}: repaired {changed} of {total} outputs")


if __name__ == "__main__":
    main()
