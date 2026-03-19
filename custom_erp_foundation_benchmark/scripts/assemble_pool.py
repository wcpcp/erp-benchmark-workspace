#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from erp_foundation_benchmark_builder.pool import assemble_benchmark_pool


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble a benchmark pool from candidate JSONL files.")
    parser.add_argument("--input-dir", required=True, help="Directory containing candidate JSONL files.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--phase", default="core_v1", choices=["core_v1", "advanced_extension", "all"])
    parser.add_argument("--target-per-task", type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    candidates = []
    for path in sorted(input_dir.rglob("*.jsonl")):
        candidates.extend(load_jsonl(path))

    selected = assemble_benchmark_pool(candidates, phase=args.phase, target_per_task=args.target_per_task)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = Counter(item["task_id"] for item in selected)
    print(json.dumps({"num_selected": len(selected), "phase": args.phase, "per_task": dict(summary)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
