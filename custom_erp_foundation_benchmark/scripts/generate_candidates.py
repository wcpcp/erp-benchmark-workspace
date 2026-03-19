#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from erp_foundation_benchmark_builder import generate_scene_candidates, load_scene_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ERP benchmark candidates from one scene metadata file.")
    parser.add_argument("--input", required=True, help="Path to one scene metadata JSON file.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--core-only", action="store_true", help="Exclude advanced extension tasks.")
    args = parser.parse_args()

    scene = load_scene_metadata(args.input)
    rows = generate_scene_candidates(scene, include_extension=not args.core_only)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"scene_id": scene.scene_id, "num_candidates": len(rows), "output": str(output_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
