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

from erp_benchmarks.data.registry import get_dataset_adapter


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract H*Bench archives and build unified protocol manifests.")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download official H*Bench archives from Hugging Face before building manifests.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    adapter = get_dataset_adapter("hstar-bench")
    if args.download:
        adapter.ensure_data(data_root)
    manifest = adapter.build_manifest(data_root, split="test")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
