#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a local benchmark model.")
    parser.add_argument(
        "--model-id",
        default="mlx-community/Qwen3-VL-4B-Instruct-4bit",
        help="Model repo id to download.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Local target directory. Defaults to benchmark/models/<repo-name>.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (root / "models" / args.model_id.split("/")[-1]).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.model_id,
        local_dir=str(output_dir),
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.model",
            "*.tiktoken",
            "*.txt",
            "*.py",
            "*.png",
            "*.md",
        ],
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
