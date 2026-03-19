#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download benchmark datasets from Hugging Face."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["osr-bench", "panoenv", "omnispatial", "hstar-bench"],
        help="Benchmark identifier in registry.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Target directory. Defaults to benchmark/_third_party/hf/<dataset>.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Optional override for the Hugging Face dataset id.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Download the full dataset snapshot. By default only metadata files are fetched.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Dataset revision to download.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        parser.error(
            "huggingface_hub is required. Install benchmark/requirements.txt first."
        )
        raise exc

    repo_map = {
        "osr-bench": "UUUserna/OSR-Bench",
        "panoenv": "7zkk/PanoEnv",
        "omnispatial": "pangyyyyy/OmniSpatial",
        "hstar-bench": "humanoid-vstar/hstar_bench",
    }
    repo_id = args.repo_id or repo_map[args.dataset]

    root = Path(__file__).resolve().parents[1]
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "_third_party" / "hf" / args.dataset
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.full:
        allow_patterns = None
    elif args.dataset == "omnispatial":
        allow_patterns = ["README.md", "*.parquet"]
    elif args.dataset == "hstar-bench":
        allow_patterns = ["*.zip"]
    else:
        allow_patterns = ["README.md", "*.json", "*.jsonl", "dataset_infos.json"]

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )

    print(f"Downloaded {repo_id} into {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
