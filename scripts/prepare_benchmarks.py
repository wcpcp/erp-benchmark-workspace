#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from erp_benchmarks.data import BENCHMARK_DATASETS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and prepare benchmark data.")
    parser.add_argument(
        "--benchmarks",
        default="osr-bench,panoenv,360loc",
        help="Comma-separated benchmark ids or 'all'.",
    )
    parser.add_argument(
        "--data-root",
        default=str(ROOT / "data"),
        help="Root directory for benchmark data.",
    )
    parser.add_argument(
        "--skip-360loc-download",
        action="store_true",
        help="Skip direct dataset download for 360Loc and only prepare directory scaffolding.",
    )
    return parser


def resolve_benchmarks(raw: str) -> list[str]:
    if raw == "all":
        items = list(BENCHMARK_DATASETS.keys())
    else:
        items = [item.strip() for item in raw.split(",") if item.strip()]
    deduped = []
    seen = set()
    for item in items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def download_360loc(data_root: Path) -> None:
    import shutil
    import zipfile

    import requests

    scene_urls = {
        "atrium": "https://hkustvgd.com/statics/360loc/atrium.zip",
        "concourse": "https://hkustvgd.com/statics/360loc/concourse.zip",
        "hall": "https://hkustvgd.com/statics/360loc/hall.zip",
        "piatrium": "https://hkustvgd.com/statics/360loc/piatrium.zip",
    }

    raw_dir = data_root / "360loc" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for scene, url in scene_urls.items():
        scene_dir = raw_dir / scene
        zip_path = raw_dir / f"{scene}.zip"
        if scene_dir.exists():
            print(f"[skip] 360Loc scene already exists: {scene}")
            continue
        if zip_path.exists() and not zipfile.is_zipfile(zip_path):
            print(f"[warn] removing invalid 360Loc archive: {zip_path}")
            zip_path.unlink()
        if not zip_path.exists():
            print(f"[download] 360Loc scene: {scene}")
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "text/html" in content_type.lower():
                    raise RuntimeError(
                        "360Loc official site returned HTML instead of a zip archive."
                    )
                with zip_path.open("wb") as handle:
                    shutil.copyfileobj(response.raw, handle)
        if not zipfile.is_zipfile(zip_path):
            raise RuntimeError(f"Downloaded file is not a valid zip archive: {zip_path}")
        print(f"[extract] 360Loc scene: {scene}")
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(raw_dir)


def main() -> int:
    args = build_parser().parse_args()
    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    for benchmark_id in resolve_benchmarks(args.benchmarks):
        adapter = BENCHMARK_DATASETS[benchmark_id]
        print(f"==> {benchmark_id}")
        adapter.ensure_data(data_root)
        if benchmark_id == "360loc" and not args.skip_360loc_download:
            download_360loc(data_root)
        manifest = adapter.build_manifest(data_root, split="test")
        print(f"[ready] manifest: {manifest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
