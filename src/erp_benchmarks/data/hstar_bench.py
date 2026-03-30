from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import DatasetAdapter
from ..utils.hstar_protocol import build_hstar_protocol_records, extract_hstar_archives
from ..utils.io import dump_json


def _has_local_hstar_raw(raw_dir: Path) -> bool:
    if (raw_dir / "hos_bench.zip").exists() or (raw_dir / "hps_bench.zip").exists():
        return True
    if (raw_dir / "hos_bench").is_dir() or (raw_dir / "hps_bench").is_dir():
        return True
    return any(raw_dir.rglob("annotation.json"))


def _resolve_hstar_source_root(raw_dir: Path, extract_root: Path) -> tuple[Path, dict[str, Path], bool]:
    extracted = extract_hstar_archives(raw_dir, extract_root)
    if extracted:
        return extract_root, extracted, True

    # Some users already store the extracted H*Bench tree instead of the zip files.
    if any(raw_dir.rglob("annotation.json")):
        return raw_dir, {"pre_extracted": raw_dir}, False

    raise FileNotFoundError(
        "H*Bench data not found. Expected either hos_bench.zip/hps_bench.zip inside "
        f"{raw_dir} or an extracted tree containing annotation.json files."
    )


class HstarBenchDataset(DatasetAdapter):
    benchmark_id = "hstar-bench"
    task_type = "search"
    supported_model_generation = False
    repo_id = "humanoid-vstar/hstar_bench"

    def ensure_data(self, data_root: Path) -> None:
        from huggingface_hub import snapshot_download

        target = data_root / self.benchmark_id / "raw"
        target.mkdir(parents=True, exist_ok=True)
        if _has_local_hstar_raw(target):
            return
        snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            local_dir=str(target),
            allow_patterns=["*.zip"],
            local_dir_use_symlinks=False,
        )

    def build_manifest(self, data_root: Path, split: str = "test") -> Path:
        del split
        raw_dir = data_root / self.benchmark_id / "raw"
        extract_root = data_root / self.benchmark_id / "extracted"
        manifests_root = data_root / self.benchmark_id / "manifests"
        manifest_path = manifests_root / "test.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        archives = [
            str(raw_dir / "hos_bench.zip"),
            str(raw_dir / "hps_bench.zip"),
        ]
        source_root, extracted, used_archives = _resolve_hstar_source_root(raw_dir, extract_root)
        protocol_manifests: dict[str, str] = {}
        records_by_protocol = build_hstar_protocol_records(source_root)
        for protocol, rows in records_by_protocol.items():
            path = manifests_root / f"{protocol}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            protocol_manifests[protocol] = str(path)

        payload = {
            "benchmark": self.benchmark_id,
            "note": (
                "Official H* benchmark data is distributed as hos_bench.zip and "
                "hps_bench.zip. The unified workspace aggregates the official HOS/HPS "
                "benchmark entries into a direct ERP target-direction manifest without "
                "adding rotation-expanded copies."
            ),
            "archives": archives if used_archives else [],
            "extracted_dirs": {name: str(path) for name, path in extracted.items()},
            "protocol_manifests": protocol_manifests,
        }
        dump_json(manifest_path, payload)
        return manifest_path

    def evaluate(
        self,
        manifest_path: Path,
        predictions_path: Path,
        report_path: Path,
    ) -> dict[str, Any]:
        del manifest_path, predictions_path
        report = {
            "benchmark": self.benchmark_id,
            "status": "external_predictions_required",
        }
        dump_json(report_path, report)
        return report


class HstarBenchErpDataset(HstarBenchDataset):
    benchmark_id = "hstar-bench-erp"
    supported_model_generation = True

    def ensure_data(self, data_root: Path) -> None:
        from huggingface_hub import snapshot_download

        target = data_root / self.benchmark_id / "raw"
        target.mkdir(parents=True, exist_ok=True)
        if _has_local_hstar_raw(target):
            return
        snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            local_dir=str(target),
            allow_patterns=["*.zip"],
            local_dir_use_symlinks=False,
        )

    def build_manifest(self, data_root: Path, split: str = "test") -> Path:
        split_key = (split or "test").strip().lower()
        raw_dir = data_root / self.benchmark_id / "raw"
        extract_root = data_root / self.benchmark_id / "extracted"
        manifests_root = data_root / self.benchmark_id / "manifests"
        manifests_root.mkdir(parents=True, exist_ok=True)

        source_root, _, _ = _resolve_hstar_source_root(raw_dir, extract_root)
        records_by_protocol = build_hstar_protocol_records(source_root)
        for protocol, rows in records_by_protocol.items():
            path = manifests_root / f"{protocol}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        direct_manifest = manifests_root / "erp_direct_submit.jsonl"
        test_manifest = manifests_root / "test.jsonl"
        if direct_manifest.exists():
            test_manifest.write_text(direct_manifest.read_text(encoding="utf-8"), encoding="utf-8")

        split_to_manifest = {
            "test": manifests_root / "test.jsonl",
            "erp_direct_submit": manifests_root / "erp_direct_submit.jsonl",
        }
        manifest_path = split_to_manifest.get(split_key, manifests_root / f"{split_key}.jsonl")
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"H*Bench manifest was not generated as expected: {manifest_path}"
            )
        return manifest_path
