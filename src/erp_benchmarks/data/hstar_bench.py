from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from .base import DatasetAdapter
from ..utils.hstar_protocol import build_hstar_protocol_records, extract_hstar_archives
from ..utils.io import dump_json


class HstarBenchDataset(DatasetAdapter):
    benchmark_id = "hstar-bench"
    task_type = "search"
    supported_model_generation = False
    repo_id = "humanoid-vstar/hstar_bench"

    def ensure_data(self, data_root: Path) -> None:
        target = data_root / self.benchmark_id / "raw"
        target.mkdir(parents=True, exist_ok=True)
        marker = target / "hos_bench.zip"
        if marker.exists():
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
        extracted = extract_hstar_archives(raw_dir, extract_root)
        protocol_manifests: dict[str, str] = {}
        if extracted:
            records_by_protocol = build_hstar_protocol_records(extract_root)
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
                "hps_bench.zip. The unified workspace now tracks both the original "
                "perspective multi-turn protocol and rotated-ERP submit manifests."
            ),
            "archives": archives,
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
        target = data_root / self.benchmark_id / "raw"
        target.mkdir(parents=True, exist_ok=True)
        marker = target / "hos_bench.zip"
        if marker.exists():
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

        extracted = extract_hstar_archives(raw_dir, extract_root)
        if extracted:
            records_by_protocol = build_hstar_protocol_records(extract_root)
            for protocol, rows in records_by_protocol.items():
                path = manifests_root / f"{protocol}.jsonl"
                with path.open("w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        split_to_manifest = {
            "test": manifests_root / "erp_rotated_submit.jsonl",
            "erp_rotated_submit": manifests_root / "erp_rotated_submit.jsonl",
            "perspective_multiturn": manifests_root / "perspective_multiturn.jsonl",
        }
        return split_to_manifest.get(split_key, manifests_root / f"{split_key}.jsonl")
