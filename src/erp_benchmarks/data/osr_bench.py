from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from .base import DatasetAdapter
from ..utils.io import dump_json, load_records
from ..utils.metrics import exact_match_report


class OsrBenchDataset(DatasetAdapter):
    benchmark_id = "osr-bench"
    task_type = "qa"
    repo_id = "UUUserna/OSR-Bench"

    def ensure_data(self, data_root: Path) -> None:
        target = data_root / self.benchmark_id / "raw"
        target.mkdir(parents=True, exist_ok=True)
        marker = target / "qa.csv"
        if marker.exists():
            return
        snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            local_dir=str(target),
            allow_patterns=["README.md", "qa.csv", "qa_with_negative.csv", "data/**"],
        )

    def build_manifest(self, data_root: Path, split: str = "test") -> Path:
        del split
        raw_dir = data_root / self.benchmark_id / "raw"
        manifest_path = data_root / self.benchmark_id / "manifests" / "test.jsonl"
        if manifest_path.exists():
            return manifest_path

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with (raw_dir / "qa.csv").open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = []
            for row in reader:
                image_path = raw_dir / row["image_id"]
                rows.append(
                    {
                        "id": f"{row['image_id']}::{row['turn_id']}",
                        "image_path": str(image_path),
                        "question": row["question"],
                        "prompt": row["question"],
                        "answer": row["answer"],
                        "skills_tested": row.get("skills_tested"),
                    }
                )

        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(f"{__import__('json').dumps(row, ensure_ascii=False)}\n")
        return manifest_path

    def evaluate(
        self,
        manifest_path: Path,
        predictions_path: Path,
        report_path: Path,
    ) -> dict[str, Any]:
        refs = {row["id"]: row["answer"] for row in load_records(manifest_path)}
        preds = {row["id"]: row["prediction"] for row in load_records(predictions_path)}
        report = exact_match_report(refs, preds)
        report["benchmark"] = self.benchmark_id
        dump_json(report_path, report)
        return report
