from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import DatasetAdapter
from ..utils.io import dump_json


class Loc360Dataset(DatasetAdapter):
    benchmark_id = "360loc"
    task_type = "localization"
    supported_model_generation = False

    def ensure_data(self, data_root: Path) -> None:
        target = data_root / self.benchmark_id
        target.mkdir(parents=True, exist_ok=True)

    def build_manifest(self, data_root: Path, split: str = "test") -> Path:
        manifest_path = data_root / self.benchmark_id / "manifests" / f"{split}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if not manifest_path.exists():
            dump_json(
                manifest_path,
                {
                    "benchmark": self.benchmark_id,
                    "note": "Use the official 360Loc dataset plus your own localization predictions. Unified generation from VLM is not implemented.",
                },
            )
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
