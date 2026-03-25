from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import DatasetAdapter
from ..utils.io import dump_json, load_records
from ..utils.metrics import exact_match_report


class PanoEnvDataset(DatasetAdapter):
    benchmark_id = "panoenv"
    task_type = "qa"
    repo_id = "7zkk/PanoEnv"

    def ensure_data(self, data_root: Path) -> None:
        from huggingface_hub import snapshot_download

        target = data_root / self.benchmark_id / "raw"
        target.mkdir(parents=True, exist_ok=True)
        marker = target / "test"
        if marker.exists():
            return
        snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            local_dir=str(target),
            allow_patterns=["README.md", "test/**"],
        )

    def build_manifest(self, data_root: Path, split: str = "test") -> Path:
        raw_dir = data_root / self.benchmark_id / "raw"
        manifest_path = data_root / self.benchmark_id / "manifests" / f"{split}.jsonl"
        if manifest_path.exists():
            return manifest_path

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for qa_file in sorted((raw_dir / split).glob("*/*/*_qa.json")):
            with qa_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            for question in payload["questions"]:
                qid = int(question["question_id"])
                pattern = f"q{qid}_*.png"
                matches = sorted((qa_file.parent / "visualizations").glob(pattern))
                image_path = matches[0] if matches else None
                rows.append(
                    {
                        "id": f"{payload['env']}::{payload['image_id']}::{qid}",
                        "image_path": str(image_path) if image_path else "",
                        "question": question["question"],
                        "prompt": question["question"],
                        "answer": question["answer"],
                        "major_category": question.get("major_category"),
                        "sub_category": question.get("sub_category"),
                        "question_type": question.get("question_type"),
                        "input_note": "Public release exposes question-specific visualization PNGs rather than raw ERP panoramas.",
                    }
                )

        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(f"{json.dumps(row, ensure_ascii=False)}\n")
        return manifest_path

    def evaluate(
        self,
        manifest_path: Path,
        predictions_path: Path,
        report_path: Path,
    ) -> dict[str, Any]:
        references = load_records(manifest_path)
        predictions = load_records(predictions_path)
        refs = {row["id"]: row["answer"] for row in references}
        preds = {row["id"]: row["prediction"] for row in predictions}
        report = exact_match_report(refs, preds)

        per_subcategory: dict[str, dict[str, Any]] = {}
        pred_map = {row["id"]: row["prediction"] for row in predictions}
        for row in references:
            sub = row.get("sub_category") or "unknown"
            bucket = per_subcategory.setdefault(sub, {"correct": 0, "total": 0})
            if row["id"] in pred_map:
                bucket["total"] += 1
                if str(pred_map[row["id"]]).strip().lower() == str(row["answer"]).strip().lower():
                    bucket["correct"] += 1

        report["per_subcategory"] = {
            key: {
                "accuracy": value["correct"] / value["total"] if value["total"] else 0.0,
                "num_samples": value["total"],
            }
            for key, value in sorted(per_subcategory.items())
        }
        report["benchmark"] = self.benchmark_id
        dump_json(report_path, report)
        return report
