from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import DatasetAdapter
from ..utils.io import dump_json, load_records


class OmniSpatialDataset(DatasetAdapter):
    benchmark_id = "omnispatial"
    task_type = "qa"
    repo_id = "pangyyyyy/OmniSpatial"

    def ensure_data(self, data_root: Path) -> None:
        from huggingface_hub import snapshot_download

        target = data_root / self.benchmark_id / "raw"
        target.mkdir(parents=True, exist_ok=True)
        marker = target / "test-00000-of-00001.parquet"
        if marker.exists():
            return
        snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            local_dir=str(target),
            allow_patterns=["README.md", "test-00000-of-00001.parquet", "image_files/**"],
            local_dir_use_symlinks=False,
        )

    def build_manifest(self, data_root: Path, split: str = "test") -> Path:
        from datasets import load_dataset

        del split
        raw_dir = data_root / self.benchmark_id / "raw"
        manifest_path = data_root / self.benchmark_id / "manifests" / "test.jsonl"
        if manifest_path.exists():
            return manifest_path

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset(
            "parquet",
            data_files=str(raw_dir / "test-00000-of-00001.parquet"),
            split="train",
        )
        with manifest_path.open("w", encoding="utf-8") as handle:
            for row_index, row in enumerate(dataset):
                options = list(row["options"])
                answer_index = int(row["answer"])
                answer_letter = str(row.get("gt", "")).strip().upper()
                answer_text = options[answer_index] if 0 <= answer_index < len(options) else answer_letter
                raw_id = str(row["id"])
                sample_id = f"{row_index}::{raw_id}"
                payload = {
                    "id": sample_id,
                    "raw_id": raw_id,
                    "row_index": row_index,
                    "image_path": str(raw_dir / row["image_path"]),
                    "question": row["question"],
                    "prompt": row["question"],
                    "options": options,
                    "answer": answer_text,
                    "answer_index": answer_index,
                    "answer_letter": answer_letter,
                    "task_type": row.get("task_type"),
                    "sub_task_type": row.get("sub_task_type"),
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return manifest_path

    def evaluate(
        self,
        manifest_path: Path,
        predictions_path: Path,
        report_path: Path,
    ) -> dict[str, Any]:
        references = load_records(manifest_path)
        predictions = load_records(predictions_path)
        pred_map = {str(row.get("id")): row for row in predictions}
        compared = 0
        correct = 0
        missing = []
        errors = []

        for row in references:
            sample_id = str(row["id"])
            pred_row = pred_map.get(sample_id)
            if pred_row is None:
                missing.append(sample_id)
                continue
            compared += 1
            prediction = pred_row.get("prediction")
            if _prediction_is_correct(
                prediction,
                row["options"],
                int(row["answer_index"]),
                str(row.get("answer_letter", "")).strip().upper(),
            ):
                correct += 1
            elif len(errors) < 10:
                errors.append(
                    {
                        "id": sample_id,
                        "raw_id": row.get("raw_id"),
                        "prediction": prediction,
                        "answer_index": row["answer_index"],
                        "answer_text": row["answer"],
                        "options": row["options"],
                    }
                )

        report = {
            "benchmark": self.benchmark_id,
            "num_references": len(references),
            "num_predictions": len(predictions),
            "num_compared": compared,
            "num_correct": correct,
            "accuracy": correct / compared if compared else 0.0,
            "coverage": compared / len(references) if references else 0.0,
            "missing_prediction_ids": missing[:10],
            "example_errors": errors,
        }
        dump_json(report_path, report)
        return report


def _prediction_is_correct(prediction: Any, options: list[str], answer_index: int, answer_letter: str = "") -> bool:
    if prediction is None:
        return False
    text = str(prediction).strip()
    if not text:
        return False

    if options and 0 <= answer_index < len(options):
        answer_text = options[answer_index].strip().lower()
        if text.lower() == answer_text:
            return True

    if text.isdigit():
        return int(text) == answer_index

    upper = text.upper()
    if len(upper) == 1 and "A" <= upper <= "Z":
        if options:
            return (ord(upper) - ord("A")) == answer_index
        return upper == answer_letter

    if not options and answer_letter and text.strip().upper() == answer_letter:
        return True

    if text.startswith(("A", "B", "C", "D")) and ":" in text:
        prefix = text.split(":", 1)[0].strip().upper()
        if len(prefix) == 1 and "A" <= prefix <= "D":
            if options:
                return (ord(prefix) - ord("A")) == answer_index
            return prefix == answer_letter

    return False
