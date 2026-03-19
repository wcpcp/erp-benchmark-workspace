from __future__ import annotations

from typing import Any

from datasets import load_dataset

from .base import BenchmarkAdapter
from ..utils.io import dump_json, load_records


def _load_hf_records(split: str, cache_dir: str | None) -> list[dict[str, Any]]:
    dataset = load_dataset("pangyyyyy/OmniSpatial", split=split, cache_dir=cache_dir)
    records = []
    for row_index, row in enumerate(dataset):
        item = dict(row)
        raw_id = str(item["id"])
        item["raw_id"] = raw_id
        item["row_index"] = row_index
        item["id"] = f"{row_index}::{raw_id}"
        records.append(item)
    return records


def _normalize_prediction(prediction: Any, options: list[str], answer_letter: str = "") -> str | int | None:
    if prediction is None:
        return None
    text = str(prediction).strip()
    if not text:
        return None
    lower = text.lower()
    for idx, option in enumerate(options):
        if lower == str(option).strip().lower():
            return idx
    if text.isdigit():
        return int(text)
    upper = text.upper()
    if len(upper) == 1 and "A" <= upper <= "Z":
        if options:
            return ord(upper) - ord("A")
        return upper
    if text.startswith(("A", "B", "C", "D")) and ":" in text:
        prefix = text.split(":", 1)[0].strip().upper()
        if len(prefix) == 1:
            if options:
                return ord(prefix) - ord("A")
            return prefix
    if not options and answer_letter and lower == answer_letter.lower():
        return answer_letter
    return None


class OmniSpatialBenchmark(BenchmarkAdapter):
    dataset_id = "pangyyyyy/OmniSpatial"

    def evaluate(self, args: Any) -> dict[str, Any]:
        if args.dataset_source == "hf":
            reference_records = _load_hf_records(args.split, args.cache_dir)
        elif args.references:
            reference_records = load_records(args.references)
        else:
            raise ValueError("Provide --references or use --dataset-source hf")

        prediction_records = load_records(args.predictions)
        pred_map = {
            str(row.get("id")): row.get("prediction")
            for row in prediction_records
            if row.get("id") is not None
        }
        compared = 0
        correct = 0
        missing = []
        errors = []

        for row in reference_records:
            sample_id = str(row["id"])
            raw_id = str(row.get("raw_id", "")).strip()

            prediction = pred_map.get(sample_id)
            if prediction is None and raw_id:
                prediction = pred_map.get(raw_id)

            if prediction is None:
                missing.append(sample_id)
                continue

            options = list(row["options"])
            if "answer_index" in row:
                answer_index = int(row["answer_index"])
            else:
                answer_index = int(row["answer"])
            answer_letter = str(row.get("gt", row.get("answer_letter", ""))).strip().upper()
            pred_index = _normalize_prediction(prediction, options, answer_letter)
            compared += 1

            is_correct = pred_index == answer_index if options else pred_index == answer_letter
            if is_correct:
                correct += 1
            elif len(errors) < 10:
                errors.append(
                    {
                        "id": sample_id,
                        "raw_id": raw_id or None,
                        "prediction": prediction,
                        "answer_index": answer_index,
                        "answer_text": options[answer_index] if options and 0 <= answer_index < len(options) else answer_letter,
                        "options": options,
                    }
                )

        report = {
            "benchmark": "omnispatial",
            "split": args.split,
            "num_references": len(reference_records),
            "num_predictions": len(prediction_records),
            "num_compared": compared,
            "num_correct": correct,
            "accuracy": correct / compared if compared else 0.0,
            "coverage": compared / len(reference_records) if reference_records else 0.0,
            "missing_prediction_ids": missing[:10],
            "example_errors": errors,
        }
        if args.report:
            dump_json(args.report, report)
        return report
