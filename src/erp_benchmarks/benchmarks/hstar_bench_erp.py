from __future__ import annotations

from typing import Any

from .base import BenchmarkAdapter
from ..utils.hstar_protocol import (
    parse_action,
    pitch_distance_to_range,
    pitch_in_range,
    yaw_distance_to_range,
    yaw_in_range,
)
from ..utils.io import dump_json, load_records


def _load_prediction_map(path: str) -> dict[str, Any]:
    rows = load_records(path)
    return {str(row.get("id")): row.get("prediction") for row in rows if row.get("id") is not None}


class HstarBenchErpBenchmark(BenchmarkAdapter):
    def evaluate(self, args: Any) -> dict[str, Any]:
        if not args.references:
            raise ValueError("hstar-bench-erp requires --references with a protocol manifest.")

        references = load_records(args.references)
        predictions = _load_prediction_map(args.predictions)

        compared = 0
        correct = 0
        exact_match = 0
        yaw_errors: list[float] = []
        pitch_errors: list[float] = []
        missing = []
        errors = []

        for row in references:
            sample_id = str(row["id"])
            raw_prediction = predictions.get(sample_id)
            if raw_prediction is None:
                missing.append(sample_id)
                continue

            compared += 1
            gold = str(row.get("answer", "")).strip()
            if str(raw_prediction).strip() == gold:
                exact_match += 1

            parsed = parse_action(raw_prediction)
            if parsed is None:
                if len(errors) < 10:
                    errors.append({"id": sample_id, "prediction": raw_prediction, "answer": gold, "reason": "parse_failed"})
                continue

            target_yaw = list(row["target_yaw"])
            target_pitch = list(row["target_pitch"])
            if parsed.name != "submit":
                if len(errors) < 10:
                    errors.append(
                        {
                            "id": sample_id,
                            "prediction": raw_prediction,
                            "answer": gold,
                            "reason": "submit_required",
                        }
                    )
                continue

            pred_yaw = parsed.yaw
            pred_pitch = parsed.pitch
            yaw_error = yaw_distance_to_range(pred_yaw, target_yaw)
            pitch_error = pitch_distance_to_range(pred_pitch, target_pitch)
            success = yaw_in_range(pred_yaw, target_yaw) and pitch_in_range(pred_pitch, target_pitch)

            yaw_errors.append(float(yaw_error))
            pitch_errors.append(float(pitch_error))
            if success:
                correct += 1
            elif len(errors) < 10:
                errors.append(
                    {
                        "id": sample_id,
                        "prediction": raw_prediction,
                        "answer": gold,
                        "variant": row.get("task_variant", "rotated_submit"),
                        "yaw_error": yaw_error,
                        "pitch_error": pitch_error,
                    }
                )

        report = {
            "benchmark": "hstar-bench-erp",
            "num_references": len(references),
            "num_predictions": len(predictions),
            "num_compared": compared,
            "num_correct": correct,
            "success_rate": correct / compared if compared else 0.0,
            "exact_match": exact_match / compared if compared else 0.0,
            "mean_abs_yaw_error": sum(yaw_errors) / len(yaw_errors) if yaw_errors else 0.0,
            "mean_abs_pitch_error": sum(pitch_errors) / len(pitch_errors) if pitch_errors else 0.0,
            "coverage": compared / len(references) if references else 0.0,
            "missing_prediction_ids": missing[:10],
            "example_errors": errors,
            "note": (
                "This evaluates rotated-ERP H*Bench submit-only variants. "
                "Each ERP input is horizontally rotated to match the sampled initial yaw, "
                "and the gold target window is rotated into the same ERP image coordinate system."
            ),
        }
        if args.report:
            dump_json(args.report, report)
        return report
