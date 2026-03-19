from __future__ import annotations

import math
from statistics import median
from typing import Any

from .io import normalize_text


def exact_match_report(
    references: dict[str, Any], predictions: dict[str, Any], max_examples: int = 10
) -> dict[str, Any]:
    compared = 0
    correct = 0
    missing = []
    errors = []

    for sample_id, answer in references.items():
        if sample_id not in predictions:
            missing.append(sample_id)
            continue
        compared += 1
        pred = predictions[sample_id]
        if normalize_text(pred) == normalize_text(answer):
            correct += 1
        elif len(errors) < max_examples:
            errors.append(
                {
                    "id": sample_id,
                    "prediction": pred,
                    "answer": answer,
                }
            )

    accuracy = correct / compared if compared else 0.0
    return {
        "num_references": len(references),
        "num_predictions": len(predictions),
        "num_compared": compared,
        "num_correct": correct,
        "accuracy": accuracy,
        "coverage": compared / len(references) if references else 0.0,
        "missing_prediction_ids": missing[:max_examples],
        "example_errors": errors,
    }


def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def localization_report(
    pairs: list[dict[str, Any]],
    thresholds: list[float],
    coordinate_system: str,
) -> dict[str, Any]:
    errors = []

    for pair in pairs:
        gt = pair["ground_truth"]
        pred = pair["prediction"]
        if coordinate_system == "geographic":
            error = _haversine_meters(gt[0], gt[1], pred[0], pred[1])
        else:
            error = _euclidean_distance(gt, pred)
        errors.append(error)

    thresholds = sorted(thresholds)
    success_rates = {
        f"success@{threshold:g}": sum(err <= threshold for err in errors) / len(errors)
        if errors
        else 0.0
        for threshold in thresholds
    }

    return {
        "num_samples": len(errors),
        "mean_error": sum(errors) / len(errors) if errors else 0.0,
        "median_error": median(errors) if errors else 0.0,
        "max_error": max(errors) if errors else 0.0,
        **success_rates,
    }


def navigation_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = ["success", "spl", "ndtw", "sdtw", "oracle_success", "goal_progress"]
    report: dict[str, Any] = {"num_episodes": len(records)}

    for name in metric_names:
        values = []
        for record in records:
            if name in record and record[name] not in (None, ""):
                values.append(float(record[name]))
        if values:
            report[name] = sum(values) / len(values)

    if "success" not in report:
        inferred_success = []
        for record in records:
            goal_distance = record.get("goal_distance")
            if goal_distance is not None:
                inferred_success.append(1.0 if float(goal_distance) <= 3.0 else 0.0)
        if inferred_success:
            report["success"] = sum(inferred_success) / len(inferred_success)

    if "spl" not in report:
        values = []
        for record in records:
            if {
                "success",
                "path_length",
                "shortest_path_length",
            }.issubset(record.keys()):
                success = float(record["success"])
                path_length = max(float(record["path_length"]), 1e-8)
                shortest = float(record["shortest_path_length"])
                values.append(success * shortest / max(path_length, shortest))
        if values:
            report["spl"] = sum(values) / len(values)

    return report
