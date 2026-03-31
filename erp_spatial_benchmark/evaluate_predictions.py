#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predictions on the ERP Spatial Benchmark.")
    parser.add_argument("--predictions", required=True, help="JSONL predictions with item_id and prediction.")
    parser.add_argument("--references", required=True, help="JSONL benchmark_public_references or any answer-bearing benchmark export.")
    parser.add_argument("--report", default="", help="Optional output JSON report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions = load_jsonl(Path(args.predictions))
    references = load_jsonl(Path(args.references))
    report = evaluate(predictions, references)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    return " ".join(text.split())


def evaluate(predictions: List[Dict[str, Any]], references: List[Dict[str, Any]]) -> Dict[str, Any]:
    pred_map = {str(row.get("item_id") or row.get("id")): row for row in predictions}
    ref_map = {str(row["item_id"]): row for row in references}

    missing = [item_id for item_id in ref_map if item_id not in pred_map]
    extra = [item_id for item_id in pred_map if item_id not in ref_map]

    rows: List[Dict[str, Any]] = []
    for item_id, ref in ref_map.items():
        pred = pred_map.get(item_id)
        rows.append(score_one(item_id, ref, pred))

    overall = aggregate_accuracy(rows)
    by_task = aggregate_group(rows, "task_id")
    by_ability = aggregate_group(rows, "ability_group")
    by_slice = aggregate_slices(rows)

    report = {
        "benchmark_name": "ERP Spatial Benchmark",
        "metric_policy": {
            "primary_metric": "ability_group_macro_accuracy",
            "item_metric": "closed_form_exact_match_with_bfov_iou_for_grounding",
            "accepted_prediction_forms": [
                "option key such as A/B/C/D/E/F",
                "exact option text",
                "reference answer_text normalized",
                "for referring_grounding_bfov: [yaw, pitch, x_fov, y_fov]",
            ],
            "grounding_metric": {
                "task_id": "referring_grounding_bfov",
                "score": "seam_aware_spherical_bfov_iou",
                "correctness_threshold": 0.5,
            },
        },
        "num_references": len(references),
        "num_predictions": len(predictions),
        "num_missing_predictions": len(missing),
        "num_extra_predictions": len(extra),
        "missing_prediction_ids": missing[:100],
        "extra_prediction_ids": extra[:100],
        "headline_score": by_ability.get("_macro_average", 0.0),
        "task_macro_score": by_task.get("_macro_average", 0.0),
        "overall": overall,
        "by_task": by_task,
        "by_ability_group": by_ability,
        "by_diagnostic_slice": by_slice,
        "grounding_regression_metrics": aggregate_grounding_metrics(rows),
    }
    return report


def score_one(item_id: str, ref: Dict[str, Any], pred: Dict[str, Any] | None) -> Dict[str, Any]:
    predicted_value = extract_prediction_value(pred)
    row = {
        "item_id": item_id,
        "task_id": ref["task_id"],
        "ability_group": ref["ability_group"],
        "diagnostic_slices": list(ref.get("diagnostic_slices", [])),
        "has_prediction": pred is not None,
        "correct": False,
    }
    if ref["task_id"] == "referring_grounding_bfov":
        gt_bfov = ref.get("answer")
        pred_bfov = parse_bfov_prediction(predicted_value)
        iou = spherical_bfov_iou(pred_bfov, gt_bfov) if pred_bfov is not None and gt_bfov is not None else 0.0
        center_error = bfov_center_error_deg(pred_bfov, gt_bfov) if pred_bfov is not None and gt_bfov is not None else None
        row["bfov_iou"] = round(iou, 6)
        row["bfov_center_error_deg"] = None if center_error is None else round(center_error, 6)
        row["correct"] = bool(iou >= 0.5)
        return row

    accepted_answers = accepted_answer_forms(ref)
    predicted_norm = normalize_text(predicted_value) if predicted_value is not None else ""
    row["correct"] = bool(predicted_norm in accepted_answers)
    return row


def accepted_answer_forms(ref: Dict[str, Any]) -> set[str]:
    accepted = set()
    if "answer" in ref:
        accepted.add(normalize_text(ref["answer"]))
    if "answer_text" in ref:
        accepted.add(normalize_text(ref["answer_text"]))

    options = ref.get("options", []) or []
    answer_key = normalize_text(ref.get("answer", ""))
    for option in options:
        key = normalize_text(option.get("key", ""))
        text = normalize_text(option.get("text", ""))
        if key and key == answer_key:
            accepted.add(key)
            if text:
                accepted.add(text)
    return {item for item in accepted if item}


def extract_prediction_value(pred: Dict[str, Any] | None) -> Any:
    if pred is None:
        return None
    for key in ("prediction", "answer", "predicted_answer", "output", "text"):
        if key in pred and pred[key] is not None:
            return pred[key]
    return None


def aggregate_accuracy(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    answered = sum(1 for row in rows if row["has_prediction"])
    return {
        "accuracy": round(correct / total, 6) if total else 0.0,
        "coverage": round(answered / total, 6) if total else 0.0,
        "num_items": total,
        "num_correct": correct,
        "num_answered": answered,
    }


def aggregate_group(rows: Iterable[Dict[str, Any]], key: str) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    payload: Dict[str, Any] = {}
    accuracies: List[float] = []
    for name, subset in sorted(grouped.items()):
        stats = aggregate_accuracy(subset)
        payload[name] = stats
        accuracies.append(float(stats["accuracy"]))
    payload["_macro_average"] = round(sum(accuracies) / len(accuracies), 6) if accuracies else 0.0
    return payload


def aggregate_slices(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for slice_name in row.get("diagnostic_slices", []):
            grouped[str(slice_name)].append(row)
    payload: Dict[str, Any] = {}
    for name, subset in sorted(grouped.items()):
        payload[name] = aggregate_accuracy(subset)
    return payload


def parse_bfov_prediction(value: Any) -> Optional[Tuple[float, float, float, float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return tuple(float(x) for x in value)  # type: ignore[return-value]
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, (list, tuple)) and len(parsed) == 4:
            return tuple(float(x) for x in parsed)  # type: ignore[return-value]
    except Exception:
        pass
    cleaned = text.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    parts = [part.strip() for part in cleaned.split(",")]
    if len(parts) != 4:
        return None
    values: List[float] = []
    for part in parts:
        if "=" in part:
            part = part.split("=", 1)[1].strip()
        try:
            values.append(float(part))
        except Exception:
            return None
    return tuple(values)  # type: ignore[return-value]


def normalize_yaw_deg(yaw: float) -> float:
    return ((float(yaw) + 180.0) % 360.0) - 180.0


def wrap_yaw_360(yaw: float) -> float:
    return float(yaw) % 360.0


def bfov_to_spherical_bounds(bfov: Sequence[float]) -> Tuple[List[Tuple[float, float]], float, float]:
    yaw, pitch, x_fov, y_fov = [float(v) for v in bfov]
    yaw = wrap_yaw_360(yaw)
    x_half = max(0.0, x_fov / 2.0)
    y_half = max(0.0, y_fov / 2.0)
    left = yaw - x_half
    right = yaw + x_half
    intervals: List[Tuple[float, float]] = []
    if left < 0.0:
        intervals.append((0.0, right))
        intervals.append((360.0 + left, 360.0))
    elif right >= 360.0:
        intervals.append((left, 360.0))
        intervals.append((0.0, right - 360.0))
    else:
        intervals.append((left, right))

    center_lat = -pitch
    lat_min = max(-90.0, center_lat - y_half)
    lat_max = min(90.0, center_lat + y_half)
    return intervals, lat_min, lat_max


def spherical_rect_area(intervals: Sequence[Tuple[float, float]], lat_min: float, lat_max: float) -> float:
    if lat_max <= lat_min:
        return 0.0
    lat_term = abs(math.sin(math.radians(lat_max)) - math.sin(math.radians(lat_min)))
    yaw_width_deg = sum(max(0.0, end - start) for start, end in intervals)
    return math.radians(yaw_width_deg) * lat_term


def spherical_bfov_iou(pred_bfov: Sequence[float], gt_bfov: Sequence[float]) -> float:
    pred_intervals, pred_lat_min, pred_lat_max = bfov_to_spherical_bounds(pred_bfov)
    gt_intervals, gt_lat_min, gt_lat_max = bfov_to_spherical_bounds(gt_bfov)
    inter_lat_min = max(pred_lat_min, gt_lat_min)
    inter_lat_max = min(pred_lat_max, gt_lat_max)
    if inter_lat_max <= inter_lat_min:
        return 0.0

    inter_intervals: List[Tuple[float, float]] = []
    for p_start, p_end in pred_intervals:
        for g_start, g_end in gt_intervals:
            start = max(p_start, g_start)
            end = min(p_end, g_end)
            if end > start:
                inter_intervals.append((start, end))

    inter_area = spherical_rect_area(inter_intervals, inter_lat_min, inter_lat_max)
    pred_area = spherical_rect_area(pred_intervals, pred_lat_min, pred_lat_max)
    gt_area = spherical_rect_area(gt_intervals, gt_lat_min, gt_lat_max)
    union = pred_area + gt_area - inter_area
    if union <= 0.0:
        return 0.0
    return max(0.0, min(1.0, inter_area / union))


def bfov_center_error_deg(pred_bfov: Sequence[float], gt_bfov: Sequence[float]) -> float:
    pred_yaw, pred_pitch = float(pred_bfov[0]), float(pred_bfov[1])
    gt_yaw, gt_pitch = float(gt_bfov[0]), float(gt_bfov[1])
    yaw_gap = abs(normalize_yaw_deg(pred_yaw - gt_yaw))
    pitch_gap = abs(pred_pitch - gt_pitch)
    return math.sqrt(yaw_gap * yaw_gap + pitch_gap * pitch_gap)


def aggregate_grounding_metrics(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    grounding_rows = [row for row in rows if row["task_id"] == "referring_grounding_bfov" and row.get("has_prediction")]
    if not grounding_rows:
        return {
            "num_items": 0,
            "mean_bfov_iou": 0.0,
            "mean_center_error_deg": None,
        }
    ious = [float(row.get("bfov_iou", 0.0)) for row in grounding_rows]
    center_errors = [float(row["bfov_center_error_deg"]) for row in grounding_rows if row.get("bfov_center_error_deg") is not None]
    return {
        "num_items": len(grounding_rows),
        "mean_bfov_iou": round(sum(ious) / len(ious), 6),
        "mean_center_error_deg": round(sum(center_errors) / len(center_errors), 6) if center_errors else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
