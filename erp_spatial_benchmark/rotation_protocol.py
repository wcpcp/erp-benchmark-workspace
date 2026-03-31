#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from evaluate_predictions import (  # type: ignore
    aggregate_accuracy,
    aggregate_group,
    aggregate_slices,
    bfov_center_error_deg,
    extract_prediction_value,
    load_jsonl,
    normalize_text,
    parse_bfov_prediction,
    spherical_bfov_iou,
)


EQUIVARIANT_TASKS = {
    "referring_grounding_bfov",
    "absolute_direction_mc",
}

INVARIANT_TASKS = {
    "relative_direction_mc",
    "camera_rotation_transform_mc",
    "object_conditioned_reorientation_mc",
    "observer_distance_choice",
    "polar_shape_recovery_mc",
}

SUPPORTED_TASKS = EQUIVARIANT_TASKS | INVARIANT_TASKS
SHIFT_OPTIONS = [45, 90, 135, 180]
ABSOLUTE_SECTORS_8 = [
    "front",
    "front-right",
    "right",
    "back-right",
    "back",
    "back-left",
    "left",
    "front-left",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or evaluate the ERP yaw-shift robustness protocol.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build yaw-shift protocol items from benchmark references.")
    build.add_argument("--references", required=True, help="Benchmark reference JSONL with answers.")
    build.add_argument("--output-dir", required=True, help="Output directory for protocol artifacts.")
    build.add_argument("--max-items-per-task", type=int, default=0, help="Optional cap per task after filtering.")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate base + yaw-shifted predictions jointly.")
    evaluate.add_argument("--base-predictions", required=True, help="Predictions on the original benchmark references.")
    evaluate.add_argument("--base-references", required=True, help="Original benchmark references.")
    evaluate.add_argument("--shifted-predictions", required=True, help="Predictions on the yaw-shift protocol prompts.")
    evaluate.add_argument("--shifted-references", required=True, help="Yaw-shift protocol references.")
    evaluate.add_argument("--report", default="", help="Optional output JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build":
        return build_protocol(Path(args.references), Path(args.output_dir), int(args.max_items_per_task))
    return evaluate_protocol(
        Path(args.base_predictions),
        Path(args.base_references),
        Path(args.shifted_predictions),
        Path(args.shifted_references),
        Path(args.report) if args.report else None,
    )


def build_protocol(references_path: Path, output_dir: Path, max_items_per_task: int) -> int:
    refs = load_jsonl(references_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets" / "yaw_shifted"
    assets_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in refs:
        task_id = str(row.get("task_id", ""))
        if task_id in SUPPORTED_TASKS:
            grouped[task_id].append(row)

    protocol_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    for task_id, rows in sorted(grouped.items()):
        selected = list(rows)
        selected.sort(key=lambda row: str(row["item_id"]))
        if max_items_per_task > 0:
            selected = selected[:max_items_per_task]
        for row in selected:
            shift_deg = select_shift_deg(str(row["item_id"]))
            shifted_path = render_shifted_erp(Path(row["image_path"]), assets_dir, shift_deg)
            shifted_row = build_shifted_row(row, shifted_path, shift_deg)
            protocol_rows.append(shifted_row)
            manifest_rows.append(
                {
                    "base_item_id": row["item_id"],
                    "protocol_item_id": shifted_row["item_id"],
                    "task_id": task_id,
                    "protocol_role": shifted_row["metadata"]["rotation_protocol"]["role"],
                    "yaw_shift_deg": shift_deg,
                    "base_image_path": row["image_path"],
                    "shifted_image_path": shifted_row["image_path"],
                }
            )

    prompts = [strip_answers(row) for row in protocol_rows]
    write_jsonl(output_dir / "rotation_protocol.jsonl", protocol_rows)
    write_jsonl(output_dir / "rotation_protocol_prompts.jsonl", prompts)
    write_jsonl(output_dir / "rotation_protocol_references.jsonl", protocol_rows)
    write_jsonl(output_dir / "rotation_protocol_manifest.jsonl", manifest_rows)

    summary = {
        "protocol_name": "ERP Yaw-Shift / Seam-Relocation Robustness Protocol",
        "num_protocol_items": len(protocol_rows),
        "supported_tasks": sorted(SUPPORTED_TASKS),
        "equivariant_tasks": sorted(EQUIVARIANT_TASKS),
        "invariant_tasks": sorted(INVARIANT_TASKS),
        "items_per_task": dict(sorted(_count_by(protocol_rows, "task_id").items())),
    }
    (output_dir / "rotation_protocol_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"stage": "done", "output_dir": str(output_dir), "num_items": len(protocol_rows)}, ensure_ascii=False))
    return 0


def evaluate_protocol(
    base_predictions_path: Path,
    base_references_path: Path,
    shifted_predictions_path: Path,
    shifted_references_path: Path,
    report_path: Optional[Path],
) -> int:
    base_predictions = load_jsonl(base_predictions_path)
    base_references = load_jsonl(base_references_path)
    shifted_predictions = load_jsonl(shifted_predictions_path)
    shifted_references = load_jsonl(shifted_references_path)

    shifted_pred_map = {str(row.get("item_id") or row.get("id")): row for row in shifted_predictions}
    base_pred_map = {str(row.get("item_id") or row.get("id")): row for row in base_predictions}
    base_ref_map = {str(row["item_id"]): row for row in base_references}

    paired_rows: List[Dict[str, Any]] = []
    paired_base_refs: List[Dict[str, Any]] = []
    for shifted_ref in shifted_references:
        protocol = (shifted_ref.get("metadata") or {}).get("rotation_protocol", {})
        base_item_id = str(protocol.get("base_item_id", ""))
        base_ref = base_ref_map.get(base_item_id)
        if not base_ref:
            continue
        paired_base_refs.append(base_ref)
        base_pred = base_pred_map.get(base_item_id)
        shifted_pred = shifted_pred_map.get(str(shifted_ref["item_id"]))
        paired_rows.append(score_pair(base_ref, shifted_ref, base_pred, shifted_pred))

    base_rows = score_rows(base_predictions, paired_base_refs)
    shifted_rows = score_rows(shifted_predictions, shifted_references)

    base_overall = aggregate_accuracy(base_rows)
    shifted_overall = aggregate_accuracy(shifted_rows)
    report = {
        "protocol_name": "ERP Yaw-Shift / Seam-Relocation Robustness Protocol",
        "base_overall": base_overall,
        "shifted_overall": shifted_overall,
        "headline_gap": round(float(shifted_overall["accuracy"]) - float(base_overall["accuracy"]), 6),
        "base_by_task": aggregate_group(base_rows, "task_id"),
        "shifted_by_task": aggregate_group(shifted_rows, "task_id"),
        "base_by_ability_group": aggregate_group(base_rows, "ability_group"),
        "shifted_by_ability_group": aggregate_group(shifted_rows, "ability_group"),
        "shifted_by_diagnostic_slice": aggregate_slices(shifted_rows),
        "pair_consistency": aggregate_pair_consistency(paired_rows),
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


def score_rows(predictions: List[Dict[str, Any]], references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pred_map = {str(row.get("item_id") or row.get("id")): row for row in predictions}
    rows: List[Dict[str, Any]] = []
    for ref in references:
        item_id = str(ref["item_id"])
        pred = pred_map.get(item_id)
        row = {
            "item_id": item_id,
            "task_id": ref["task_id"],
            "ability_group": ref["ability_group"],
            "diagnostic_slices": list(ref.get("diagnostic_slices", [])),
            "has_prediction": pred is not None,
            "correct": False,
        }
        predicted_value = extract_prediction_value(pred)
        if ref["task_id"] == "referring_grounding_bfov":
            gt_bfov = ref.get("answer")
            pred_bfov = parse_bfov_prediction(predicted_value)
            iou = spherical_bfov_iou(pred_bfov, gt_bfov) if pred_bfov is not None and gt_bfov is not None else 0.0
            center_error = bfov_center_error_deg(pred_bfov, gt_bfov) if pred_bfov is not None and gt_bfov is not None else None
            row["bfov_iou"] = round(iou, 6)
            row["bfov_center_error_deg"] = None if center_error is None else round(center_error, 6)
            row["correct"] = bool(iou >= 0.5)
        else:
            accepted = accepted_answer_forms(ref)
            row["correct"] = bool(normalize_text(predicted_value) in accepted) if predicted_value is not None else False
        rows.append(row)
    return rows


def score_pair(
    base_ref: Dict[str, Any],
    shifted_ref: Dict[str, Any],
    base_pred: Optional[Dict[str, Any]],
    shifted_pred: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    protocol = (shifted_ref.get("metadata") or {}).get("rotation_protocol", {})
    role = str(protocol.get("role", "invariant"))
    shift_deg = float(protocol.get("yaw_shift_deg", 0.0))
    task_id = str(base_ref["task_id"])
    row = {
        "task_id": task_id,
        "role": role,
        "base_item_id": base_ref["item_id"],
        "shifted_item_id": shifted_ref["item_id"],
        "consistent": False,
    }
    base_value = extract_prediction_value(base_pred)
    shifted_value = extract_prediction_value(shifted_pred)
    if base_value is None or shifted_value is None:
        return row

    if task_id == "referring_grounding_bfov":
        base_bfov = parse_bfov_prediction(base_value)
        shifted_bfov = parse_bfov_prediction(shifted_value)
        if base_bfov is None or shifted_bfov is None:
            return row
        transformed_base = shift_bfov(base_bfov, shift_deg)
        transformed_iou = spherical_bfov_iou(transformed_base, shifted_bfov)
        row["transformed_iou"] = round(transformed_iou, 6)
        row["consistent"] = bool(transformed_iou >= 0.5)
        return row

    base_semantic = semantic_prediction(base_value, base_ref)
    shifted_semantic = semantic_prediction(shifted_value, shifted_ref)
    if base_semantic is None or shifted_semantic is None:
        return row
    if task_id == "absolute_direction_mc":
        row["expected_shifted_answer"] = shift_absolute_sector(base_semantic, shift_deg)
        row["consistent"] = bool(row["expected_shifted_answer"] == shifted_semantic)
        return row
    row["consistent"] = bool(base_semantic == shifted_semantic)
    return row


def aggregate_pair_consistency(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    total = len(rows)
    consistent = sum(1 for row in rows if row["consistent"])
    payload: Dict[str, Any] = {
        "overall_consistency": round(consistent / total, 6) if total else 0.0,
        "num_pairs": total,
        "num_consistent": consistent,
    }
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_role: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    grounding_ious: List[float] = []
    for row in rows:
        by_task[row["task_id"]].append(row)
        by_role[row["role"]].append(row)
        if row.get("transformed_iou") is not None:
            grounding_ious.append(float(row["transformed_iou"]))
    payload["by_task"] = {
        task_id: round(sum(1 for row in subset if row["consistent"]) / len(subset), 6)
        for task_id, subset in sorted(by_task.items())
    }
    payload["by_role"] = {
        role: round(sum(1 for row in subset if row["consistent"]) / len(subset), 6)
        for role, subset in sorted(by_role.items())
    }
    payload["grounding_transformed_mean_iou"] = round(sum(grounding_ious) / len(grounding_ious), 6) if grounding_ious else None
    return payload


def build_shifted_row(ref: Dict[str, Any], shifted_path: Path, shift_deg: int) -> Dict[str, Any]:
    task_id = str(ref["task_id"])
    role = "equivariant" if task_id in EQUIVARIANT_TASKS else "invariant"
    shifted = dict(ref)
    shifted["item_id"] = f"{ref['item_id']}__yaw_shift_{shift_deg}"
    shifted["image_path"] = str(shifted_path)
    shifted["image_paths"] = []
    shifted["metadata"] = dict(ref.get("metadata") or {})
    shifted["metadata"]["rotation_protocol"] = {
        "base_item_id": ref["item_id"],
        "role": role,
        "yaw_shift_deg": shift_deg,
    }
    shifted["diagnostic_slices"] = sorted(set(list(ref.get("diagnostic_slices", [])) + ["rotation_protocol"]))

    if task_id == "referring_grounding_bfov":
        gt_bfov = ref.get("answer") or ref.get("metadata", {}).get("target_bfov")
        transformed = shift_bfov(gt_bfov, shift_deg)
        shifted["answer"] = list(transformed)
        shifted["answer_text"] = bfov_text(transformed)
        return shifted

    if task_id == "absolute_direction_mc":
        correct = shift_absolute_sector(str(ref["answer_text"]), shift_deg)
        options = [correct] + sector_distractors(correct)[:3]
        shifted["options"] = choice_rows(options)
        shifted["answer"] = shifted["options"][0]["key"]
        shifted["answer_text"] = correct
        shifted["metadata"]["rotation_protocol"]["expected_shifted_answer"] = correct
        return shifted

    return shifted


def render_shifted_erp(image_path: Path, assets_dir: Path, shift_deg: int) -> Path:
    assets_dir.mkdir(parents=True, exist_ok=True)
    stem = hashlib.md5(f"{image_path}:{shift_deg}".encode("utf-8")).hexdigest()[:12]
    out_path = assets_dir / f"{image_path.stem}_yaw_shift_{shift_deg}_{stem}{image_path.suffix or '.jpg'}"
    if out_path.exists():
        return out_path
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    shift_px = int(round((shift_deg / 360.0) * width)) % max(width, 1)
    if shift_px == 0:
        image.save(out_path)
        return out_path
    left = image.crop((0, 0, width - shift_px, height))
    right = image.crop((width - shift_px, 0, width, height))
    canvas = Image.new("RGB", (width, height))
    canvas.paste(right, (0, 0))
    canvas.paste(left, (shift_px, 0))
    canvas.save(out_path)
    return out_path


def shift_bfov(bfov: Sequence[float], shift_deg: float) -> Tuple[float, float, float, float]:
    yaw, pitch, x_fov, y_fov = [float(v) for v in bfov]
    return (normalize_yaw(yaw + shift_deg), pitch, x_fov, y_fov)


def shift_absolute_sector(sector: str, shift_deg: float) -> str:
    steps = int(round(shift_deg / 45.0)) % len(ABSOLUTE_SECTORS_8)
    idx = ABSOLUTE_SECTORS_8.index(str(sector))
    return ABSOLUTE_SECTORS_8[(idx + steps) % len(ABSOLUTE_SECTORS_8)]


def sector_distractors(correct: str) -> List[str]:
    idx = ABSOLUTE_SECTORS_8.index(correct)
    order = [
        ABSOLUTE_SECTORS_8[(idx - 1) % 8],
        ABSOLUTE_SECTORS_8[(idx + 1) % 8],
        ABSOLUTE_SECTORS_8[(idx + 4) % 8],
        ABSOLUTE_SECTORS_8[(idx + 2) % 8],
    ]
    return dedupe_keep_order(order)


def choice_rows(options: Sequence[str]) -> List[Dict[str, str]]:
    return [{"key": chr(ord("A") + idx), "text": text} for idx, text in enumerate(options)]


def semantic_prediction(value: Any, ref: Dict[str, Any]) -> Optional[str]:
    if value is None:
        return None
    text = normalize_text(value)
    options = ref.get("options", []) or []
    for option in options:
        key = normalize_text(option.get("key"))
        label = normalize_text(option.get("text"))
        if text == key:
            return label
        if text == label:
            return label
    answer_text = normalize_text(ref.get("answer_text"))
    if text == answer_text:
        return answer_text
    return text or None


def accepted_answer_forms(ref: Dict[str, Any]) -> set[str]:
    accepted = set()
    if "answer" in ref:
        accepted.add(normalize_text(ref["answer"]))
    if "answer_text" in ref:
        accepted.add(normalize_text(ref["answer_text"]))
    for option in ref.get("options", []) or []:
        key = normalize_text(option.get("key", ""))
        text = normalize_text(option.get("text", ""))
        if key:
            accepted.add(key)
        if text:
            accepted.add(text)
    return {item for item in accepted if item}


def strip_answers(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(row)
    payload.pop("answer", None)
    payload.pop("answer_text", None)
    return payload


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_shift_deg(item_id: str) -> int:
    digest = hashlib.md5(item_id.encode("utf-8")).hexdigest()
    return SHIFT_OPTIONS[int(digest[:8], 16) % len(SHIFT_OPTIONS)]


def normalize_yaw(yaw: float) -> float:
    return ((float(yaw) + 180.0) % 360.0) - 180.0


def bfov_text(bfov: Sequence[float]) -> str:
    return f"[yaw={bfov[0]:.1f}, pitch={bfov[1]:.1f}, x_fov={bfov[2]:.1f}, y_fov={bfov[3]:.1f}]"


def dedupe_keep_order(items: Sequence[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _count_by(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row[key])] += 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
