#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from erp_spatial_benchmark._vendor.schemas import Entity, SceneMetadata
from erp_spatial_benchmark.build_benchmark import (
    bfov_extents_from_item,
    find_entity,
    pick_variant,
    rotate_yaw_pitch,
    transformed_bbox,
    write_pitch_rotated_erp_image,
    xyz_camera_from_yaw_pitch_depth,
)

OBSERVER_DISTANCE_TEMPLATES = [
    "Which listed object is physically closest to the current observer in the 3D scene?",
    "From the current observer position, which listed object is nearest in 3D space?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harden easy observer_distance_choice items by pitch-rotating scenes so a wrong option is pushed into a high-distortion latitude band."
    )
    parser.add_argument("--input-jsonl", required=True, help="Filtered benchmark JSONL to rewrite.")
    parser.add_argument(
        "--predictions-jsonl",
        required=True,
        help="Model predictions JSONL. Correct observer_distance_choice items are candidates for hardening.",
    )
    parser.add_argument(
        "--metadata-roots",
        nargs="+",
        required=True,
        help="One or more roots containing metadata.json and/or derived_metadata/*.json files.",
    )
    parser.add_argument("--output-jsonl", required=True, help="Output rewritten JSONL path.")
    parser.add_argument(
        "--output-root",
        default="",
        help="Directory for generated rotated images and derived metadata. Defaults to <output-jsonl-dir>/observer_distance_hardened.",
    )
    parser.add_argument("--target-lat-min-deg", type=float, default=68.0)
    parser.add_argument("--target-lat-max-deg", type=float, default=82.0)
    parser.add_argument("--max-entities-per-scene", type=int, default=0, help="Reserved for future use.")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_prediction_map(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for row in iter_jsonl(path):
        item_id = str(row.get("item_id", "")).strip()
        prediction = str(row.get("prediction", "")).strip()
        if item_id:
            mapping[item_id] = prediction
    return mapping


def discover_scene_metadata_paths(roots: Sequence[Path]) -> Dict[str, Path]:
    scene_index: Dict[str, Path] = {}
    for root in roots:
        if root.is_file():
            candidates = [root]
        else:
            candidates = []
            for dirpath, _, filenames in os.walk(root):
                for filename in filenames:
                    path = Path(dirpath) / filename
                    if filename == "metadata.json":
                        candidates.append(path)
                    elif path.suffix == ".json" and path.parent.name == "derived_metadata":
                        candidates.append(path)
        for path in candidates:
            try:
                if path.name == "metadata.json":
                    data = load_json(path)
                    scene_id = str(data.get("scene_id", "") or data.get("image_id", "")).strip()
                else:
                    scene_id = path.stem
                if scene_id and scene_id not in scene_index:
                    scene_index[scene_id] = path
            except Exception:
                continue
    return scene_index


def load_scene_from_path(path: Path) -> SceneMetadata:
    return SceneMetadata.from_dict(load_json(path))


def resolve_scene_metadata_path(item: Dict[str, Any], scene_index: Dict[str, Path]) -> Optional[Path]:
    metadata = item.get("metadata") or {}
    derived_metadata_path = metadata.get("derived_metadata_path")
    if derived_metadata_path:
        candidate = Path(str(derived_metadata_path))
        if candidate.exists():
            return candidate

    image_path = str(item.get("image_path", "")).strip()
    image_stem = Path(image_path).stem if image_path else ""
    for key in [
        str(item.get("scene_id", "")).strip(),
        image_stem,
        str((metadata.get("derived_rotation") or {}).get("derived_scene_id", "")).strip(),
        str((metadata.get("derived_rotation") or {}).get("source_scene_id", "")).strip(),
    ]:
        if key and key in scene_index:
            return scene_index[key]
    return None


def choose_target_lat(item_id: str, min_lat_deg: float, max_lat_deg: float) -> float:
    import hashlib

    value = int(hashlib.md5(item_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return min_lat_deg + value * (max_lat_deg - min_lat_deg)


def choose_pitch_shift_for_entity(
    item_id: str,
    entity: Entity,
    *,
    min_lat_deg: float,
    max_lat_deg: float,
) -> Optional[float]:
    bfov = entity.resolved_bfov
    if bfov is None:
        return None
    yaw_deg = float(bfov[0])
    pitch_deg = float(bfov[1])
    target_lat = choose_target_lat(item_id, min_lat_deg, max_lat_deg)
    best: Optional[Tuple[float, float, float]] = None
    for shift in range(-89, 90):
        if shift == 0:
            continue
        _, new_pitch = rotate_yaw_pitch(yaw_deg, pitch_deg, pitch_shift_deg=float(shift))
        abs_lat = abs(-new_pitch)
        if min_lat_deg <= abs_lat <= max_lat_deg:
            candidate = (abs(abs_lat - target_lat), abs(shift), float(shift))
            if best is None or candidate < best:
                best = candidate
    return None if best is None else best[2]


def nearest_incorrect_entity_ids(item: Dict[str, Any]) -> List[str]:
    metadata = item.get("metadata") or {}
    candidate_depths = metadata.get("candidate_depths_m") or {}
    if candidate_depths:
        sorted_depths = sorted(
            ((str(entity_id), float(depth)) for entity_id, depth in candidate_depths.items()),
            key=lambda pair: pair[1],
        )
        if sorted_depths:
            correct_entity_id = sorted_depths[0][0]
            return [entity_id for entity_id, _ in sorted_depths[1:] if entity_id != correct_entity_id]
    target_entities = [str(entity_id) for entity_id in (item.get("target_entities") or [])]
    return target_entities[1:]


def entity_can_be_pitch_hardened(entity: Optional[Entity]) -> bool:
    return entity is not None and entity.resolved_bfov is not None


def build_pitch_rotated_scene(
    scene: SceneMetadata,
    *,
    pitch_shift_deg: float,
    suffix: str,
    output_root: Path,
) -> Tuple[SceneMetadata, Path]:
    raw = copy.deepcopy(scene.raw)
    raw["scene_id"] = f"{scene.scene_id}__{suffix}"
    raw["erp_width"] = int(scene.erp_width or 0)
    raw["erp_height"] = int(scene.erp_height or 0)

    src_image = Path(scene.erp_image_path)
    image_dir = output_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rotated_image = image_dir / f"{raw['scene_id']}{src_image.suffix}"
    write_pitch_rotated_erp_image(src_image, rotated_image, pitch_shift_deg)
    raw["erp_image_path"] = str(rotated_image)
    raw["image_path"] = str(rotated_image)

    transformed_entities: List[Dict[str, Any]] = []
    for entity_raw in scene.raw.get("entities", []):
        item = copy.deepcopy(entity_raw)
        bfov = item.get("bfov", {}) or {}
        x_fov_deg, y_fov_deg = bfov_extents_from_item(item)
        yaw_old = float(bfov.get("yaw_deg", 0.0))
        pitch_old = float(bfov.get("pitch_deg", 0.0))
        yaw_new, pitch_new = rotate_yaw_pitch(
            yaw_old,
            pitch_old,
            pitch_shift_deg=pitch_shift_deg,
        )
        lat_new = -pitch_new
        lon_rad = math.radians(yaw_new)
        lat_rad = math.radians(lat_new)
        item["lon_lat"] = [lon_rad, lat_rad]
        item.setdefault("bfov", {})
        item["bfov"]["yaw_deg"] = yaw_new
        item["bfov"]["pitch_deg"] = pitch_new
        if x_fov_deg is not None:
            item["bfov"]["x_fov_deg"] = x_fov_deg
        if y_fov_deg is not None:
            item["bfov"]["y_fov_deg"] = y_fov_deg
        if x_fov_deg is not None and y_fov_deg is not None:
            item["entity_bfov"] = [yaw_new, pitch_new, x_fov_deg, y_fov_deg]
        bbox, seam_cross = transformed_bbox(
            Entity.from_dict(entity_raw),
            scene,
            pitch_shift_deg=pitch_shift_deg,
        )
        item["bbox_erp"] = bbox
        item["seam_crossing_flag"] = bool(seam_cross)
        item["pole_proximity_flag"] = abs(lat_new) >= 60.0
        rotated_xyz = xyz_camera_from_yaw_pitch_depth(yaw_new, pitch_new, item.get("entity_center_depth"))
        item["entity_xyz_camera"] = rotated_xyz
        spatial = item.get("spatial", {}) or {}
        spatial["yaw_deg"] = yaw_new
        spatial["pitch_deg"] = pitch_new
        spatial["xyz_camera_m"] = rotated_xyz
        if item.get("entity_center_depth") is not None:
            spatial["range_m"] = float(item["entity_center_depth"])
        item["spatial"] = spatial
        item["mask_rle"] = {}
        transformed_entities.append(item)

    raw["entities"] = transformed_entities
    metadata_dir = output_root / "derived_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f"{raw['scene_id']}.json"
    metadata_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return SceneMetadata.from_dict(raw), metadata_path


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    predictions_path = Path(args.predictions_jsonl)
    output_jsonl = Path(args.output_jsonl)
    output_root = Path(args.output_root) if args.output_root else output_jsonl.parent / "observer_distance_hardened"
    output_root.mkdir(parents=True, exist_ok=True)

    predictions = build_prediction_map(predictions_path)
    scene_index = discover_scene_metadata_paths([Path(path) for path in args.metadata_roots])

    output_rows: List[Dict[str, Any]] = []
    report = Counter()
    failures: List[Dict[str, Any]] = []

    for row in iter_jsonl(input_path):
        item = copy.deepcopy(row)
        if str(item.get("task_id", "")) != "observer_distance_choice":
            output_rows.append(item)
            continue

        item["question"] = pick_variant(
            OBSERVER_DISTANCE_TEMPLATES,
            f"observer_distance_choice:{item.get('item_id', '')}",
        )
        item_id = str(item.get("item_id", ""))
        prediction = predictions.get(item_id, "")
        if not prediction:
            report["observer_missing_prediction"] += 1
            output_rows.append(item)
            continue
        if prediction != str(item.get("answer", "")):
            report["observer_incorrect_kept"] += 1
            output_rows.append(item)
            continue

        scene_path = resolve_scene_metadata_path(item, scene_index)
        if scene_path is None:
            report["observer_hardening_failed"] += 1
            failures.append({"item_id": item_id, "reason": "scene_metadata_not_found"})
            output_rows.append(item)
            continue

        scene = load_scene_from_path(scene_path)
        hardened = False
        for distractor_id in nearest_incorrect_entity_ids(item):
            distractor = find_entity(scene, distractor_id)
            if not entity_can_be_pitch_hardened(distractor):
                continue
            shift = choose_pitch_shift_for_entity(
                item_id,
                distractor,
                min_lat_deg=float(args.target_lat_min_deg),
                max_lat_deg=float(args.target_lat_max_deg),
            )
            if shift is None:
                continue

            suffix = f"observer_distance_pitch_{int(round(shift))}_{distractor.entity_id}"
            derived_scene, metadata_path = build_pitch_rotated_scene(
                scene,
                pitch_shift_deg=shift,
                suffix=suffix,
                output_root=output_root,
            )
            derived_distractor = find_entity(derived_scene, distractor.entity_id)
            abs_lat = abs(derived_distractor.lat_deg) if derived_distractor is not None else None

            item["scene_id"] = derived_scene.scene_id
            item["image_path"] = derived_scene.erp_image_path
            item.setdefault("metadata", {})
            item["metadata"]["source_item_id"] = item_id
            item["metadata"]["source_scene_id"] = scene.scene_id
            item["metadata"]["derived_metadata_path"] = str(metadata_path)
            item["metadata"]["observer_distance_hardening"] = {
                "strategy": "pitch_rotate_wrong_option_to_high_latitude",
                "distractor_entity_id": distractor.entity_id,
                "correct_prediction_before_hardening": prediction,
                "pitch_shift_deg": float(shift),
                "distractor_abs_lat_deg": None if abs_lat is None else round(float(abs_lat), 2),
            }
            item["difficulty"] = "hard"
            slices = set(item.get("diagnostic_slices") or [])
            slices.update({"observer_distance_hardened", "pole"})
            item["diagnostic_slices"] = sorted(slices)
            hardened = True
            report["observer_hardened"] += 1
            break

        if not hardened:
            report["observer_correct_but_unmodified"] += 1
            failures.append({"item_id": item_id, "reason": "no_valid_pitch_hardening_candidate"})

        output_rows.append(item)

    write_jsonl(output_jsonl, output_rows)
    report_path = output_jsonl.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(
            {
                "input_jsonl": str(input_path),
                "predictions_jsonl": str(predictions_path),
                "output_jsonl": str(output_jsonl),
                "output_root": str(output_root),
                "scene_index_size": len(scene_index),
                "counts": dict(sorted(report.items())),
                "failure_examples": failures[:50],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_jsonl": str(output_jsonl),
                "report_path": str(report_path),
                "counts": dict(sorted(report.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
