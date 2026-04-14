#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
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
    contextual_entity_ref,
    find_entity,
    pick_variant,
    rotate_yaw_pitch,
    shuffled_choice_rows,
    stable_hash,
    transformed_bbox,
    write_pitch_rotated_erp_image,
    write_yaw_shifted_erp_image,
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
    parser.add_argument("--target-lat-min-deg", type=float, default=55.0)
    parser.add_argument("--target-lat-max-deg", type=float, default=78.0)
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
    value = int(hashlib.md5(item_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return min_lat_deg + value * (max_lat_deg - min_lat_deg)


def choose_combined_rotation_for_entity(
    item_id: str,
    entity: Entity,
    *,
    min_lat_deg: float,
    max_lat_deg: float,
) -> Optional[Tuple[float, float]]:
    bfov = entity.resolved_bfov
    if bfov is None:
        return None
    yaw_deg = float(bfov[0])
    pitch_deg = float(bfov[1])
    target_lat = choose_target_lat(item_id, min_lat_deg, max_lat_deg)
    best: Optional[Tuple[float, float, float, float]] = None
    yaw_candidates = [0.0, -150.0, -120.0, -90.0, -60.0, -45.0, -30.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 180.0]
    for yaw_shift in yaw_candidates:
        for pitch_shift in range(-89, 90):
            if yaw_shift == 0.0 and pitch_shift == 0:
                continue
            _, new_pitch = rotate_yaw_pitch(
                yaw_deg,
                pitch_deg,
                yaw_shift_deg=float(yaw_shift),
                pitch_shift_deg=float(pitch_shift),
            )
            abs_lat = abs(-new_pitch)
            if min_lat_deg <= abs_lat <= max_lat_deg:
                candidate = (
                    abs(abs_lat - target_lat),
                    abs(float(yaw_shift)),
                    abs(float(pitch_shift)),
                    float(yaw_shift),
                    float(pitch_shift),
                )
                if best is None or candidate < best:
                    best = candidate
    return None if best is None else (best[3], best[4])


def random_fallback_rotation(item_id: str, entity_id: str) -> Tuple[float, float]:
    seed = stable_hash(f"{item_id}:{entity_id}:fallback_rotation")
    yaw_mag = 30.0 + float(seed % 21)
    pitch_mag = 30.0 + float((seed // 29) % 21)
    yaw_sign = -1.0 if ((seed // 7) % 2 == 0) else 1.0
    pitch_sign = -1.0 if ((seed // 11) % 2 == 0) else 1.0
    return yaw_sign * yaw_mag, pitch_sign * pitch_mag


def ordered_candidate_entity_ids(item: Dict[str, Any]) -> List[str]:
    metadata = item.get("metadata") or {}
    candidate_depths = metadata.get("candidate_depths_m") or {}
    if candidate_depths:
        sorted_depths = sorted(
            ((str(entity_id), float(depth)) for entity_id, depth in candidate_depths.items()),
            key=lambda pair: pair[1],
        )
        return [entity_id for entity_id, _ in sorted_depths]
    return [str(entity_id) for entity_id in (item.get("target_entities") or [])]


def nearest_incorrect_entity_ids(item: Dict[str, Any]) -> List[str]:
    ordered = ordered_candidate_entity_ids(item)
    return ordered[1:]


def entity_can_be_pitch_hardened(entity: Optional[Entity]) -> bool:
    return (
        entity is not None
        and entity.resolved_bfov is not None
        and entity.entity_center_depth is not None
        and float(entity.area_ratio or 0.0) > 0.0004
    )


def supplemental_distractor_ids(scene: SceneMetadata, item: Dict[str, Any]) -> List[str]:
    ordered_ids = ordered_candidate_entity_ids(item)
    existing_ids = set(ordered_ids)
    correct_id = ordered_ids[0] if ordered_ids else ""
    correct_entity = find_entity(scene, correct_id) if correct_id else None
    correct_depth = float(correct_entity.entity_center_depth) if correct_entity and correct_entity.entity_center_depth is not None else None
    ranked: List[Tuple[float, float, str]] = []
    for entity in scene.entities:
        if entity.entity_id in existing_ids:
            continue
        if not entity_can_be_pitch_hardened(entity):
            continue
        depth = float(entity.entity_center_depth)
        depth_gap = abs(depth - correct_depth) if correct_depth is not None else depth
        ranked.append(
            (
                depth_gap,
                -float(entity.area_ratio or 0.0),
                str(entity.entity_id),
            )
        )
    ranked.sort()
    return [entity_id for _, _, entity_id in ranked]


def build_pitch_rotated_scene(
    scene: SceneMetadata,
    *,
    yaw_shift_deg: float,
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
    if abs(yaw_shift_deg) > 1e-6 and abs(pitch_shift_deg) > 1e-6:
        tmp_dir = output_root / "_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_image = tmp_dir / f"{raw['scene_id']}__yaw_tmp{src_image.suffix}"
        write_yaw_shifted_erp_image(src_image, tmp_image, yaw_shift_deg)
        write_pitch_rotated_erp_image(tmp_image, rotated_image, pitch_shift_deg)
        tmp_image.unlink(missing_ok=True)
    elif abs(yaw_shift_deg) > 1e-6:
        write_yaw_shifted_erp_image(src_image, rotated_image, yaw_shift_deg)
    else:
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
            yaw_shift_deg=yaw_shift_deg,
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
            yaw_shift_deg=yaw_shift_deg,
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


def choose_option_entity_ids(scene: SceneMetadata, item: Dict[str, Any], hard_entity_id: str) -> List[str]:
    ordered = ordered_candidate_entity_ids(item)
    if not ordered:
        return [hard_entity_id]
    correct_id = ordered[0]
    existing_wrong_ids = [entity_id for entity_id in ordered[1:] if entity_id != hard_entity_id]
    desired_count = max(3, len(item.get("options") or []), len(ordered))
    selected = [correct_id]
    if hard_entity_id != correct_id:
        selected.append(hard_entity_id)
    for entity_id in existing_wrong_ids:
        if entity_id not in selected:
            selected.append(entity_id)
        if len(selected) >= desired_count:
            break
    if len(selected) < desired_count:
        for entity_id in supplemental_distractor_ids(scene, item):
            if entity_id not in selected and entity_id != correct_id:
                selected.append(entity_id)
            if len(selected) >= desired_count:
                break
    return selected[:desired_count]


def rebuild_observer_options(scene: SceneMetadata, item: Dict[str, Any], option_entity_ids: Sequence[str]) -> Tuple[List[Dict[str, str]], str, str, List[str], Dict[str, float]]:
    option_entities: List[Entity] = []
    for entity_id in option_entity_ids:
        entity = find_entity(scene, entity_id)
        if entity is not None:
            option_entities.append(entity)
    if len(option_entities) < 3:
        raise ValueError("Need at least three valid option entities to rebuild observer_distance_choice.")
    option_entities = sorted(option_entities, key=lambda entity: float(entity.entity_center_depth or 1e9))
    correct_entity = option_entities[0]
    option_refs = [contextual_entity_ref(scene, entity) for entity in option_entities]
    choices, answer_key = shuffled_choice_rows(
        option_refs,
        option_refs[0],
        f"observer_distance_choice:{item.get('item_id', '')}:rebuilt_options",
    )
    candidate_depths = {
        entity.entity_id: round(float(entity.entity_center_depth or 0.0), 3)
        for entity in option_entities
    }
    return choices, answer_key, option_refs[0], [entity.entity_id for entity in option_entities], candidate_depths


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
        original_wrong_ids = nearest_incorrect_entity_ids(item)
        original_wrong_id_set = set(original_wrong_ids)
        candidate_ids = original_wrong_ids + [
            entity_id for entity_id in supplemental_distractor_ids(scene, item)
            if entity_id not in original_wrong_id_set
        ]
        for distractor_id in candidate_ids:
            distractor = find_entity(scene, distractor_id)
            if not entity_can_be_pitch_hardened(distractor):
                continue
            rotation = choose_combined_rotation_for_entity(
                item_id,
                distractor,
                min_lat_deg=float(args.target_lat_min_deg),
                max_lat_deg=float(args.target_lat_max_deg),
            )
            fallback_used = False
            if rotation is None:
                rotation = random_fallback_rotation(item_id, distractor.entity_id)
                fallback_used = True
            yaw_shift, pitch_shift = rotation
            option_entity_ids = choose_option_entity_ids(scene, item, distractor.entity_id)
            try:
                choices, answer_key, answer_text, target_entities, candidate_depths = rebuild_observer_options(
                    scene,
                    item,
                    option_entity_ids,
                )
            except ValueError:
                continue

            suffix = f"observer_distance_rot_y{int(round(yaw_shift))}_p{int(round(pitch_shift))}_{distractor.entity_id}"
            derived_scene, metadata_path = build_pitch_rotated_scene(
                scene,
                yaw_shift_deg=yaw_shift,
                pitch_shift_deg=pitch_shift,
                suffix=suffix,
                output_root=output_root,
            )
            derived_distractor = find_entity(derived_scene, distractor.entity_id)
            abs_lat = abs(derived_distractor.lat_deg) if derived_distractor is not None else None

            item["scene_id"] = derived_scene.scene_id
            item["image_path"] = derived_scene.erp_image_path
            item["options"] = choices
            item["answer"] = answer_key
            item["answer_text"] = answer_text
            item["target_entities"] = target_entities
            item.setdefault("metadata", {})
            item["metadata"]["source_item_id"] = item_id
            item["metadata"]["source_scene_id"] = scene.scene_id
            item["metadata"]["derived_metadata_path"] = str(metadata_path)
            item["metadata"]["candidate_refs"] = [choice["text"] for choice in choices]
            item["metadata"]["candidate_depths_m"] = candidate_depths
            item["metadata"]["observer_distance_hardening"] = {
                "strategy": "yaw_pitch_rotate_hard_distractor" if not fallback_used else "fallback_random_yaw_pitch_rotation",
                "distractor_entity_id": distractor.entity_id,
                "correct_prediction_before_hardening": prediction,
                "yaw_shift_deg": float(yaw_shift),
                "pitch_shift_deg": float(pitch_shift),
                "distractor_abs_lat_deg": None if abs_lat is None else round(float(abs_lat), 2),
                "used_extra_distractor": distractor.entity_id not in original_wrong_id_set,
                "fallback_used": bool(fallback_used),
            }
            item["difficulty"] = "hard"
            slices = set(item.get("diagnostic_slices") or [])
            slices.update({"observer_distance_hardened", "pole"})
            item["diagnostic_slices"] = sorted(slices)
            hardened = True
            if fallback_used:
                report["observer_hardened_with_fallback_rotation"] += 1
            else:
                report["observer_hardened"] += 1
            break

        if not hardened:
            report["observer_correct_but_unmodified"] += 1
            failures.append({"item_id": item_id, "reason": "no_valid_rotation_candidate"})

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
