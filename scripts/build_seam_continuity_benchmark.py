#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from erp_spatial_benchmark._vendor.entity_selector import score_entity
from erp_spatial_benchmark._vendor.schemas import Entity, SceneMetadata
from erp_spatial_benchmark.build_benchmark import (
    bfov_extents_from_item,
    benchmark_entity_eligible,
    contextual_entity_ref,
    find_entity,
    pick_variant,
    reference_is_resolvable,
    rotate_yaw_pitch,
    shuffled_choice_rows,
    slices_for_entity,
    stable_hash,
    transformed_bbox,
    wrapped_delta_deg,
    write_yaw_shifted_erp_image,
    xyz_camera_from_yaw_pitch_depth,
    yaw_deg_360,
)

ABILITY_GROUP = "erp_representation_understanding"
TASK_ID = "seam_continuity_mc"
NEAREST_SUBTYPE = "nearest_across_seam"
IDENTITY_SUBTYPE = "identity_place_consistency"
NEAREST_TEMPLATES = [
    "For {target_ref} near the {target_side} boundary of the ERP panorama, which listed object is nearest to it?",
    "For {target_ref} close to the {target_side} side of the ERP panorama, which candidate is nearest to it?",
]
IDENTITY_TEMPLATES = [
    "In the 360 scene, the left-boundary appearance described as {left_ref} and the right-boundary appearance described as {right_ref} are best described as:",
    "How should the left-boundary appearance {left_ref} and the right-boundary appearance {right_ref} be interpreted in the 360 scene?",
]
IDENTITY_FIXED_OPTIONS = [
    {"key": "A", "text": "two different objects at the same scene location"},
    {"key": "B", "text": "two appearances of the same object at the same scene location"},
    {"key": "C", "text": "two different objects at different scene locations"},
    {"key": "D", "text": "two appearances of the same object at different scene locations"},
]
IDENTITY_CORRECT_KEY = "B"
IDENTITY_CORRECT_TEXT = "two appearances of the same object at the same scene location"


@dataclass
class Candidate:
    subtype: str
    source_scene_id: str
    source_metadata_path: Path
    score: float
    yaw_shift_deg: float
    target_entity_id: str
    partner_entity_id: str = ""
    distractor_entity_ids: Tuple[str, ...] = ()
    target_side: str = ""
    angular_distance_deg: float = 0.0
    edge_distance_deg: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build only seam_continuity_mc benchmark items from metadatabfov roots.")
    parser.add_argument(
        "--metadata-roots",
        nargs="+",
        required=True,
        help="Roots containing scene-level metadata.json files.",
    )
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory for generated rotated images and derived metadata.",
    )
    parser.add_argument("--nearest-target", type=int, default=300, help="Target number of nearest_across_seam items.")
    parser.add_argument("--identity-target", type=int, default=150, help="Target number of identity_place_consistency items.")
    parser.add_argument("--max-scenes", type=int, default=0, help="Optional cap for scene scanning; 0 means all.")
    return parser.parse_args()


def iter_metadata_files(roots: Sequence[Path]) -> Iterable[Path]:
    for root in roots:
        if root.is_file():
            if root.name == "metadata.json":
                yield root
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for filename in sorted(filenames):
                if filename == "metadata.json":
                    yield Path(dirpath) / filename


def load_scene(path: Path) -> SceneMetadata:
    return SceneMetadata.from_dict(json.loads(path.read_text(encoding="utf-8")))


def entity_quality(entity: Entity, scene: SceneMetadata) -> float:
    label_counts = Counter(item.label for item in scene.entities)
    return float(score_entity(entity, label_counts, scene))


def eligible_entities(scene: SceneMetadata) -> List[Entity]:
    entities: List[Entity] = []
    for entity in scene.entities:
        if not benchmark_entity_eligible(entity):
            continue
        if not reference_is_resolvable(scene, entity):
            continue
        entities.append(entity)
    return entities


def spherical_distance_deg(entity_a: Entity, entity_b: Entity) -> float:
    bfov_a = entity_a.resolved_bfov
    bfov_b = entity_b.resolved_bfov
    if bfov_a is None or bfov_b is None:
        return 180.0
    yaw_a = math.radians(float(bfov_a[0]))
    yaw_b = math.radians(float(bfov_b[0]))
    pitch_a = math.radians(float(bfov_a[1]))
    pitch_b = math.radians(float(bfov_b[1]))
    va = (
        math.cos(pitch_a) * math.sin(yaw_a),
        math.sin(-pitch_a),
        math.cos(pitch_a) * math.cos(yaw_a),
    )
    vb = (
        math.cos(pitch_b) * math.sin(yaw_b),
        math.sin(-pitch_b),
        math.cos(pitch_b) * math.cos(yaw_b),
    )
    dot = max(-1.0, min(1.0, va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2]))
    return math.degrees(math.acos(dot))


def pair_midpoint_yaw_deg(entity_a: Entity, entity_b: Entity) -> float:
    rad_a = math.radians(yaw_deg_360(entity_a))
    rad_b = math.radians(yaw_deg_360(entity_b))
    x = math.cos(rad_a) + math.cos(rad_b)
    y = math.sin(rad_a) + math.sin(rad_b)
    if abs(x) < 1e-6 and abs(y) < 1e-6:
        return yaw_deg_360(entity_a)
    return wrapped_delta_deg(math.degrees(math.atan2(y, x)))


def choose_yaw_shift_for_pair(entity_a: Entity, entity_b: Entity) -> float:
    midpoint = pair_midpoint_yaw_deg(entity_a, entity_b)
    return wrapped_delta_deg(midpoint - 180.0)


def rotated_signed_yaw(entity: Entity, yaw_shift_deg: float) -> float:
    yaw, _ = rotate_yaw_pitch(yaw_deg_360(entity), float(entity.resolved_bfov[1]), yaw_shift_deg=yaw_shift_deg, pitch_shift_deg=0.0)  # type: ignore[index]
    return float(yaw)


def seam_side_from_yaw(signed_yaw_deg: float) -> str:
    return "left" if signed_yaw_deg < 0.0 else "right"


def edge_distance_from_seam(signed_yaw_deg: float) -> float:
    return max(0.0, 180.0 - abs(float(signed_yaw_deg)))


def entity_by_id(scene: SceneMetadata, entity_id: str) -> Optional[Entity]:
    return find_entity(scene, entity_id)


def unique_ref(entity: Entity, scene: SceneMetadata) -> str:
    return contextual_entity_ref(scene, entity)


def choose_distractors(
    scene: SceneMetadata,
    *,
    target: Entity,
    partner: Entity,
    all_entities: Sequence[Entity],
    correct_distance_deg: float,
) -> List[Entity]:
    target_ref = unique_ref(target, scene)
    partner_ref = unique_ref(partner, scene)
    used_refs = {target_ref, partner_ref}
    candidates: List[Tuple[float, float, str, Entity]] = []
    for entity in all_entities:
        if entity.entity_id in {target.entity_id, partner.entity_id}:
            continue
        ref = unique_ref(entity, scene)
        if not ref or ref in used_refs:
            continue
        distance_deg = spherical_distance_deg(target, entity)
        if distance_deg <= max(correct_distance_deg + 8.0, 18.0):
            continue
        quality = entity_quality(entity, scene)
        candidates.append((-quality, -distance_deg, entity.entity_id, entity))
    candidates.sort()
    distractors: List[Entity] = []
    for _, _, _, entity in candidates:
        ref = unique_ref(entity, scene)
        if ref in used_refs:
            continue
        used_refs.add(ref)
        distractors.append(entity)
        if len(distractors) == 3:
            break
    return distractors


def candidate_sort_key(candidate: Candidate) -> Tuple[float, int]:
    return (-candidate.score, stable_hash(f"{candidate.subtype}:{candidate.source_scene_id}:{candidate.target_entity_id}:{candidate.partner_entity_id}"))


def evaluate_directed_nearest_candidate(
    scene: SceneMetadata,
    *,
    target: Entity,
    partner: Entity,
    all_entities: Sequence[Entity],
) -> Optional[Candidate]:
    distance_deg = spherical_distance_deg(target, partner)
    yaw_shift_deg = choose_yaw_shift_for_pair(target, partner)
    target_yaw = rotated_signed_yaw(target, yaw_shift_deg)
    partner_yaw = rotated_signed_yaw(partner, yaw_shift_deg)
    if seam_side_from_yaw(target_yaw) == seam_side_from_yaw(partner_yaw):
        return None
    target_edge_distance = edge_distance_from_seam(target_yaw)
    partner_edge_distance = edge_distance_from_seam(partner_yaw)
    if max(target_edge_distance, partner_edge_distance) > 40.0:
        return None
    distractors = choose_distractors(
        scene,
        target=target,
        partner=partner,
        all_entities=all_entities,
        correct_distance_deg=distance_deg,
    )
    if len(distractors) < 3:
        return None
    avg_quality = (entity_quality(target, scene) + entity_quality(partner, scene)) / 2.0
    seam_bonus = max(0.0, 1.0 - (max(target_edge_distance, partner_edge_distance) / 40.0))
    closeness_bonus = max(0.0, 1.0 - (distance_deg / 45.0))
    score = avg_quality + 0.25 * seam_bonus + 0.25 * closeness_bonus
    return Candidate(
        subtype=NEAREST_SUBTYPE,
        source_scene_id=scene.scene_id,
        source_metadata_path=Path(scene.raw.get("__metadata_path__", "")),
        score=score,
        yaw_shift_deg=yaw_shift_deg,
        target_entity_id=target.entity_id,
        partner_entity_id=partner.entity_id,
        distractor_entity_ids=tuple(entity.entity_id for entity in distractors),
        target_side=seam_side_from_yaw(target_yaw),
        angular_distance_deg=distance_deg,
        edge_distance_deg=max(target_edge_distance, partner_edge_distance),
    )


def best_nearest_candidate(scene: SceneMetadata, entities: Sequence[Entity]) -> Optional[Candidate]:
    best: Optional[Candidate] = None
    for index, entity_a in enumerate(entities):
        for entity_b in entities[index + 1 :]:
            candidate_a = evaluate_directed_nearest_candidate(scene, target=entity_a, partner=entity_b, all_entities=entities)
            candidate_b = evaluate_directed_nearest_candidate(scene, target=entity_b, partner=entity_a, all_entities=entities)
            for candidate in (candidate_a, candidate_b):
                if candidate is None:
                    continue
                if best is None or candidate_sort_key(candidate) < candidate_sort_key(best):
                    best = candidate
    return best


def best_identity_candidate(scene: SceneMetadata, entities: Sequence[Entity]) -> Optional[Candidate]:
    best: Optional[Candidate] = None
    for entity in entities:
        bfov = entity.resolved_bfov
        if bfov is None:
            continue
        x_fov = abs(float(bfov[2]))
        if x_fov < 8.0:
            continue
        quality = entity_quality(entity, scene)
        yaw_shift_deg = wrapped_delta_deg(yaw_deg_360(entity) - 180.0)
        score = quality + min(x_fov, 25.0) / 100.0
        candidate = Candidate(
            subtype=IDENTITY_SUBTYPE,
            source_scene_id=scene.scene_id,
            source_metadata_path=Path(scene.raw.get("__metadata_path__", "")),
            score=score,
            yaw_shift_deg=yaw_shift_deg,
            target_entity_id=entity.entity_id,
        )
        if best is None or candidate_sort_key(candidate) < candidate_sort_key(best):
            best = candidate
    return best


def output_image_path(output_root: Path, source_scene: SceneMetadata, derived_scene_id: str) -> Path:
    suffix = Path(source_scene.erp_image_path).suffix or ".jpg"
    return output_root / "images" / f"{derived_scene_id}{suffix}"


def build_rotated_scene_local(
    scene: SceneMetadata,
    *,
    yaw_shift_deg: float,
    suffix: str,
    output_root: Path,
) -> Optional[SceneMetadata]:
    source_image = Path(scene.erp_image_path)
    if not source_image.exists():
        return None

    raw = copy.deepcopy(scene.raw)
    raw["scene_id"] = f"{scene.scene_id}__{suffix}"
    raw["erp_width"] = int(scene.erp_width or 0)
    raw["erp_height"] = int(scene.erp_height or 0)
    rotated_image = output_image_path(output_root, scene, raw["scene_id"])
    rotated_image.parent.mkdir(parents=True, exist_ok=True)
    write_yaw_shifted_erp_image(source_image, rotated_image, yaw_shift_deg)
    raw["erp_image_path"] = str(rotated_image)
    raw["image_path"] = str(rotated_image)

    transformed_entities: List[Dict[str, Any]] = []
    for entity in scene.raw.get("entities", []):
        item = copy.deepcopy(entity)
        bfov = item.get("bfov", {}) or {}
        x_fov_deg, y_fov_deg = bfov_extents_from_item(item)
        yaw_old = float(bfov.get("yaw_deg", math.degrees(float(item.get("lon_lat", [0.0, 0.0])[0]))))
        pitch_old = float(bfov.get("pitch_deg", -math.degrees(float(item.get("lon_lat", [0.0, 0.0])[1]))))
        yaw_new, pitch_new = rotate_yaw_pitch(
            yaw_old,
            pitch_old,
            yaw_shift_deg=yaw_shift_deg,
            pitch_shift_deg=0.0,
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
        else:
            item.pop("entity_bfov", None)
        bbox, seam_cross = transformed_bbox(
            Entity.from_dict(entity),
            scene,
            yaw_shift_deg=yaw_shift_deg,
            pitch_shift_deg=0.0,
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
    raw["__metadata_path__"] = ""
    metadata_path = output_root / "derived_metadata" / f"{raw['scene_id']}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    raw["__metadata_path__"] = str(metadata_path)
    return SceneMetadata.from_dict(raw)


def benchmark_row(
    *,
    scene: SceneMetadata,
    item_id: str,
    question: str,
    answer: str,
    answer_text: str,
    options: Sequence[Dict[str, str]],
    target_entities: Sequence[str],
    metadata: Dict[str, Any],
    answer_format: str,
    quality_score: float,
    diagnostic_slices: Sequence[str],
    difficulty: str = "hard",
) -> Dict[str, Any]:
    return {
        "item_id": item_id,
        "scene_id": scene.scene_id,
        "task_id": TASK_ID,
        "ability_group": ABILITY_GROUP,
        "answer_format": answer_format,
        "image_path": scene.erp_image_path,
        "image_paths": [],
        "question": question,
        "answer": answer,
        "answer_text": answer_text,
        "options": list(options),
        "target_entities": list(target_entities),
        "metadata": metadata,
        "difficulty": difficulty,
        "quality_score": round(float(quality_score), 4),
        "diagnostic_slices": sorted(set(diagnostic_slices)),
        "requires_manual_review": True,
        "review_notes": ["Verify that the seam-based transformation and labels remain correct in the rotated ERP panorama."],
    }


def materialize_nearest_candidate(candidate: Candidate, output_root: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    scene = load_scene(candidate.source_metadata_path)
    scene.raw["__metadata_path__"] = str(candidate.source_metadata_path)
    suffix = f"seam_pair_yaw_{int(round(candidate.yaw_shift_deg))}_{candidate.target_entity_id}_{candidate.partner_entity_id}"
    derived_scene = build_rotated_scene_local(scene, yaw_shift_deg=candidate.yaw_shift_deg, suffix=suffix, output_root=output_root)
    if derived_scene is None:
        return None, "source_image_missing_or_rotation_failed"

    target = entity_by_id(derived_scene, candidate.target_entity_id)
    partner = entity_by_id(derived_scene, candidate.partner_entity_id)
    distractors = [entity_by_id(derived_scene, entity_id) for entity_id in candidate.distractor_entity_ids]
    if target is None or partner is None or any(entity is None for entity in distractors):
        return None, "missing_rotated_entities"

    target_ref = contextual_entity_ref(derived_scene, target)
    partner_ref = contextual_entity_ref(derived_scene, partner)
    distractor_refs = [contextual_entity_ref(derived_scene, entity) for entity in distractors if entity is not None]
    if len({target_ref, partner_ref, *distractor_refs}) < 5:
        return None, "non_unique_references_after_rotation"

    item_key = f"{derived_scene.scene_id}:{candidate.target_entity_id}:{candidate.partner_entity_id}"
    question = pick_variant(NEAREST_TEMPLATES, item_key).format(
        target_ref=target_ref,
        target_side=candidate.target_side,
    )
    option_texts = [partner_ref] + distractor_refs
    options, answer_key = shuffled_choice_rows(option_texts, partner_ref, f"nearest_across_seam:{item_key}")
    metadata = {
        "seam_subtype": NEAREST_SUBTYPE,
        "target_ref": target_ref,
        "target_side": candidate.target_side,
        "correct_ref": partner_ref,
        "derived_rotation": {
            "source_scene_id": candidate.source_scene_id,
            "derived_scene_id": derived_scene.scene_id,
            "source_metadata_path": str(candidate.source_metadata_path),
            "derived_metadata_path": str((output_root / 'derived_metadata' / f'{derived_scene.scene_id}.json')),
        },
        "pair_angular_distance_deg": round(candidate.angular_distance_deg, 2),
        "boundary_edge_distance_deg": round(candidate.edge_distance_deg, 2),
        "yaw_shift_deg": round(candidate.yaw_shift_deg, 2),
    }
    quality = (entity_quality(target, derived_scene) + entity_quality(partner, derived_scene)) / 2.0
    diagnostic_slices = list(slices_for_entity(target) + slices_for_entity(partner) + ["derived_rotation", "seam"])
    return (
        benchmark_row(
            scene=derived_scene,
            item_id=f"{derived_scene.scene_id}_seam_continuity_mc_nearest_across_seam_{candidate.target_entity_id}_{candidate.partner_entity_id}",
            question=question,
            answer=answer_key,
            answer_text=partner_ref,
            options=options,
            target_entities=[candidate.target_entity_id, candidate.partner_entity_id, *candidate.distractor_entity_ids],
            metadata=metadata,
            answer_format="4_way_multiple_choice_object_candidates",
            quality_score=quality,
            diagnostic_slices=diagnostic_slices,
            difficulty="hard",
        ),
        None,
    )


def materialize_identity_candidate(candidate: Candidate, output_root: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    scene = load_scene(candidate.source_metadata_path)
    scene.raw["__metadata_path__"] = str(candidate.source_metadata_path)
    suffix = f"seam_identity_yaw_{int(round(candidate.yaw_shift_deg))}_{candidate.target_entity_id}"
    derived_scene = build_rotated_scene_local(scene, yaw_shift_deg=candidate.yaw_shift_deg, suffix=suffix, output_root=output_root)
    if derived_scene is None:
        return None, "source_image_missing_or_rotation_failed"

    target = entity_by_id(derived_scene, candidate.target_entity_id)
    if target is None:
        return None, "missing_rotated_entity"

    base_ref = contextual_entity_ref(derived_scene, target)
    left_ref = f"{base_ref} near the left boundary"
    right_ref = f"{base_ref} near the right boundary"
    item_key = f"{derived_scene.scene_id}:{candidate.target_entity_id}"
    question = pick_variant(IDENTITY_TEMPLATES, item_key).format(
        left_ref=left_ref,
        right_ref=right_ref,
    )
    metadata = {
        "seam_subtype": IDENTITY_SUBTYPE,
        "target_ref": base_ref,
        "left_ref": left_ref,
        "right_ref": right_ref,
        "derived_rotation": {
            "source_scene_id": candidate.source_scene_id,
            "derived_scene_id": derived_scene.scene_id,
            "source_metadata_path": str(candidate.source_metadata_path),
            "derived_metadata_path": str((output_root / 'derived_metadata' / f'{derived_scene.scene_id}.json')),
        },
        "yaw_shift_deg": round(candidate.yaw_shift_deg, 2),
    }
    diagnostic_slices = list(slices_for_entity(target) + ["derived_rotation", "seam"])
    return (
        benchmark_row(
            scene=derived_scene,
            item_id=f"{derived_scene.scene_id}_seam_continuity_mc_identity_place_consistency_{candidate.target_entity_id}",
            question=question,
            answer=IDENTITY_CORRECT_KEY,
            answer_text=IDENTITY_CORRECT_TEXT,
            options=IDENTITY_FIXED_OPTIONS,
            target_entities=[candidate.target_entity_id],
            metadata=metadata,
            answer_format="4_way_multiple_choice_fixed_labels",
            quality_score=entity_quality(target, derived_scene),
            diagnostic_slices=diagnostic_slices,
            difficulty="medium",
        ),
        None,
    )


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    metadata_roots = [Path(path) for path in args.metadata_roots]
    output_jsonl = Path(args.output_jsonl)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_paths = list(iter_metadata_files(metadata_roots))
    if args.max_scenes > 0:
        metadata_paths = metadata_paths[: int(args.max_scenes)]

    nearest_candidates: List[Candidate] = []
    identity_candidates: List[Candidate] = []
    scan_failures: List[Dict[str, Any]] = []

    for metadata_path in metadata_paths:
        try:
            scene = load_scene(metadata_path)
            scene.raw["__metadata_path__"] = str(metadata_path)
        except Exception as exc:
            scan_failures.append({"path": str(metadata_path), "reason": f"scene_load_failed:{type(exc).__name__}"})
            continue
        entities = eligible_entities(scene)
        if len(entities) < 2:
            continue
        nearest = best_nearest_candidate(scene, entities)
        if nearest is not None:
            nearest_candidates.append(nearest)
            continue
        identity = best_identity_candidate(scene, entities)
        if identity is not None:
            identity_candidates.append(identity)

    nearest_candidates.sort(key=candidate_sort_key)
    identity_candidates.sort(key=candidate_sort_key)

    selected_rows: List[Dict[str, Any]] = []
    used_source_scenes: set[str] = set()
    materialization_failures: List[Dict[str, Any]] = []
    selected_counts = {NEAREST_SUBTYPE: 0, IDENTITY_SUBTYPE: 0}

    for candidate in nearest_candidates:
        if selected_counts[NEAREST_SUBTYPE] >= int(args.nearest_target):
            break
        if candidate.source_scene_id in used_source_scenes:
            continue
        row, reason = materialize_nearest_candidate(candidate, output_root)
        if row is None:
            materialization_failures.append(
                {
                    "source_scene_id": candidate.source_scene_id,
                    "subtype": candidate.subtype,
                    "reason": reason or "unknown",
                }
            )
            continue
        selected_rows.append(row)
        used_source_scenes.add(candidate.source_scene_id)
        selected_counts[NEAREST_SUBTYPE] += 1

    for candidate in identity_candidates:
        if selected_counts[IDENTITY_SUBTYPE] >= int(args.identity_target):
            break
        if candidate.source_scene_id in used_source_scenes:
            continue
        row, reason = materialize_identity_candidate(candidate, output_root)
        if row is None:
            materialization_failures.append(
                {
                    "source_scene_id": candidate.source_scene_id,
                    "subtype": candidate.subtype,
                    "reason": reason or "unknown",
                }
            )
            continue
        selected_rows.append(row)
        used_source_scenes.add(candidate.source_scene_id)
        selected_counts[IDENTITY_SUBTYPE] += 1

    selected_rows.sort(key=lambda row: (row["metadata"].get("seam_subtype", ""), row["scene_id"], row["item_id"]))
    write_jsonl(output_jsonl, selected_rows)

    report = {
        "metadata_roots": [str(path) for path in metadata_roots],
        "output_jsonl": str(output_jsonl),
        "output_root": str(output_root),
        "targets": {
            NEAREST_SUBTYPE: int(args.nearest_target),
            IDENTITY_SUBTYPE: int(args.identity_target),
        },
        "counts": {
            "metadata_files_scanned": len(metadata_paths),
            "nearest_candidates": len(nearest_candidates),
            "identity_candidates": len(identity_candidates),
            "selected_nearest_across_seam": selected_counts[NEAREST_SUBTYPE],
            "selected_identity_place_consistency": selected_counts[IDENTITY_SUBTYPE],
            "selected_total": len(selected_rows),
            "scan_failures": len(scan_failures),
            "materialization_failures": len(materialization_failures),
        },
        "scan_failure_examples": scan_failures[:50],
        "materialization_failure_examples": materialization_failures[:50],
    }
    output_jsonl.with_suffix(output_jsonl.suffix + ".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
