#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_GEN_SRC = REPO_ROOT / "data_generation" / "src"
if str(DATA_GEN_SRC) not in sys.path:
    sys.path.insert(0, str(DATA_GEN_SRC))

from erp_data_generation.entity_selector import (  # type: ignore
    choose_relation_partners,
    infer_pole_proximity,
    infer_seam_adjacency,
    score_entity,
    select_anchor_entities,
)
from erp_data_generation.schemas import Entity, SceneMetadata  # type: ignore


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
PANORAMIC_RELATION_LABELS = ["right", "back-right", "opposite", "back-left", "left"]
REORIENTED_RELATION_LABELS = ["right", "back-right", "behind", "back-left", "left"]
SHAPE_FALLBACK_POOL = [
    "round",
    "rectangular",
    "square",
    "oval",
    "cylindrical",
    "spherical",
    "triangular",
    "arched",
]
ROTATION_OPTIONS = [
    ("right", 90),
    ("left", 90),
    ("right", 135),
    ("left", 135),
    ("right", 180),
    ("left", 180),
]
MANUAL_REVIEW_TASKS = {
    "seam_continuity_mc",
    "polar_shape_recovery_mc",
    "relative_3d_position_mc",
}
ANCHOR_LABEL_BLOCKLIST_SUBSTRINGS = (
    "tree",
    "window",
    "leaf",
    "branch",
    "foliage",
    "bush",
    "shrub",
    "plant",
)

TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "referring_grounding_bfov": {
        "ability_group": "spherical_localization_and_panoramic_topology",
        "templates": [
            "Provide the BFOV [yaw, pitch, x_fov, y_fov] of {target_ref} in the full ERP panorama.",
            "What is the BFOV [yaw, pitch, x_fov, y_fov] for {target_ref} in this ERP image?",
        ],
        "answer_format": "bfov_regression",
    },
    "absolute_direction_mc": {
        "ability_group": "spherical_localization_and_panoramic_topology",
        "templates": [
            "In the complete 360 panorama, which direction sector best contains {target_ref}?",
            "Which absolute panorama sector best matches {target_ref} in the ERP image?",
        ],
        "answer_format": "4_way_multiple_choice",
    },
    "relative_direction_mc": {
        "ability_group": "spherical_localization_and_panoramic_topology",
        "templates": [
            "On the panoramic ring, where does {target_ref} fall relative to {reference_ref}?",
            "Around the full panorama ring, what is the angular relation of {target_ref} to {reference_ref}?",
        ],
        "answer_format": "5_way_multiple_choice",
    },
    "camera_rotation_transform_mc": {
        "ability_group": "viewpoint_conditioned_spatial_updating",
        "templates": [
            "If the observer turns {angle_deg} degrees to the {turn_direction}, where would {target_ref} appear in the new view?",
            "After turning {angle_deg} degrees to the {turn_direction}, where does {target_ref} appear in the updated view?",
        ],
        "answer_format": "5_way_multiple_choice",
    },
    "object_conditioned_reorientation_mc": {
        "ability_group": "viewpoint_conditioned_spatial_updating",
        "templates": [
            "Once {facing_ref} is centered as the new front direction, where does {target_ref} lie?",
            "If you turn to face {facing_ref}, where would {target_ref} appear in the reoriented view?",
        ],
        "answer_format": "5_way_multiple_choice",
    },
    "observer_distance_choice": {
        "ability_group": "observer_centered_3d_layout_understanding",
        "templates": [
            "Which of these objects is closest to the current observer in the full panorama?",
            "From the current camera position, which listed object is nearest?",
        ],
        "answer_format": "4_way_multiple_choice",
    },
    "relative_3d_position_mc": {
        "ability_group": "observer_centered_3d_layout_understanding",
        "templates": [
            "In the current camera-centered 3D frame, which relation best describes {entity_a_ref} relative to {entity_b_ref}?",
            "From the current camera viewpoint, which camera-centered 3D relation best matches {entity_a_ref} relative to {entity_b_ref}?",
        ],
        "answer_format": "4_way_multiple_choice",
    },
    "seam_continuity_mc": {
        "ability_group": "erp_representation_understanding",
        "templates": [
            "Which listed object is actually nearest to {target_ref} near the {target_side} image edge?",
            "For the {target_ref} close to the {target_side} image boundary, which candidate is truly the nearest neighbor?",
        ],
        "answer_format": "4_way_multiple_choice",
    },
    "polar_shape_recovery_mc": {
        "ability_group": "erp_representation_understanding",
        "templates": [
            "What is the true shape of {target_ref} in this ERP panorama?",
            "Which shape best matches the real object geometry of {target_ref} in this high-latitude ERP region?",
        ],
        "answer_format": "4_way_multiple_choice",
    },
}


@dataclass
class SceneSideInfo:
    scene_id: str
    group_id: str
    source_id: str
    domain: str
    split_lock: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an ERP-only benchmark pool from benchmark metadata.")
    parser.add_argument("--input-root", required=True, help="Root directory containing many metadata.json files, or a single metadata.json file for local smoke testing.")
    parser.add_argument("--output-dir", required=True, help="Output directory for benchmark artifacts.")
    parser.add_argument("--scene-manifest", default="", help="Optional JSONL with scene_id/group_id/source_id/domain metadata.")
    parser.add_argument("--target-public-per-task", type=int, default=250, help="Target number of selected public benchmark items per task.")
    parser.add_argument("--max-per-scene-per-task", type=int, default=1, help="Maximum selected items from one scene for one task.")
    parser.add_argument("--seed", type=int, default=20260327, help="Random seed for deterministic splitting and selection.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_manifest = load_scene_manifest(args.scene_manifest)
    scene_paths = list(discover_metadata_files(input_root))
    print(json.dumps({"stage": "discover", "num_scene_files": len(scene_paths)}, ensure_ascii=False))

    all_candidates: List[Dict[str, Any]] = []
    scene_infos: Dict[str, SceneSideInfo] = {}
    split_seed = int(args.seed)

    for idx, metadata_path in enumerate(scene_paths, start=1):
        scene = load_scene_metadata(metadata_path)
        info = build_scene_side_info(scene, scene_manifest)
        scene_infos[scene.scene_id] = info
        candidates = generate_scene_candidates(scene)
        all_candidates.extend(candidates)
        if idx % 25 == 0 or idx == len(scene_paths):
            print(json.dumps({"stage": "candidates", "processed_scenes": idx, "candidate_count": len(all_candidates)}, ensure_ascii=False))

    public_selected = select_split_pool(
        all_candidates,
        target_per_task=int(args.target_public_per_task),
        max_per_scene_per_task=args.max_per_scene_per_task,
        seed=split_seed + 11,
    )

    review_queue = [
        row
        for row in all_candidates
        if row.get("requires_manual_review")
    ]

    write_jsonl(output_dir / "candidate_pool.jsonl", all_candidates)
    write_jsonl(output_dir / "review_queue.jsonl", review_queue)
    write_jsonl(output_dir / "benchmark_public.jsonl", public_selected)
    write_jsonl(output_dir / "benchmark_public_prompts.jsonl", [strip_answers(row) for row in public_selected])
    write_jsonl(output_dir / "benchmark_public_references.jsonl", public_selected)

    summary = build_summary(
        scene_infos=scene_infos,
        all_candidates=all_candidates,
        public_selected=public_selected,
        review_queue=review_queue,
        target_public_per_task=int(args.target_public_per_task),
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "done", "output_dir": str(output_dir), "summary_path": str(output_dir / "summary.json")}, ensure_ascii=False))
    return 0


def discover_metadata_files(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            if filename == "metadata.json":
                yield Path(dirpath) / filename


def load_scene_manifest(path_str: str) -> Dict[str, SceneSideInfo]:
    if not path_str:
        return {}
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Scene manifest not found: {path}")
    mapping: Dict[str, SceneSideInfo] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            scene_id = str(row["scene_id"])
            mapping[scene_id] = SceneSideInfo(
                scene_id=scene_id,
                group_id=str(row.get("group_id") or scene_id),
                source_id=str(row.get("source_id") or "unknown"),
                domain=str(row.get("domain") or "unknown"),
                split_lock=str(row.get("split_lock") or ""),
            )
    return mapping


def load_scene_metadata(path: Path) -> SceneMetadata:
    data = json.loads(path.read_text(encoding="utf-8"))
    return SceneMetadata.from_dict(data)


def build_scene_side_info(scene: SceneMetadata, manifest: Dict[str, SceneSideInfo]) -> SceneSideInfo:
    if scene.scene_id in manifest:
        return manifest[scene.scene_id]
    tags = scene.scene_global_tags or {}
    domain = first_nonempty_str(
        tags.get("domain"),
        tags.get("scene_domain"),
        tags.get("scene_type"),
        "unknown",
    )
    source_id = first_nonempty_str(scene.raw.get("source_id"), scene.raw.get("provider"), "unknown")
    group_id = first_nonempty_str(scene.raw.get("group_id"), scene.raw.get("capture_id"), scene.scene_id)
    return SceneSideInfo(
        scene_id=scene.scene_id,
        group_id=group_id,
        source_id=source_id,
        domain=domain,
        split_lock="",
    )


def generate_scene_candidates(scene: SceneMetadata) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    label_counts = Counter(entity.label for entity in scene.entities)
    anchors = [item for item in select_anchor_entities(scene, max_anchors=8) if benchmark_anchor_eligible(item["entity"])]

    used_item_ids: set[str] = set()
    for anchor_index, anchor_payload in enumerate(anchors):
        anchor = anchor_payload["entity"]
        quality = float(score_entity(anchor, label_counts, scene))
        for row in [
            build_referring_grounding_bfov(scene, anchor, anchors, anchor_index, quality),
            build_absolute_direction_mc(scene, anchor, anchor_index, quality),
            build_polar_shape_recovery(scene, anchor, anchor_index, quality),
        ]:
            if row and row["item_id"] not in used_item_ids:
                candidates.append(row)
                used_item_ids.add(row["item_id"])

        for row in build_seam_continuity_items(scene, anchor, anchors, anchor_index, quality):
            if row and row["item_id"] not in used_item_ids:
                candidates.append(row)
                used_item_ids.add(row["item_id"])

        partners = choose_relation_partners(anchor, scene, max_partners=4)
        for partner_index, partner_payload in enumerate(partners):
            partner = partner_payload["entity"]
            for row in [
                build_relative_direction_mc(scene, anchor, partner, anchor_index, partner_index, quality),
                build_camera_rotation_transform_mc(scene, anchor, anchor_index, partner_index, quality),
                build_object_conditioned_reorientation_mc(scene, anchor, partner, anchor_index, partner_index, quality),
                build_relative_3d_position_mc(scene, anchor, partner, anchor_index, partner_index, quality),
            ]:
                if row and row["item_id"] not in used_item_ids:
                    candidates.append(row)
                    used_item_ids.add(row["item_id"])

        observer_choice = build_observer_distance_choice(scene, anchors, anchor_index)
        if observer_choice and observer_choice["item_id"] not in used_item_ids:
            candidates.append(observer_choice)
            used_item_ids.add(observer_choice["item_id"])

    return candidates


def benchmark_entity_eligible(entity: Entity) -> bool:
    if not entity.verified_semantics:
        return False
    if float(entity.confidence or 0.0) < 0.55:
        return False
    if float(entity.area_ratio or 0.0) <= 0.0004:
        return False
    return bool(entity.resolved_bfov)


def benchmark_anchor_eligible(entity: Entity) -> bool:
    if not benchmark_entity_eligible(entity):
        return False
    label = normalize_phrase(entity.label)
    if any(token in label for token in ANCHOR_LABEL_BLOCKLIST_SUBSTRINGS):
        return False
    ref = normalize_phrase(descriptive_entity_ref(entity))
    if any(token in ref for token in ANCHOR_LABEL_BLOCKLIST_SUBSTRINGS):
        return False
    return True


def descriptive_entity_ref(entity: Entity) -> str:
    return (
        entity.semantic.reground_query
        or entity.semantic.caption_brief
        or f"the {entity.label}"
    )


def normalized_bbox_1000(entity: Entity, scene: SceneMetadata) -> Optional[Tuple[int, int, int, int]]:
    if len(entity.bbox_erp) != 4 or not scene.erp_width or not scene.erp_height:
        return None
    x1, y1, x2, y2 = entity.bbox_erp

    def clamp_round(value: float) -> int:
        return max(0, min(1000, int(round(value))))

    return (
        clamp_round((float(x1) / float(scene.erp_width)) * 1000.0),
        clamp_round((float(y1) / float(scene.erp_height)) * 1000.0),
        clamp_round((float(x2) / float(scene.erp_width)) * 1000.0),
        clamp_round((float(y2) / float(scene.erp_height)) * 1000.0),
    )


def safe_entity_ref(entity: Entity, scene: SceneMetadata, key: str) -> str:
    label = normalize_phrase(entity.label) or "object"
    prefer_box = (stable_hash(f"safe_ref:{key}:{entity.entity_id}") % 2) == 0
    bbox = normalized_bbox_1000(entity, scene)
    bfov = entity.resolved_bfov

    if prefer_box and bbox is not None:
        x1, y1, x2, y2 = bbox
        return f"the {label} at box [{x1}, {y1}, {x2}, {y2}]"
    if (not prefer_box) and bfov is not None:
        yaw, pitch, x_fov, y_fov = bfov
        return f"the {label} at BFOV [yaw={yaw:.1f}, pitch={pitch:.1f}, x_fov={x_fov:.1f}, y_fov={y_fov:.1f}]"
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        return f"the {label} at box [{x1}, {y1}, {x2}, {y2}]"
    if bfov is not None:
        yaw, pitch, x_fov, y_fov = bfov
        return f"the {label} at BFOV [yaw={yaw:.1f}, pitch={pitch:.1f}, x_fov={x_fov:.1f}, y_fov={y_fov:.1f}]"
    return f"the {label}"


def entity_ref(entity: Entity, scene: Optional[SceneMetadata] = None, key: str = "") -> str:
    if scene is not None and key:
        return safe_entity_ref(entity, scene, key)
    return descriptive_entity_ref(entity)


def normalize_phrase(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def first_nonempty_str(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "none":
            return text
    return ""


def wrapped_delta_deg(delta: float) -> float:
    delta = ((delta + 180.0) % 360.0) - 180.0
    if delta == -180.0:
        return 180.0
    return delta


def yaw_deg_360(entity: Entity) -> float:
    bfov = entity.resolved_bfov
    if bfov is not None:
        return float(bfov[0]) % 360.0
    return float(entity.lon_deg) % 360.0


def pitch_deg(entity: Entity) -> float:
    bfov = entity.resolved_bfov
    if bfov is not None:
        return float(bfov[1])
    return -float(entity.lat_deg)


def absolute_sector_8way(entity: Entity) -> str:
    yaw = yaw_deg_360(entity)
    idx = int(((yaw % 360.0) + 22.5) // 45.0) % 8
    return ABSOLUTE_SECTORS_8[idx]


def absolute_sector_margin(entity: Entity) -> float:
    yaw = yaw_deg_360(entity) % 45.0
    return min(abs(yaw - 22.5), abs(yaw - 0.0), abs(yaw - 45.0))


def panoramic_relation_from_delta(delta_yaw: float, *, opposite_label: str) -> Optional[str]:
    if abs(delta_yaw) < 15.0:
        return None
    if 15.0 <= delta_yaw < 90.0:
        return "right"
    if 90.0 <= delta_yaw < 150.0:
        return "back-right"
    if delta_yaw >= 150.0 or delta_yaw < -150.0:
        return opposite_label
    if -150.0 <= delta_yaw < -90.0:
        return "back-left"
    return "left"


def panoramic_relation(reference: Entity, target: Entity, *, opposite_label: str) -> Optional[str]:
    delta = wrapped_delta_deg(yaw_deg_360(target) - yaw_deg_360(reference))
    return panoramic_relation_from_delta(delta, opposite_label=opposite_label)


def camera_rotation_relation(entity: Entity, rotation_direction: str, angle_deg: int) -> Optional[str]:
    yaw = yaw_deg_360(entity)
    observer_forward = float(angle_deg) if rotation_direction == "right" else -float(angle_deg)
    delta = wrapped_delta_deg(yaw - (observer_forward % 360.0))
    return panoramic_relation_from_delta(delta, opposite_label="behind")


def closest_boundary_margin(delta_yaw: float, boundaries: Sequence[float]) -> float:
    return min(abs(delta_yaw - boundary) for boundary in boundaries)


def shape_value(entity: Entity) -> Optional[str]:
    attrs = entity.semantic.attributes or {}
    raw = attrs.get("shape")
    if raw is None:
        return None
    value = normalize_phrase(str(raw))
    return value if value else None


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def pick_template(task_id: str, key: str) -> str:
    spec = TASK_SPECS[task_id]
    variants = spec["templates"]
    return variants[stable_hash(f"{task_id}:{key}") % len(variants)]


def pick_variant(variants: Sequence[str], key: str) -> str:
    variants = list(variants)
    return variants[stable_hash(key) % len(variants)]


def option_key(index: int) -> str:
    return chr(ord("A") + index)


def choice_rows(options: Sequence[str]) -> List[Dict[str, str]]:
    return [{"key": option_key(i), "text": value} for i, value in enumerate(options)]


def shuffled_choice_rows(options: Sequence[str], correct_text: str, seed_key: str) -> Tuple[List[Dict[str, str]], str]:
    ordered = list(options)
    rotation = stable_hash(seed_key) % len(ordered)
    ordered = ordered[rotation:] + ordered[:rotation]
    choices = choice_rows(ordered)
    answer_key = label_to_choice_key(ordered, correct_text)
    return choices, answer_key


def difficulty_from_margin(margin: float) -> str:
    if margin >= 30.0:
        return "easy"
    if margin >= 16.0:
        return "medium"
    return "hard"


def compact_for_relative_3d(entity: Entity) -> bool:
    bfov = entity.resolved_bfov
    if bfov is None:
        return False
    x_fov = abs(float(bfov[2]))
    y_fov = abs(float(bfov[3]))
    return x_fov <= 45.0 and y_fov <= 45.0 and (x_fov * y_fov) <= 1800.0


def approx_axis_radius(entity: Entity, axis: str) -> float:
    bfov = entity.resolved_bfov
    depth = entity.entity_center_depth
    if bfov is None or depth is None:
        return 0.0
    fov = float(bfov[2] if axis == "x" else bfov[3])
    if fov <= 0.0:
        return 0.0
    return float(depth) * math.tan(math.radians(fov / 2.0))


def axis_clear_x(entity_a: Entity, entity_b: Entity, dx: float) -> bool:
    return abs(dx) >= max(0.35, approx_axis_radius(entity_a, "x") + approx_axis_radius(entity_b, "x"))


def axis_clear_y(entity_a: Entity, entity_b: Entity, dy: float) -> bool:
    return abs(dy) >= max(0.25, approx_axis_radius(entity_a, "y") + approx_axis_radius(entity_b, "y"))


def relative_3d_relation(entity_a: Entity, entity_b: Entity) -> Tuple[str, List[str]]:
    xyz_a = entity_a.erp_consistent_xyz_camera
    xyz_b = entity_b.erp_consistent_xyz_camera
    if xyz_a is None or xyz_b is None:
        return "", []
    dx = float(xyz_a[0]) - float(xyz_b[0])
    dy = float(xyz_a[1]) - float(xyz_b[1])
    dz = float(xyz_a[2]) - float(xyz_b[2])
    parts: List[str] = []
    if axis_clear_x(entity_a, entity_b, dx):
        parts.append("right of" if dx > 0 else "left of")
    if axis_clear_y(entity_a, entity_b, dy):
        parts.append("above" if dy > 0 else "below")
    if abs(dz) >= 0.6:
        parts.append("in front of" if dz > 0 else "behind")
    if not parts:
        return "", []
    if len(parts) > 2:
        return "", []
    if len(parts) == 1:
        return parts[0], parts
    return f"{parts[0]} and {parts[1]}", parts


def relative_3d_choices(entity_a: Entity, entity_b: Entity, answer: str, parts: Sequence[str]) -> List[str]:
    xyz_a = entity_a.erp_consistent_xyz_camera
    xyz_b = entity_b.erp_consistent_xyz_camera
    if xyz_a is None or xyz_b is None:
        pool = [answer, "left of", "right of", "behind"]
        return dedupe_keep_order(pool)[:4]

    dx = float(xyz_a[0]) - float(xyz_b[0])
    dy = float(xyz_a[1]) - float(xyz_b[1])
    dz = float(xyz_a[2]) - float(xyz_b[2])
    axis_pairs = {
        "x": ("right of", "left of") if dx > 0 else ("left of", "right of"),
        "y": ("above", "below") if dy > 0 else ("below", "above"),
        "z": ("in front of", "behind") if dz > 0 else ("behind", "in front of"),
    }
    active_axes: List[str] = []
    if axis_clear_x(entity_a, entity_b, dx):
        active_axes.append("x")
    if axis_clear_y(entity_a, entity_b, dy):
        active_axes.append("y")
    if abs(dz) >= 0.6:
        active_axes.append("z")

    options = [answer]
    for axis in active_axes:
        flipped = []
        for candidate_axis in active_axes:
            flipped.append(axis_pairs[candidate_axis][1] if candidate_axis == axis else axis_pairs[candidate_axis][0])
        options.append(join_relations(flipped))
    fallback_pool = ["left of", "right of", "above", "below", "in front of", "behind"]
    for relation in fallback_pool:
        if relation not in options:
            options.append(relation)
        if len(options) >= 4:
            break
    return dedupe_keep_order(options)[:4]


def join_relations(parts: Sequence[str]) -> str:
    parts = [item for item in parts if item]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    if not parts:
        return ""
    return f"{parts[0]}, {parts[1]}, and {parts[2]}"


def dedupe_keep_order(items: Sequence[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def bfov_text(entity: Entity) -> str:
    bfov = entity.resolved_bfov
    if bfov is None:
        return "[unknown]"
    return f"[yaw={bfov[0]:.1f}, pitch={bfov[1]:.1f}, x_fov={bfov[2]:.1f}, y_fov={bfov[3]:.1f}]"


def benchmark_item(
    *,
    scene: SceneMetadata,
    task_id: str,
    item_id: str,
    question: str,
    answer: Any,
    answer_text: str,
    options: Optional[List[Dict[str, str]]] = None,
    target_entities: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    image_paths: Optional[List[str]] = None,
    difficulty: str = "medium",
    quality_score: float = 0.0,
    diagnostic_slices: Optional[List[str]] = None,
    requires_manual_review: bool = False,
    review_notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    spec = TASK_SPECS[task_id]
    return {
        "item_id": item_id,
        "scene_id": scene.scene_id,
        "task_id": task_id,
        "ability_group": spec["ability_group"],
        "answer_format": spec["answer_format"],
        "image_path": scene.erp_image_path,
        "image_paths": image_paths or [],
        "question": question,
        "answer": answer,
        "answer_text": answer_text,
        "options": options or [],
        "target_entities": target_entities or [],
        "metadata": metadata or {},
        "difficulty": difficulty,
        "quality_score": round(float(quality_score), 4),
        "diagnostic_slices": sorted(set(diagnostic_slices or [])),
        "requires_manual_review": bool(requires_manual_review),
        "review_notes": review_notes or [],
    }


def build_referring_grounding_bfov(scene: SceneMetadata, target: Entity, anchors: Sequence[Dict[str, Any]], anchor_index: int, quality: float) -> Optional[Dict[str, Any]]:
    if target.resolved_bfov is None:
        return None
    target_ref = entity_ref(target, scene, f"{scene.scene_id}:referring_grounding_bfov:{target.entity_id}:target")
    question = pick_template("referring_grounding_bfov", f"{scene.scene_id}:{target.entity_id}").format(target_ref=target_ref)
    answer_text = bfov_text(target)
    return benchmark_item(
        scene=scene,
        task_id="referring_grounding_bfov",
        item_id=f"{scene.scene_id}_referring_grounding_bfov_{target.entity_id}",
        question=question,
        answer=list(target.resolved_bfov),
        answer_text=answer_text,
        options=[],
        target_entities=[target.entity_id],
        metadata={
            "target_ref": target_ref,
            "target_bfov": list(target.resolved_bfov),
        },
        difficulty="medium",
        quality_score=quality,
        diagnostic_slices=slices_for_entity(target),
    )


def build_absolute_direction_mc(scene: SceneMetadata, target: Entity, anchor_index: int, quality: float) -> Optional[Dict[str, Any]]:
    sector = absolute_sector_8way(target)
    margin = absolute_sector_margin(target)
    if margin < 8.0:
        return None
    neighbors = sector_distractors(sector)
    options = [sector] + neighbors[:3]
    choices = choice_rows(options)
    target_ref = entity_ref(target, scene, f"{scene.scene_id}:absolute_direction_mc:{target.entity_id}:target")
    question = pick_template("absolute_direction_mc", f"{scene.scene_id}:{target.entity_id}").format(target_ref=target_ref)
    return benchmark_item(
        scene=scene,
        task_id="absolute_direction_mc",
        item_id=f"{scene.scene_id}_absolute_direction_mc_{target.entity_id}",
        question=question,
        answer=choices[0]["key"],
        answer_text=sector,
        options=choices,
        target_entities=[target.entity_id],
        metadata={
            "target_ref": target_ref,
            "yaw_deg": round(yaw_deg_360(target), 1),
            "sector": sector,
            "sector_margin_deg": round(margin, 2),
        },
        difficulty=difficulty_from_margin(margin),
        quality_score=quality,
        diagnostic_slices=slices_for_entity(target),
    )


def sector_distractors(correct: str) -> List[str]:
    idx = ABSOLUTE_SECTORS_8.index(correct)
    order = [ABSOLUTE_SECTORS_8[(idx - 1) % 8], ABSOLUTE_SECTORS_8[(idx + 1) % 8], ABSOLUTE_SECTORS_8[(idx + 4) % 8], ABSOLUTE_SECTORS_8[(idx + 2) % 8]]
    return dedupe_keep_order(order)


def seam_contact_side(entity: Entity, scene: SceneMetadata) -> Optional[str]:
    if len(entity.bbox_erp) != 4 or not scene.erp_width:
        return None
    x1, _, x2, _ = entity.bbox_erp
    margin = max(12.0, scene.erp_width * 0.03)
    left_touch = x1 <= margin
    right_touch = x2 >= (scene.erp_width - margin)
    if left_touch and right_touch:
        return "left" if stable_hash(f"{scene.scene_id}:{entity.entity_id}:seam_side") % 2 == 0 else "right"
    if left_touch:
        return "left"
    if right_touch:
        return "right"
    return None


def seam_primary_side(entity: Entity, scene: SceneMetadata) -> Optional[str]:
    side = seam_contact_side(entity, scene)
    if side is not None:
        return side
    if not infer_seam_adjacency(entity, scene):
        return None
    return "left" if yaw_deg_360(entity) >= 180.0 else "right"


def wrap_yaw_gap_deg(entity_a: Entity, entity_b: Entity) -> float:
    return abs(wrapped_delta_deg(yaw_deg_360(entity_b) - yaw_deg_360(entity_a)))


def flat_x_gap_px(entity_a: Entity, entity_b: Entity) -> float:
    return abs(float(entity_a.center_xy[0]) - float(entity_b.center_xy[0]))


SEAM_RELATION_OPTIONS = [
    "adjacent across the boundary",
    "opposite in the scene",
    "far apart on the same side",
    "cannot determine",
]
SEAM_DEDUP_OPTIONS = [
    "one continuous object",
    "two different objects",
    "cannot determine",
]
SEAM_STRUCTURE_OPTIONS = [
    "one continuous structure",
    "two different structures",
    "a reflection",
    "cannot determine",
]
SEAM_SAME_ENTITY_OPTIONS = [
    "same object at different image positions",
    "same object at the same image position",
    "different objects at different image positions",
    "different objects at the same image position",
]
STRUCTURAL_LABEL_HINTS = {
    "wall",
    "table",
    "desk",
    "counter",
    "countertop",
    "road",
    "floor",
    "ground",
    "ceiling",
    "railing",
    "guardrail",
    "handrail",
    "barrier",
    "ledge",
    "curb",
}
STRUCTURAL_LABEL_SUBSTRINGS = (
    "wall",
    "table",
    "desk",
    "counter",
    "countertop",
    "road",
    "floor",
    "ground",
    "ceiling",
    "railing",
    "guardrail",
    "handrail",
)
SEAM_SUBTYPE_TEMPLATES: Dict[str, List[str]] = {
    "nearest_neighbor": [
        "Which listed object is actually nearest to {target_ref} near the {target_side} image edge?",
    ],
    "relative_direction": [
        "What is the relation of {neighbor_ref} relative to {target_ref} across the left-right image boundary?",
    ],
    "dedup_count": [
        "The left-edge and right-edge visible parts of {target_ref} should be counted as:",
    ],
    "structure_continuity": [
        "For the {target_ref} touching both image sides, which explanation is more reasonable?",
    ],
    "same_entity_judgement": [
        "The left-edge and right-edge appearances of {target_ref} are best described as:",
    ],
}


def seam_wrap_candidate_sets(scene: SceneMetadata, target: Entity) -> Optional[Dict[str, Any]]:
    if not scene.erp_width:
        return None
    target_side = seam_contact_side(target, scene)
    if target_side is None:
        return None
    opposite_side = "right" if target_side == "left" else "left"
    width = float(scene.erp_width)

    correct_pool: List[Tuple[float, float, float, Entity]] = []
    lure_pool: List[Tuple[float, float, float, Entity]] = []
    distractor_pool: List[Tuple[float, float, float, Entity]] = []

    label_counts = Counter(entity.label for entity in scene.entities)
    for entity in scene.entities:
        if entity.entity_id == target.entity_id:
            continue
        if not benchmark_entity_eligible(entity):
            continue
        gap = wrap_yaw_gap_deg(target, entity)
        flat_gap = flat_x_gap_px(target, entity)
        score = float(score_entity(entity, label_counts, scene))
        same_side = seam_contact_side(entity, scene)

        if same_side == opposite_side and gap <= 15.0 and flat_gap >= width * 0.65:
            correct_pool.append((gap, -flat_gap, -score, entity))
            continue
        if flat_gap <= width * 0.25 and gap >= 35.0:
            lure_pool.append((flat_gap, gap, -score, entity))
            continue
        if gap >= 35.0:
            distractor_pool.append((gap, -score, flat_gap, entity))

    if not correct_pool or not lure_pool or len(distractor_pool) < 2:
        return None

    correct_pool.sort(key=lambda item: (item[0], item[1], item[2], item[3].entity_id))
    lure_pool.sort(key=lambda item: (item[0], item[1], item[2], item[3].entity_id))
    distractor_pool.sort(key=lambda item: (item[0], item[1], item[2], item[3].entity_id))

    correct = correct_pool[0][3]
    lure = lure_pool[0][3]
    distractors = [item[3] for item in distractor_pool[:2]]
    return {
        "target_side": target_side,
        "correct": correct,
        "correct_gap": correct_pool[0][0],
        "correct_flat_gap": -correct_pool[0][1],
        "lure": lure,
        "lure_gap": lure_pool[0][1],
        "lure_flat_gap": lure_pool[0][0],
        "distractors": distractors,
    }


def seam_boundary_direction(scene: SceneMetadata, entity: Entity) -> str:
    side = seam_contact_side(entity, scene)
    if side:
        return side
    return seam_primary_side(entity, scene) or "boundary"


def seam_structure_like(entity: Entity) -> bool:
    label = normalize_phrase(entity.label)
    if label in STRUCTURAL_LABEL_HINTS:
        return True
    return any(token in label for token in STRUCTURAL_LABEL_SUBSTRINGS)


def build_seam_nearest_neighbor_mc(scene: SceneMetadata, target: Entity, quality: float) -> Optional[Dict[str, Any]]:
    bundles = seam_wrap_candidate_sets(scene, target)
    if bundles is None:
        return None
    correct = bundles["correct"]
    lure = bundles["lure"]
    distractors = bundles["distractors"]
    target_side = bundles["target_side"]

    option_entities = [correct, lure] + distractors
    item_key = f"{scene.scene_id}:seam_nearest_neighbor:{target.entity_id}:{correct.entity_id}:{lure.entity_id}"
    target_ref = entity_ref(target, scene, f"{item_key}:target")
    correct_ref = entity_ref(correct, scene, f"{item_key}:correct")
    lure_ref = entity_ref(lure, scene, f"{item_key}:lure")
    distractor_refs = [entity_ref(entity, scene, f"{item_key}:distractor:{idx}:{entity.entity_id}") for idx, entity in enumerate(distractors)]
    option_texts = [correct_ref, lure_ref] + distractor_refs
    question = pick_variant(
        SEAM_SUBTYPE_TEMPLATES["nearest_neighbor"],
        f"{item_key}:question",
    ).format(
        target_ref=target_ref,
        target_side=target_side,
    )
    choices, answer_key = shuffled_choice_rows(
        option_texts,
        correct_ref,
        f"seam_continuity_mc:nearest:{item_key}",
    )
    return benchmark_item(
        scene=scene,
        task_id="seam_continuity_mc",
        item_id=f"{scene.scene_id}_seam_continuity_mc_nearest_neighbor_{target.entity_id}_{correct.entity_id}",
        question=question,
        answer=answer_key,
        answer_text=correct_ref,
        options=choices,
        target_entities=[target.entity_id, correct.entity_id, lure.entity_id] + [entity.entity_id for entity in distractors],
        metadata={
            "seam_subtype": "nearest_neighbor",
            "target_ref": target_ref,
            "target_side": target_side,
            "correct_ref": correct_ref,
            "correct_wrap_gap_deg": round(float(bundles["correct_gap"]), 2),
            "correct_flat_x_gap_px": round(float(bundles["correct_flat_gap"]), 2),
            "lure_ref": lure_ref,
            "lure_wrap_gap_deg": round(float(bundles["lure_gap"]), 2),
            "lure_flat_x_gap_px": round(float(bundles["lure_flat_gap"]), 2),
        },
        difficulty="hard",
        quality_score=(quality + float(score_entity(correct, Counter(e.label for e in scene.entities), scene))) / 2.0,
        diagnostic_slices=dedupe_keep_order(slices_for_entity(target) + slices_for_entity(correct) + ["seam"]),
        requires_manual_review=True,
        review_notes=[
            "Verify that the correct candidate is truly the nearest wrap-around neighbor and that the lure only looks close in the flat ERP rectangle."
        ],
    )


def build_seam_relative_direction_mc(scene: SceneMetadata, target: Entity, quality: float) -> Optional[Dict[str, Any]]:
    bundles = seam_wrap_candidate_sets(scene, target)
    if bundles is None:
        return None
    correct = bundles["correct"]
    item_key = f"{scene.scene_id}:seam_relative_direction:{target.entity_id}:{correct.entity_id}"
    neighbor_ref = entity_ref(correct, scene, f"{item_key}:neighbor")
    target_ref = entity_ref(target, scene, f"{item_key}:target")
    question = pick_variant(
        SEAM_SUBTYPE_TEMPLATES["relative_direction"],
        f"{item_key}:question",
    ).format(
        neighbor_ref=neighbor_ref,
        target_ref=target_ref,
    )
    choices = choice_rows(SEAM_RELATION_OPTIONS)
    answer_text = "adjacent across the boundary"
    return benchmark_item(
        scene=scene,
        task_id="seam_continuity_mc",
        item_id=f"{scene.scene_id}_seam_continuity_mc_relative_direction_{target.entity_id}_{correct.entity_id}",
        question=question,
        answer=label_to_choice_key(SEAM_RELATION_OPTIONS, answer_text),
        answer_text=answer_text,
        options=choices,
        target_entities=[target.entity_id, correct.entity_id],
        metadata={
            "seam_subtype": "relative_direction",
            "target_ref": target_ref,
            "neighbor_ref": neighbor_ref,
            "target_side": bundles["target_side"],
            "neighbor_side": seam_boundary_direction(scene, correct),
            "wrap_gap_deg": round(float(bundles["correct_gap"]), 2),
        },
        difficulty="medium",
        quality_score=(quality + float(score_entity(correct, Counter(e.label for e in scene.entities), scene))) / 2.0,
        diagnostic_slices=dedupe_keep_order(slices_for_entity(target) + slices_for_entity(correct) + ["seam"]),
        requires_manual_review=True,
        review_notes=[
            "Verify that the pair is genuinely close only under wrap-around and that 'adjacent across the boundary' is the correct relation."
        ],
    )


def build_seam_dedup_count_mc(scene: SceneMetadata, target: Entity, quality: float) -> Optional[Dict[str, Any]]:
    side = seam_contact_side(target, scene)
    if side is None:
        return None
    if not bool(target.seam_crossing_flag):
        return None
    item_key = f"{scene.scene_id}:seam_dedup_count:{target.entity_id}"
    target_ref = entity_ref(target, scene, f"{item_key}:target")
    question = pick_variant(
        SEAM_SUBTYPE_TEMPLATES["dedup_count"],
        f"{item_key}:question",
    ).format(
        target_ref=target_ref,
    )
    choices = choice_rows(SEAM_DEDUP_OPTIONS)
    answer_text = "one continuous object"
    return benchmark_item(
        scene=scene,
        task_id="seam_continuity_mc",
        item_id=f"{scene.scene_id}_seam_continuity_mc_dedup_count_{target.entity_id}",
        question=question,
        answer=label_to_choice_key(SEAM_DEDUP_OPTIONS, answer_text),
        answer_text=answer_text,
        options=choices,
        target_entities=[target.entity_id],
        metadata={
            "seam_subtype": "dedup_count",
            "target_ref": target_ref,
            "target_side": side,
            "seam_crossing_flag": bool(target.seam_crossing_flag),
        },
        difficulty="medium",
        quality_score=quality,
        diagnostic_slices=dedupe_keep_order(slices_for_entity(target) + ["seam"]),
        requires_manual_review=True,
        review_notes=[
            "Verify that the entity truly wraps across the seam and should be counted once rather than twice."
        ],
    )


def build_seam_structure_continuity_mc(scene: SceneMetadata, target: Entity, quality: float) -> Optional[Dict[str, Any]]:
    side = seam_contact_side(target, scene)
    if side is None or not bool(target.seam_crossing_flag):
        return None
    if not seam_structure_like(target):
        return None
    item_key = f"{scene.scene_id}:seam_structure_continuity:{target.entity_id}"
    target_ref = entity_ref(target, scene, f"{item_key}:target")
    question = pick_variant(
        SEAM_SUBTYPE_TEMPLATES["structure_continuity"],
        f"{item_key}:question",
    ).format(
        target_ref=target_ref,
    )
    choices = choice_rows(SEAM_STRUCTURE_OPTIONS)
    answer_text = "one continuous structure"
    return benchmark_item(
        scene=scene,
        task_id="seam_continuity_mc",
        item_id=f"{scene.scene_id}_seam_continuity_mc_structure_continuity_{target.entity_id}",
        question=question,
        answer=label_to_choice_key(SEAM_STRUCTURE_OPTIONS, answer_text),
        answer_text=answer_text,
        options=choices,
        target_entities=[target.entity_id],
        metadata={
            "seam_subtype": "structure_continuity",
            "target_ref": target_ref,
            "target_label": target.label,
            "target_side": side,
            "seam_crossing_flag": bool(target.seam_crossing_flag),
        },
        difficulty="hard",
        quality_score=quality,
        diagnostic_slices=dedupe_keep_order(slices_for_entity(target) + ["seam"]),
        requires_manual_review=True,
        review_notes=[
            "Verify that this is a structure-like category where seam continuity is a meaningful interpretation."
        ],
    )


def build_seam_same_entity_mc(scene: SceneMetadata, target: Entity, quality: float) -> Optional[Dict[str, Any]]:
    side = seam_contact_side(target, scene)
    if side is None or not bool(target.seam_crossing_flag):
        return None
    item_key = f"{scene.scene_id}:seam_same_entity:{target.entity_id}"
    target_ref = entity_ref(target, scene, f"{item_key}:target")
    question = pick_variant(
        SEAM_SUBTYPE_TEMPLATES["same_entity_judgement"],
        f"{item_key}:question",
    ).format(
        target_ref=target_ref,
    )
    choices = choice_rows(SEAM_SAME_ENTITY_OPTIONS)
    answer_text = "same object at different image positions"
    return benchmark_item(
        scene=scene,
        task_id="seam_continuity_mc",
        item_id=f"{scene.scene_id}_seam_continuity_mc_same_entity_{target.entity_id}",
        question=question,
        answer=label_to_choice_key(SEAM_SAME_ENTITY_OPTIONS, answer_text),
        answer_text=answer_text,
        options=choices,
        target_entities=[target.entity_id],
        metadata={
            "seam_subtype": "same_entity_judgement",
            "target_ref": target_ref,
            "target_side": side,
            "seam_crossing_flag": bool(target.seam_crossing_flag),
        },
        difficulty="medium",
        quality_score=quality,
        diagnostic_slices=dedupe_keep_order(slices_for_entity(target) + ["seam"]),
        requires_manual_review=True,
        review_notes=[
            "Verify that the two boundary appearances truly come from one seam-crossing entity rather than two different nearby instances."
        ],
    )


def build_seam_continuity_items(
    scene: SceneMetadata,
    target: Entity,
    anchors: Sequence[Dict[str, Any]],
    anchor_index: int,
    quality: float,
) -> List[Dict[str, Any]]:
    rows: List[Optional[Dict[str, Any]]] = [
        build_seam_nearest_neighbor_mc(scene, target, quality),
        build_seam_relative_direction_mc(scene, target, quality),
        build_seam_dedup_count_mc(scene, target, quality),
        build_seam_structure_continuity_mc(scene, target, quality),
        build_seam_same_entity_mc(scene, target, quality),
    ]
    return [row for row in rows if row is not None]


def build_relative_direction_mc(scene: SceneMetadata, reference: Entity, target: Entity, anchor_index: int, partner_index: int, quality: float) -> Optional[Dict[str, Any]]:
    delta = wrapped_delta_deg(yaw_deg_360(target) - yaw_deg_360(reference))
    relation = panoramic_relation_from_delta(delta, opposite_label="opposite")
    if relation is None:
        return None
    margin = closest_boundary_margin(abs(delta), [15.0, 90.0, 150.0, 180.0])
    if margin < 8.0:
        return None
    item_key = f"{scene.scene_id}:relative_direction_mc:{reference.entity_id}:{target.entity_id}"
    reference_ref = entity_ref(reference, scene, f"{item_key}:reference")
    target_ref = entity_ref(target, scene, f"{item_key}:target")
    choices = choice_rows(PANORAMIC_RELATION_LABELS)
    question = pick_template("relative_direction_mc", f"{scene.scene_id}:{reference.entity_id}:{target.entity_id}").format(
        reference_ref=reference_ref,
        target_ref=target_ref,
    )
    return benchmark_item(
        scene=scene,
        task_id="relative_direction_mc",
        item_id=f"{scene.scene_id}_relative_direction_mc_{reference.entity_id}_{target.entity_id}",
        question=question,
        answer=label_to_choice_key(PANORAMIC_RELATION_LABELS, relation),
        answer_text=relation,
        options=choices,
        target_entities=[reference.entity_id, target.entity_id],
        metadata={
            "reference_ref": reference_ref,
            "target_ref": target_ref,
            "reference_yaw_deg": round(yaw_deg_360(reference), 1),
            "target_yaw_deg": round(yaw_deg_360(target), 1),
            "delta_yaw_deg": round(delta, 2),
        },
        difficulty=difficulty_from_margin(margin),
        quality_score=(quality + float(score_entity(target, Counter(e.label for e in scene.entities), scene))) / 2.0,
        diagnostic_slices=sorted(set(slices_for_entity(reference) + slices_for_entity(target))),
    )


def build_camera_rotation_transform_mc(scene: SceneMetadata, target: Entity, anchor_index: int, partner_index: int, quality: float) -> Optional[Dict[str, Any]]:
    rotation_direction, angle_deg = ROTATION_OPTIONS[stable_hash(f"{scene.scene_id}:{target.entity_id}:rotation") % len(ROTATION_OPTIONS)]
    relation = camera_rotation_relation(target, rotation_direction, angle_deg)
    if relation is None:
        return None
    raw_delta = wrapped_delta_deg(yaw_deg_360(target) - ((float(angle_deg) if rotation_direction == "right" else -float(angle_deg)) % 360.0))
    margin = closest_boundary_margin(abs(raw_delta), [15.0, 90.0, 150.0, 180.0])
    item_key = f"{scene.scene_id}:camera_rotation_transform_mc:{target.entity_id}:{rotation_direction}:{angle_deg}"
    target_ref = entity_ref(target, scene, f"{item_key}:target")
    choices = choice_rows(REORIENTED_RELATION_LABELS)
    question = pick_template("camera_rotation_transform_mc", f"{scene.scene_id}:{target.entity_id}:{rotation_direction}:{angle_deg}").format(
        angle_deg=angle_deg,
        turn_direction=rotation_direction,
        target_ref=target_ref,
    )
    return benchmark_item(
        scene=scene,
        task_id="camera_rotation_transform_mc",
        item_id=f"{scene.scene_id}_camera_rotation_transform_mc_{target.entity_id}_{rotation_direction}_{angle_deg}",
        question=question,
        answer=label_to_choice_key(REORIENTED_RELATION_LABELS, relation),
        answer_text=relation,
        options=choices,
        target_entities=[target.entity_id],
        metadata={
            "target_ref": target_ref,
            "target_yaw_deg": round(yaw_deg_360(target), 1),
            "turn_direction": rotation_direction,
            "angle_deg": angle_deg,
            "delta_after_rotation_deg": round(raw_delta, 2),
        },
        difficulty=difficulty_from_margin(margin),
        quality_score=quality,
        diagnostic_slices=dedupe_keep_order(slices_for_entity(target) + ["rotation"]),
    )


def build_object_conditioned_reorientation_mc(scene: SceneMetadata, facing: Entity, target: Entity, anchor_index: int, partner_index: int, quality: float) -> Optional[Dict[str, Any]]:
    delta = wrapped_delta_deg(yaw_deg_360(target) - yaw_deg_360(facing))
    relation = panoramic_relation_from_delta(delta, opposite_label="behind")
    if relation is None:
        return None
    margin = closest_boundary_margin(abs(delta), [15.0, 90.0, 150.0, 180.0])
    if margin < 8.0:
        return None
    item_key = f"{scene.scene_id}:object_conditioned_reorientation_mc:{facing.entity_id}:{target.entity_id}"
    facing_ref = entity_ref(facing, scene, f"{item_key}:facing")
    target_ref = entity_ref(target, scene, f"{item_key}:target")
    choices = choice_rows(REORIENTED_RELATION_LABELS)
    question = pick_template("object_conditioned_reorientation_mc", f"{scene.scene_id}:{facing.entity_id}:{target.entity_id}").format(
        facing_ref=facing_ref,
        target_ref=target_ref,
    )
    return benchmark_item(
        scene=scene,
        task_id="object_conditioned_reorientation_mc",
        item_id=f"{scene.scene_id}_object_conditioned_reorientation_mc_{facing.entity_id}_{target.entity_id}",
        question=question,
        answer=label_to_choice_key(REORIENTED_RELATION_LABELS, relation),
        answer_text=relation,
        options=choices,
        target_entities=[facing.entity_id, target.entity_id],
        metadata={
            "facing_ref": facing_ref,
            "target_ref": target_ref,
            "facing_yaw_deg": round(yaw_deg_360(facing), 1),
            "target_yaw_deg": round(yaw_deg_360(target), 1),
            "delta_yaw_deg": round(delta, 2),
        },
        difficulty=difficulty_from_margin(margin),
        quality_score=(quality + float(score_entity(target, Counter(e.label for e in scene.entities), scene))) / 2.0,
        diagnostic_slices=dedupe_keep_order(slices_for_entity(facing) + slices_for_entity(target)),
    )


def build_polar_shape_recovery(scene: SceneMetadata, target: Entity, anchor_index: int, quality: float) -> Optional[Dict[str, Any]]:
    shape = shape_value(target)
    if not shape:
        return None
    if not (abs(target.lat_deg) >= 60.0 or infer_pole_proximity(target)):
        return None
    target_ref = entity_ref(target, scene, f"{scene.scene_id}:polar_shape_recovery_mc:{target.entity_id}:target")
    distractors = [item for item in SHAPE_FALLBACK_POOL if item != shape][:3]
    if len(distractors) < 3:
        return None
    options = [shape] + distractors
    choices = choice_rows(options)
    question = pick_template("polar_shape_recovery_mc", f"{scene.scene_id}:{target.entity_id}").format(target_ref=target_ref)
    return benchmark_item(
        scene=scene,
        task_id="polar_shape_recovery_mc",
        item_id=f"{scene.scene_id}_polar_shape_recovery_mc_{target.entity_id}",
        question=question,
        answer=choices[0]["key"],
        answer_text=shape,
        options=choices,
        target_entities=[target.entity_id],
        metadata={
            "target_ref": target_ref,
            "true_shape": shape,
            "target_bfov": list(target.resolved_bfov or ()),
            "abs_lat_deg": round(abs(target.lat_deg), 2),
        },
        difficulty="hard",
        quality_score=quality,
        diagnostic_slices=dedupe_keep_order(slices_for_entity(target) + ["pole"]),
        requires_manual_review=True,
        review_notes=["Verify that the labeled shape is visible enough and that the item is a true ERP distortion case."],
    )


def build_observer_distance_choice(scene: SceneMetadata, anchors: Sequence[Dict[str, Any]], anchor_index: int) -> Optional[Dict[str, Any]]:
    candidates = [item["entity"] for item in anchors if item["entity"].entity_center_depth is not None]
    if len(candidates) < 4:
        return None
    candidates = sorted(candidates, key=lambda entity: float(entity.entity_center_depth))[:6]
    selected = candidates[:4]
    depths = [float(entity.entity_center_depth) for entity in selected]
    if min(abs(depths[i] - depths[i + 1]) for i in range(len(depths) - 1)) < 0.35:
        return None
    closest = min(selected, key=lambda entity: float(entity.entity_center_depth))
    item_key = f"{scene.scene_id}:observer_distance_choice:{selected[0].entity_id}:{selected[-1].entity_id}"
    option_texts = [entity_ref(entity, scene, f"{item_key}:candidate:{idx}:{entity.entity_id}") for idx, entity in enumerate(selected)]
    choices = choice_rows(option_texts)
    question = pick_template("observer_distance_choice", f"{scene.scene_id}:{selected[0].entity_id}:{selected[-1].entity_id}")
    answer_key = choices[[entity.entity_id for entity in selected].index(closest.entity_id)]["key"]
    avg_quality = sum(float(score_entity(entity, Counter(e.label for e in scene.entities), scene)) for entity in selected) / len(selected)
    return benchmark_item(
        scene=scene,
        task_id="observer_distance_choice",
        item_id=f"{scene.scene_id}_observer_distance_choice_{selected[0].entity_id}_{selected[-1].entity_id}",
        question=question,
        answer=answer_key,
        answer_text=option_texts[[entity.entity_id for entity in selected].index(closest.entity_id)],
        options=choices,
        target_entities=[entity.entity_id for entity in selected],
        metadata={
            "candidate_refs": option_texts,
            "candidate_depths_m": {entity.entity_id: round(float(entity.entity_center_depth), 3) for entity in selected},
        },
        difficulty="medium",
        quality_score=avg_quality,
        diagnostic_slices=sorted({slice_name for entity in selected for slice_name in slices_for_entity(entity)}),
    )


def build_relative_3d_position_mc(scene: SceneMetadata, entity_a: Entity, entity_b: Entity, anchor_index: int, partner_index: int, quality: float) -> Optional[Dict[str, Any]]:
    if not compact_for_relative_3d(entity_a) or not compact_for_relative_3d(entity_b):
        return None
    relation, parts = relative_3d_relation(entity_a, entity_b)
    if not relation:
        return None
    item_key = f"{scene.scene_id}:relative_3d_position_mc:{entity_a.entity_id}:{entity_b.entity_id}"
    entity_a_ref = entity_ref(entity_a, scene, f"{item_key}:entity_a")
    entity_b_ref = entity_ref(entity_b, scene, f"{item_key}:entity_b")
    options = relative_3d_choices(entity_a, entity_b, relation, parts)
    choices = choice_rows(options)
    question = pick_template("relative_3d_position_mc", f"{scene.scene_id}:{entity_a.entity_id}:{entity_b.entity_id}").format(
        entity_a_ref=entity_a_ref,
        entity_b_ref=entity_b_ref,
    )
    xyz_a = entity_a.erp_consistent_xyz_camera
    xyz_b = entity_b.erp_consistent_xyz_camera
    dx = float(xyz_a[0] - xyz_b[0]) if xyz_a and xyz_b else 0.0
    dy = float(xyz_a[1] - xyz_b[1]) if xyz_a and xyz_b else 0.0
    dz = float(xyz_a[2] - xyz_b[2]) if xyz_a and xyz_b else 0.0
    return benchmark_item(
        scene=scene,
        task_id="relative_3d_position_mc",
        item_id=f"{scene.scene_id}_relative_3d_position_mc_{entity_a.entity_id}_{entity_b.entity_id}",
        question=question,
        answer=label_to_choice_key(options, relation),
        answer_text=relation,
        options=choices,
        target_entities=[entity_a.entity_id, entity_b.entity_id],
        metadata={
            "entity_a_ref": entity_a_ref,
            "entity_b_ref": entity_b_ref,
            "bfov_a": list(entity_a.resolved_bfov or ()),
            "bfov_b": list(entity_b.resolved_bfov or ()),
            "depth_a_m": round(float(entity_a.entity_center_depth or 0.0), 3),
            "depth_b_m": round(float(entity_b.entity_center_depth or 0.0), 3),
            "delta_xyz_m": [round(dx, 3), round(dy, 3), round(dz, 3)],
            "geometry_source": "erp_bfov_depth_derived",
            "camera_viewpoint_relation": True,
        },
        difficulty="medium",
        quality_score=(quality + float(score_entity(entity_b, Counter(e.label for e in scene.entities), scene))) / 2.0,
        diagnostic_slices=dedupe_keep_order(slices_for_entity(entity_a) + slices_for_entity(entity_b)),
        requires_manual_review=True,
        review_notes=["Verify camera-centered front/behind wording from the actual ERP viewpoint."],
    )


def slices_for_entity(entity: Entity) -> List[str]:
    slices: List[str] = []
    if entity.seam_crossing_flag:
        slices.append("seam")
    if infer_pole_proximity(entity):
        slices.append("pole")
    return slices


def label_to_choice_key(options: Sequence[str], label: str) -> str:
    idx = list(options).index(label)
    return option_key(idx)


def pluralize_label(label: str) -> str:
    text = str(label).strip()
    if not text:
        return "objects"
    if text.endswith("s"):
        return text
    if text.endswith("y") and len(text) > 1 and text[-2].lower() not in "aeiou":
        return text[:-1] + "ies"
    return text + "s"


def select_split_pool(candidates: Sequence[Dict[str, Any]], target_per_task: int, max_per_scene_per_task: int, seed: int) -> List[Dict[str, Any]]:
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_task[row["task_id"]].append(row)

    selected: List[Dict[str, Any]] = []
    for task_id, rows in sorted(by_task.items()):
        rows = list(rows)
        if task_id == "seam_continuity_mc":
            rows.sort(key=lambda row: stable_hash(f"{seed}:{row['item_id']}"))
        else:
            rows.sort(
                key=lambda row: (
                    -float(row.get("quality_score", 0.0)),
                    stable_hash(f"{seed}:{row['item_id']}"),
                )
            )
        per_scene_counter: Counter[str] = Counter()
        kept = 0
        for row in rows:
            if kept >= target_per_task:
                break
            if per_scene_counter[row["scene_id"]] >= max_per_scene_per_task:
                continue
            selected.append(row)
            per_scene_counter[row["scene_id"]] += 1
            kept += 1
    selected.sort(key=lambda row: (row["task_id"], row["scene_id"], row["item_id"]))
    return selected


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


def build_summary(
    *,
    scene_infos: Dict[str, SceneSideInfo],
    all_candidates: Sequence[Dict[str, Any]],
    public_selected: Sequence[Dict[str, Any]],
    review_queue: Sequence[Dict[str, Any]],
    target_public_per_task: int,
) -> Dict[str, Any]:
    def task_counter(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        counts = Counter(row["task_id"] for row in rows)
        return dict(sorted(counts.items()))

    def group_counter(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        counts = Counter(row["ability_group"] for row in rows)
        return dict(sorted(counts.items()))

    split_counts = Counter()
    for info in scene_infos.values():
        split_counts["total_scenes"] += 1
        if info.source_id:
            split_counts[f"source::{info.source_id}"] += 1
        if info.domain:
            split_counts[f"domain::{info.domain}"] += 1

    return {
        "benchmark_name": "ERP Spatial Benchmark",
        "benchmark_version": "v2_spatial_core",
        "design_policy": {
            "all_answers_public": True,
            "benchmark_only_templates": True,
            "closed_form_answers_preferred": True,
            "manual_review_queue_exported": True,
        },
        "ability_groups": sorted({spec["ability_group"] for spec in TASK_SPECS.values()}),
        "target_public_per_task": target_public_per_task,
        "num_scenes": len(scene_infos),
        "candidate_pool_size": len(all_candidates),
        "review_queue_size": len(review_queue),
        "benchmark_public_size": len(public_selected),
        "candidate_per_task": task_counter(all_candidates),
        "benchmark_public_per_task": task_counter(public_selected),
        "benchmark_public_per_group": group_counter(public_selected),
        "scene_metadata_overview": dict(sorted(split_counts.items())),
        "leakage_controls": {
            "training_overlap_expected": "no_scene_overlap_by_construction",
            "template_overlap_control": "benchmark_template_family_separate_from_training_templates",
            "selection_overlap_control": "per_scene_cap_with_public_release_only",
            "manual_review_required_for_fragile_tasks": sorted(MANUAL_REVIEW_TASKS),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
