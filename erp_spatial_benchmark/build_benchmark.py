#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from _vendor.entity_selector import (
    choose_relation_partners,
    infer_pole_proximity,
    infer_seam_adjacency,
    score_entity,
    select_anchor_entities,
)
from _vendor.schemas import Entity, SceneMetadata


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
ABSOLUTE_DIRECTION_CHALLENGE_SECTORS = {
    "right",
    "back-right",
    "back",
    "back-left",
    "left",
}
ABSOLUTE_DIRECTION_CHALLENGE_CENTERS = {
    "right": 90.0,
    "back-right": 135.0,
    "back": 180.0,
    "back-left": 225.0,
    "left": 270.0,
}
ABSOLUTE_DIRECTION_PUBLIC_WEIGHTS = {
    "right": 0.10,
    "back-right": 0.2666666667,
    "back": 0.2666666667,
    "back-left": 0.2666666666,
    "left": 0.10,
}
RELATIVE_DIRECTION_PUBLIC_WEIGHTS = {
    "right": 0.10,
    "back-right": 0.2666666667,
    "opposite": 0.2666666667,
    "back-left": 0.2666666666,
    "left": 0.10,
}
OBJECT_REORIENTATION_PUBLIC_WEIGHTS = {
    "right": 0.10,
    "back-right": 0.2666666667,
    "behind": 0.2666666667,
    "back-left": 0.2666666666,
    "left": 0.10,
}
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
SHAPE_CANONICAL_MAP = {
    "round": "round",
    "circular": "round",
    "circle": "round",
    "rounded": "round",
    "roundish": "round",
    "disc": "round",
    "disk": "round",
    "rectangular": "rectangular",
    "rectangle": "rectangular",
    "boxy": "rectangular",
    "box-shaped": "rectangular",
    "flat rectangular": "rectangular",
    "flat-rectangular": "rectangular",
    "box like": "rectangular",
    "box-like": "rectangular",
    "cuboid": "rectangular",
    "cuboidal": "rectangular",
    "square": "square",
    "oval": "oval",
    "elliptical": "oval",
    "ellipse": "oval",
    "oblong": "oval",
    "ellipsoid": "oval",
    "cylindrical": "cylindrical",
    "cylinder": "cylindrical",
    "tube": "cylindrical",
    "tubular": "cylindrical",
    "barrel": "cylindrical",
    "barrel-shaped": "cylindrical",
    "spherical": "spherical",
    "sphere": "spherical",
    "dome": "spherical",
    "globular": "spherical",
    "triangular": "triangular",
    "triangle": "triangular",
    "arched": "arched",
    "arch": "arched",
    "arc-shaped": "arched",
    "arc shaped": "arched",
}
SHAPE_DISTRACTOR_MAP = {
    "round": ["oval", "spherical", "cylindrical"],
    "rectangular": ["square", "oval", "cylindrical"],
    "square": ["rectangular", "triangular", "round"],
    "oval": ["round", "rectangular", "cylindrical"],
    "cylindrical": ["round", "rectangular", "square"],
    "spherical": ["round", "oval", "cylindrical"],
    "triangular": ["arched", "rectangular", "square"],
    "arched": ["triangular", "rectangular", "square"],
}
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
    "polar_shape_matching_mc",
    "polar_cross_latitude_matching_mc",
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
ENTITY_LABEL_BLOCKLIST_SUBSTRINGS = (
    "tree",
    "window",
    "leaf",
    "branch",
    "foliage",
    "bush",
    "shrub",
    "plant",
    "grass",
    "sky",
    "cloud",
)
MIN_DETECTION_SCORE = 0.65
MIN_REGROUND_SCORE = 0.65
ABSOLUTE_DIRECTION_MIN_MARGIN_DEG = 15.0
RELATION_MIN_MARGIN_DEG = 15.0
DIRECTION_MAX_X_FOV_DEG = 35.0
DIRECTION_MAX_Y_FOV_DEG = 30.0
DIRECTION_MAX_AREA_RATIO = 0.08
DERIVED_MIN_SEAM_CANDIDATES = 40
DERIVED_MIN_POLAR_CANDIDATES = 40
DERIVED_MIN_ABSOLUTE_DIRECTION_CANDIDATES = 40
DERIVED_ROTATION_DIRNAME = "derived_rotations"
DERIVED_MAX_PER_SCENE_PER_TASK = 3
POLAR_TARGET_LAT_MIN_DEG = 75.0
POLAR_TARGET_LAT_MAX_DEG = 85.0
POLAR_TARGET_LAT_CENTER_DEG = 80.0
POLAR_NATURAL_MIN_LAT_DEG = 60.0
POLAR_LOW_LAT_MAX_DEG = 35.0
RELATIVE_3D_MAX_X_FOV_DEG = 35.0
RELATIVE_3D_MAX_Y_FOV_DEG = 35.0
RELATIVE_3D_MAX_FOV_AREA = 1200.0
RELATIVE_3D_BLOCKLIST_SUBSTRINGS = (
    "building",
    "wall",
    "ceiling",
    "roof",
    "floor",
    "ground",
    "road",
    "sidewalk",
    "facade",
    "fence",
    "railing",
    "gate",
)
REFERENCE_STOPWORDS = {
    "the",
    "a",
    "an",
    "with",
    "and",
    "of",
    "on",
    "in",
    "at",
    "near",
    "to",
    "from",
    "for",
    "by",
    "visible",
    "standing",
    "parked",
    "side",
    "front",
    "back",
    "left",
    "right",
    "upper",
    "lower",
    "top",
    "bottom",
    "center",
    "middle",
}
POLAR_SHAPE_LEAK_TOKENS = frozenset(set(SHAPE_CANONICAL_MAP.keys()) | set(SHAPE_CANONICAL_MAP.values()))

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
            "In the complete 360 panorama, with the ERP image center treated as the current front direction, which direction sector best contains {target_ref}?",
            "Using the current ERP view as the reference frame (image center = front), which absolute panorama sector best matches {target_ref}?",
        ],
        "answer_format": "4_way_multiple_choice",
    },
    "relative_direction_mc": {
        "ability_group": "spherical_localization_and_panoramic_topology",
        "templates": [
            "On the observer-centered panoramic ring, where does {target_ref} fall relative to {reference_ref}?",
            "Using observer-centered panorama angles around the current camera, what is the angular relation of {target_ref} to {reference_ref}?",
        ],
        "answer_format": "5_way_multiple_choice",
    },
    "camera_rotation_transform_mc": {
        "ability_group": "viewpoint_conditioned_spatial_updating",
        "templates": [
            "If the observer stays in place and turns {angle_deg} degrees to the {turn_direction}, where would {target_ref} appear in the new front-centered view?",
            "After turning {angle_deg} degrees to the {turn_direction} without changing position, where does {target_ref} appear in the updated front-centered view?",
        ],
        "answer_format": "5_way_multiple_choice",
    },
    "object_conditioned_reorientation_mc": {
        "ability_group": "viewpoint_conditioned_spatial_updating",
        "templates": [
            "Once {facing_ref} is centered as the new front direction, where does {target_ref} lie in that reoriented observer frame?",
            "If you turn to face {facing_ref}, where would {target_ref} appear in the reoriented front-centered view?",
        ],
        "answer_format": "5_way_multiple_choice",
    },
    "observer_distance_choice": {
        "ability_group": "observer_centered_3d_layout_understanding",
        "templates": [
            "Which of these objects is closest to the current observer in the full panorama?",
            "From the current camera position, which listed object is nearest?",
        ],
        "answer_format": "3_or_4_way_multiple_choice",
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
    "polar_shape_matching_mc": {
        "ability_group": "erp_representation_understanding",
        "templates": [
            "Which listed object best matches the real geometry of {target_ref} in this high-latitude ERP region?",
            "Which candidate object most closely matches the true shape of {target_ref} in this high-latitude ERP panorama?",
        ],
        "answer_format": "4_way_multiple_choice",
    },
    "polar_cross_latitude_matching_mc": {
        "ability_group": "erp_representation_understanding",
        "templates": [
            "Which lower-latitude object best matches the real geometry of {target_ref} in this high-latitude ERP region?",
            "Which object from a less distorted region most closely matches the true shape of {target_ref}?",
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
    parser.add_argument(
        "--max-per-scene-per-task",
        type=int,
        default=0,
        help="Maximum selected items from one scene for one task. Use 0 for no per-scene cap.",
    )
    parser.add_argument("--seed", type=int, default=20260327, help="Random seed for deterministic splitting and selection.")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=[],
        help="Optional task-id allowlist. When set, only these task types are generated and selected.",
    )
    parser.add_argument(
        "--fail-on-invalid-json",
        action="store_true",
        help="Fail immediately when a metadata.json file is empty or invalid. By default invalid files are skipped.",
    )
    return parser.parse_args()


def normalize_task_filter(raw_tasks: Sequence[str]) -> Optional[set[str]]:
    normalized: set[str] = set()
    for raw in raw_tasks:
        for part in str(raw).split(","):
            task_id = part.strip()
            if task_id:
                normalized.add(task_id)
    if not normalized:
        return None
    unknown = sorted(task_id for task_id in normalized if task_id not in TASK_SPECS)
    if unknown:
        raise ValueError(f"Unknown task ids in --tasks: {', '.join(unknown)}")
    return normalized


def task_enabled(task_id: str, enabled_tasks: Optional[set[str]]) -> bool:
    return enabled_tasks is None or task_id in enabled_tasks


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    enabled_tasks = normalize_task_filter(args.tasks)

    scene_manifest = load_scene_manifest(args.scene_manifest)
    scene_paths = list(discover_metadata_files(input_root))
    print(json.dumps({"stage": "discover", "num_scene_files": len(scene_paths)}, ensure_ascii=False))

    all_candidates: List[Dict[str, Any]] = []
    scene_infos: Dict[str, SceneSideInfo] = {}
    skipped_invalid_metadata: List[Dict[str, Any]] = []
    scenes: List[SceneMetadata] = []
    split_seed = int(args.seed)

    for idx, metadata_path in enumerate(scene_paths, start=1):
        try:
            scene = load_scene_metadata(metadata_path)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            record = {
                "path": str(metadata_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            skipped_invalid_metadata.append(record)
            print(json.dumps({"stage": "skip_invalid_metadata", **record}, ensure_ascii=False))
            if args.fail_on_invalid_json:
                raise
            continue
        info = build_scene_side_info(scene, scene_manifest)
        scene_infos[scene.scene_id] = info
        scenes.append(scene)
        candidates = generate_scene_candidates(scene, enabled_tasks=enabled_tasks)
        all_candidates.extend(candidates)
        if idx % 25 == 0 or idx == len(scene_paths):
            print(json.dumps({"stage": "candidates", "processed_scenes": idx, "candidate_count": len(all_candidates)}, ensure_ascii=False))

    derived_rows = augment_representation_stress_candidates(
        scenes,
        all_candidates,
        scene_infos,
        output_dir,
        target_public_per_task=int(args.target_public_per_task),
        seed=split_seed,
        enabled_tasks=enabled_tasks,
    )
    if derived_rows:
        all_candidates.extend(derived_rows)
        print(json.dumps({"stage": "derived_representation_stress", "added_candidates": len(derived_rows), "candidate_count": len(all_candidates)}, ensure_ascii=False))

    public_selected = select_split_pool(
        all_candidates,
        target_per_task=int(args.target_public_per_task),
        max_per_scene_per_task=args.max_per_scene_per_task,
        seed=split_seed + 11,
    )
    public_selected = rebalance_mc_answer_keys(public_selected, split_seed + 29)

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
    write_jsonl(output_dir / "skipped_invalid_metadata.jsonl", skipped_invalid_metadata)

    summary = build_summary(
        scene_infos=scene_infos,
        all_candidates=all_candidates,
        public_selected=public_selected,
        review_queue=review_queue,
        target_public_per_task=int(args.target_public_per_task),
        skipped_invalid_metadata=skipped_invalid_metadata,
        num_input_scenes=len(scenes),
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
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        raise ValueError(f"Empty metadata file: {path}")
    data = json.loads(raw)
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


def generate_scene_candidates(scene: SceneMetadata, enabled_tasks: Optional[set[str]] = None) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    label_counts = Counter(entity.label for entity in scene.entities)
    anchors = [
        item
        for item in select_anchor_entities(scene, max_anchors=0)
        if anchor_pool_entity_eligible(item["entity"])
    ]

    used_item_ids: set[str] = set()
    for anchor_index, anchor_payload in enumerate(anchors):
        anchor = anchor_payload["entity"]
        quality = float(score_entity(anchor, label_counts, scene))
        anchor_rows: List[Optional[Dict[str, Any]]] = []
        if task_enabled("referring_grounding_bfov", enabled_tasks):
            anchor_rows.append(build_referring_grounding_bfov(scene, anchor, anchors, anchor_index, quality))
        if task_enabled("absolute_direction_mc", enabled_tasks):
            anchor_rows.append(build_absolute_direction_mc(scene, anchor, anchor_index, quality))
        if task_enabled("polar_shape_recovery_mc", enabled_tasks):
            anchor_rows.append(build_polar_shape_recovery(scene, anchor, anchor_index, quality))
        if task_enabled("polar_shape_matching_mc", enabled_tasks):
            anchor_rows.append(build_polar_shape_matching(scene, anchor, anchor_index, quality))
        if task_enabled("polar_cross_latitude_matching_mc", enabled_tasks):
            anchor_rows.append(build_polar_cross_latitude_matching(scene, anchor, anchor_index, quality))
        for row in anchor_rows:
            if row and row["item_id"] not in used_item_ids:
                candidates.append(row)
                used_item_ids.add(row["item_id"])

        if task_enabled("seam_continuity_mc", enabled_tasks):
            for row in build_seam_continuity_items(scene, anchor, anchors, anchor_index, quality):
                if row and row["item_id"] not in used_item_ids:
                    candidates.append(row)
                    used_item_ids.add(row["item_id"])

        partners = choose_relation_partners(anchor, scene, max_partners=0)
        for partner_index, partner_payload in enumerate(partners):
            partner = partner_payload["entity"]
            relation_rows: List[Optional[Dict[str, Any]]] = []
            if task_enabled("relative_direction_mc", enabled_tasks):
                relation_rows.append(build_relative_direction_mc(scene, anchor, partner, anchor_index, partner_index, quality))
            if task_enabled("camera_rotation_transform_mc", enabled_tasks):
                relation_rows.append(build_camera_rotation_transform_mc(scene, anchor, anchor_index, partner_index, quality))
            if task_enabled("object_conditioned_reorientation_mc", enabled_tasks):
                relation_rows.append(build_object_conditioned_reorientation_mc(scene, anchor, partner, anchor_index, partner_index, quality))
            if task_enabled("relative_3d_position_mc", enabled_tasks):
                relation_rows.append(build_relative_3d_position_mc(scene, anchor, partner, anchor_index, partner_index, quality))
            for row in relation_rows:
                if row and row["item_id"] not in used_item_ids:
                    candidates.append(row)
                    used_item_ids.add(row["item_id"])

        if task_enabled("observer_distance_choice", enabled_tasks):
            observer_choice = build_observer_distance_choice(scene, anchors, anchor_index)
            if observer_choice and observer_choice["item_id"] not in used_item_ids:
                candidates.append(observer_choice)
                used_item_ids.add(observer_choice["item_id"])

    return candidates


def benchmark_entity_eligible(entity: Entity) -> bool:
    label = normalize_phrase(entity.label)
    if not label or label in {"unknown", "object"}:
        return False
    if any(token in label for token in ENTITY_LABEL_BLOCKLIST_SUBSTRINGS):
        return False
    det_score = float(entity.best_score or entity.confidence or 0.0)
    if det_score < MIN_DETECTION_SCORE:
        return False
    reground_score = float(entity.local_reground_pred_score or 0.0)
    if reground_score < MIN_REGROUND_SCORE:
        return False
    if float(entity.area_ratio or 0.0) <= 0.0004:
        return False
    return bool(entity.resolved_bfov)


def anchor_pool_entity_eligible(entity: Entity) -> bool:
    label = normalize_phrase(entity.label)
    if not label or label in {"unknown", "object"}:
        return False
    if any(token in label for token in ENTITY_LABEL_BLOCKLIST_SUBSTRINGS):
        return False
    det_score = float(entity.best_score or entity.confidence or 0.0)
    if det_score < MIN_DETECTION_SCORE:
        return False
    reground_score = float(entity.local_reground_pred_score or 0.0)
    if reground_score < MIN_REGROUND_SCORE:
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


def entity_has_large_extent(entity: Entity, *, x_limit: float, y_limit: float, area_limit: float) -> bool:
    bfov = entity.resolved_bfov
    if bfov is None:
        return True
    x_fov = abs(float(bfov[2]))
    y_fov = abs(float(bfov[3]))
    return x_fov > x_limit or y_fov > y_limit or float(entity.area_ratio or 0.0) > area_limit


def direction_task_entity_eligible(entity: Entity) -> bool:
    if not benchmark_entity_eligible(entity):
        return False
    return not entity_has_large_extent(
        entity,
        x_limit=DIRECTION_MAX_X_FOV_DEG,
        y_limit=DIRECTION_MAX_Y_FOV_DEG,
        area_limit=DIRECTION_MAX_AREA_RATIO,
    )


def ref_has_spatial_hint(text: str) -> bool:
    normalized = normalize_phrase(text)
    hint_tokens = (
        "left",
        "right",
        "front",
        "back",
        "center",
        "middle",
        "upper",
        "lower",
        "top",
        "bottom",
        "near",
    )
    return any(token in normalized for token in hint_tokens)


def reference_tokens(text: str) -> set[str]:
    return {
        token
        for token in normalize_phrase(text).split()
        if token and token not in REFERENCE_STOPWORDS and len(token) > 2
    }


def refs_semantically_similar(text_a: str, text_b: str) -> bool:
    norm_a = normalize_phrase(text_a)
    norm_b = normalize_phrase(text_b)
    if not norm_a or not norm_b:
        return False
    if norm_a == norm_b:
        return True
    tokens_a = reference_tokens(norm_a)
    tokens_b = reference_tokens(norm_b)
    if not tokens_a or not tokens_b:
        return False
    overlap = len(tokens_a & tokens_b) / max(1, len(tokens_a | tokens_b))
    return overlap >= 0.72


def duplicate_label_count(scene: SceneMetadata, entity: Entity) -> int:
    label = normalize_phrase(entity.label)
    return sum(1 for other in scene.entities if normalize_phrase(other.label) == label)


def similar_duplicate_entities(scene: SceneMetadata, entity: Entity) -> List[Entity]:
    label = normalize_phrase(entity.label)
    base_ref = descriptive_entity_ref(entity)
    similar: List[Entity] = []
    for other in scene.entities:
        if normalize_phrase(other.label) != label:
            continue
        if refs_semantically_similar(base_ref, descriptive_entity_ref(other)):
            similar.append(other)
    return similar


def coarse_sector_hint(entity: Entity) -> str:
    yaw = yaw_deg_360(entity)
    if 45.0 <= yaw < 135.0:
        return "right side"
    if 135.0 <= yaw < 225.0:
        return "back side"
    if 225.0 <= yaw < 315.0:
        return "left side"
    return "front side"


def duplicate_disambiguation_hint(scene: SceneMetadata, entity: Entity) -> Optional[str]:
    similar = similar_duplicate_entities(scene, entity)
    if len(similar) <= 1:
        return ""

    side_counts = Counter(coarse_sector_hint(other) for other in similar)
    entity_side = coarse_sector_hint(entity)
    if side_counts[entity_side] == 1:
        return f"near the {entity_side}"

    centers = []
    for other in similar:
        if len(other.bbox_erp) == 4:
            x1, y1, x2, y2 = other.bbox_erp
            centers.append((other.entity_id, (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0))

    if len(centers) >= 2:
        width = float(scene.erp_width or 0.0)
        height = float(scene.erp_height or 0.0)

        centers_x = sorted(centers, key=lambda item: item[1])
        if width > 0 and (centers_x[-1][1] - centers_x[0][1]) >= 0.18 * width:
            if centers_x[0][0] == entity.entity_id:
                return "toward the left side"
            if centers_x[-1][0] == entity.entity_id:
                return "toward the right side"

        centers_y = sorted(centers, key=lambda item: item[2])
        if height > 0 and (centers_y[-1][2] - centers_y[0][2]) >= 0.14 * height:
            if centers_y[0][0] == entity.entity_id:
                return "in the upper part of the scene"
            if centers_y[-1][0] == entity.entity_id:
                return "in the lower part of the scene"

    if len(similar) >= 3:
        return None
    return f"near the {entity_side}"


def reference_is_resolvable(scene: SceneMetadata, entity: Entity) -> bool:
    return duplicate_disambiguation_hint(scene, entity) is not None


def contextual_entity_ref(scene: SceneMetadata, entity: Entity) -> str:
    ref = descriptive_entity_ref(entity)
    similar = similar_duplicate_entities(scene, entity)
    if len(similar) <= 1:
        return ref
    if ref_has_spatial_hint(ref):
        return ref
    hint = duplicate_disambiguation_hint(scene, entity)
    if not hint:
        return ref
    return f"{ref} {hint}"


def polar_shape_leak_terms(entity: Entity) -> set[str]:
    terms = set(POLAR_SHAPE_LEAK_TOKENS)
    attrs = entity.semantic.attributes or {}
    raw_shape = first_nonempty_str(attrs.get("shape"))
    if raw_shape:
        normalized = normalize_phrase(raw_shape.replace("-", " "))
        terms.update(token for token in normalized.split() if token)
        terms.add(normalized)
    canonical = shape_value(entity)
    if canonical:
        terms.update(token for token in canonical.replace("-", " ").split() if token)
        terms.add(canonical)
    return terms


def strip_polar_shape_tokens(text: str, entity: Entity) -> str:
    leak_terms = polar_shape_leak_terms(entity)
    tokens: List[str] = []
    for token in str(text).split():
        normalized = normalize_phrase(token.strip(".,;:!?()[]{}\"'").replace("-", " "))
        if normalized in leak_terms:
            continue
        tokens.append(token)
    return " ".join(tokens).strip()


def polar_entity_ref(scene: SceneMetadata, entity: Entity) -> str:
    cleaned = strip_polar_shape_tokens(contextual_entity_ref(scene, entity), entity)
    if cleaned:
        return cleaned
    hint = duplicate_disambiguation_hint(scene, entity)
    base = f"the {normalize_phrase(entity.label) or 'object'}"
    if hint:
        return f"{base} {hint}"
    return base


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
    yaw = yaw_deg_360(entity)
    center = float(ABSOLUTE_SECTORS_8.index(absolute_sector_8way(entity)) * 45)
    return max(0.0, 22.5 - abs(wrapped_delta_deg(yaw - center)))


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


def effective_direction_margin(entity: Entity, raw_margin: float) -> float:
    bfov = entity.resolved_bfov
    if bfov is None:
        return raw_margin
    x_half = abs(float(bfov[2])) / 2.0
    return raw_margin - x_half


def shape_value(entity: Entity) -> Optional[str]:
    attrs = entity.semantic.attributes or {}
    raw = attrs.get("shape")
    if raw is None:
        return None
    value = normalize_phrase(str(raw).replace("-", " "))
    exact = SHAPE_CANONICAL_MAP.get(value)
    if exact:
        return exact
    padded = f" {value} "
    for alias, canonical in sorted(SHAPE_CANONICAL_MAP.items(), key=lambda item: len(item[0]), reverse=True):
        alias_norm = normalize_phrase(alias.replace("-", " "))
        if not alias_norm:
            continue
        if f" {alias_norm} " in padded:
            return canonical
    return None


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


def rebalance_choice_rows(options: Sequence[Dict[str, str]], answer_text: str, answer_key: str, target_key: str, seed_key: str) -> Tuple[List[Dict[str, str]], str]:
    ordered = [str(option["text"]) for option in options]
    current_correct_text = answer_text
    if current_correct_text not in ordered:
        current_index = ord(answer_key) - ord("A")
        if 0 <= current_index < len(ordered):
            current_correct_text = ordered[current_index]
        else:
            raise ValueError("Cannot recover correct option text for answer rebalancing.")
    incorrect = [text for text in ordered if text != current_correct_text]
    if incorrect:
        rotation = stable_hash(f"{seed_key}:incorrects") % len(incorrect)
        incorrect = incorrect[rotation:] + incorrect[:rotation]
    target_index = ord(target_key) - ord("A")
    rebuilt = list(incorrect)
    rebuilt.insert(target_index, current_correct_text)
    return choice_rows(rebuilt), target_key


def rebalance_mc_answer_keys(rows: Sequence[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    balanced = [copy.deepcopy(row) for row in rows]
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in balanced:
        if row.get("options"):
            by_task[str(row["task_id"])].append(row)

    for task_id, task_rows in by_task.items():
        counts: Counter[str] = Counter()
        task_rows.sort(key=lambda row: stable_hash(f"{seed}:{task_id}:{row['item_id']}:answer_balance"))
        for row in task_rows:
            options = list(row.get("options") or [])
            if not options:
                continue
            valid_keys = [option_key(i) for i in range(len(options))]
            target_key = min(
                valid_keys,
                key=lambda key: (
                    counts[key],
                    stable_hash(f"{seed}:{task_id}:{row['item_id']}:{key}"),
                ),
            )
            new_options, new_answer = rebalance_choice_rows(
                options,
                str(row.get("answer_text", "")),
                str(row.get("answer", "")),
                target_key,
                f"{seed}:{task_id}:{row['item_id']}",
            )
            row["options"] = new_options
            row["answer"] = new_answer
            counts[new_answer] += 1
    return balanced


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
    label = normalize_phrase(entity.label)
    if any(token in label for token in RELATIVE_3D_BLOCKLIST_SUBSTRINGS):
        return False
    return (
        x_fov <= RELATIVE_3D_MAX_X_FOV_DEG
        and y_fov <= RELATIVE_3D_MAX_Y_FOV_DEG
        and (x_fov * y_fov) <= RELATIVE_3D_MAX_FOV_AREA
    )


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


def polar_target_eligible(entity: Entity) -> bool:
    return bool(shape_value(entity)) and (abs(entity.lat_deg) >= POLAR_NATURAL_MIN_LAT_DEG or infer_pole_proximity(entity))


def polar_option_entities(
    scene: SceneMetadata,
    target: Entity,
    *,
    low_lat_only: bool = False,
) -> List[Entity]:
    label_counts = Counter(entity.label for entity in scene.entities)
    candidates: List[Entity] = []
    for entity in scene.entities:
        if entity.entity_id == target.entity_id:
            continue
        if not benchmark_entity_eligible(entity):
            continue
        if not reference_is_resolvable(scene, entity):
            continue
        if shape_value(entity) is None:
            continue
        if low_lat_only and abs(entity.lat_deg) > POLAR_LOW_LAT_MAX_DEG:
            continue
        candidates.append(entity)
    candidates.sort(
        key=lambda entity: (
            -float(score_entity(entity, label_counts, scene)),
            stable_hash(f"polar_option:{scene.scene_id}:{entity.entity_id}"),
        )
    )
    return candidates


def unique_polar_entity_refs(scene: SceneMetadata, entities: Sequence[Entity]) -> List[Tuple[Entity, str]]:
    seen_refs: set[str] = set()
    unique: List[Tuple[Entity, str]] = []
    for entity in entities:
        ref = polar_entity_ref(scene, entity)
        if not ref or ref in seen_refs:
            continue
        seen_refs.add(ref)
        unique.append((entity, ref))
    return unique


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


def find_entity(scene: SceneMetadata, entity_id: str) -> Optional[Entity]:
    for entity in scene.entities:
        if entity.entity_id == entity_id:
            return entity
    return None


def yaw360_to_signed(yaw_deg: float) -> float:
    return wrapped_delta_deg(float(yaw_deg) % 360.0)


def bbox_dims(entity: Entity) -> Tuple[float, float]:
    if len(entity.bbox_erp) != 4:
        return (0.0, 0.0)
    x1, y1, x2, y2 = entity.bbox_erp
    return (max(1.0, float(x2) - float(x1)), max(1.0, float(y2) - float(y1)))


def yaw_to_erp_x(yaw_deg: float, width: int) -> float:
    # ERP convention used in this benchmark: yaw=0 is the front-facing center,
    # while the horizontal seam sits at yaw=+/-180 on the left/right image edge.
    return (((float(yaw_deg) + 180.0) % 360.0) / 360.0) * float(width)


def erp_x_to_yaw(x_px: float, width: int) -> float:
    return yaw360_to_signed(((float(x_px) / float(width)) * 360.0) - 180.0)


def pitch_to_erp_y(pitch_deg: float, height: int) -> float:
    # Benchmark convention: pitch=0 is horizon center, negative pitch is upward,
    # positive pitch is downward. Therefore top-of-image is pitch=-90 and bottom
    # is pitch=+90.
    return (((float(pitch_deg) + 90.0) / 180.0)) * float(height)


def erp_y_to_pitch(y_px: float, height: int) -> float:
    return ((float(y_px) / float(height)) * 180.0) - 90.0


def spherical_vector_from_yaw_pitch(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    lon = math.radians(yaw360_to_signed(yaw_deg))
    lat = math.radians(-float(pitch_deg))
    return np.array(
        [
            math.cos(lat) * math.sin(lon),
            math.sin(lat),
            math.cos(lat) * math.cos(lon),
        ],
        dtype=np.float64,
    )


def yaw_pitch_from_vector(vec: np.ndarray) -> Tuple[float, float]:
    x, y, z = [float(v) for v in vec]
    lon = math.degrees(math.atan2(x, z))
    lat = math.degrees(math.asin(max(-1.0, min(1.0, y))))
    return yaw360_to_signed(lon), -lat


def rotate_vector_pitch(vec: np.ndarray, pitch_deg: float) -> np.ndarray:
    rad = math.radians(float(pitch_deg))
    rot = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(rad), -math.sin(rad)],
            [0.0, math.sin(rad), math.cos(rad)],
        ],
        dtype=np.float64,
    )
    return rot @ vec


def rotate_yaw_pitch(yaw_deg: float, pitch_deg: float, *, yaw_shift_deg: float = 0.0, pitch_shift_deg: float = 0.0) -> Tuple[float, float]:
    if abs(pitch_shift_deg) < 1e-6:
        return yaw360_to_signed(float(yaw_deg) - float(yaw_shift_deg)), float(pitch_deg)
    vec = spherical_vector_from_yaw_pitch(yaw_deg, pitch_deg)
    if abs(yaw_shift_deg) > 1e-6:
        yaw_rad = math.radians(float(yaw_shift_deg))
        yaw_rot = np.array(
            [
                [math.cos(yaw_rad), 0.0, math.sin(yaw_rad)],
                [0.0, 1.0, 0.0],
                [-math.sin(yaw_rad), 0.0, math.cos(yaw_rad)],
            ],
            dtype=np.float64,
        )
        vec = yaw_rot.T @ vec
    vec = rotate_vector_pitch(vec, pitch_shift_deg)
    return yaw_pitch_from_vector(vec)


def write_yaw_shifted_erp_image(src_path: Path, dst_path: Path, yaw_shift_deg: float) -> None:
    from PIL import Image

    image = Image.open(src_path).convert("RGB")
    width = image.size[0]
    shift_px = int(round((float(yaw_shift_deg) / 360.0) * width))
    rolled = np.roll(np.asarray(image), -shift_px, axis=1)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rolled).save(dst_path)


def write_pitch_rotated_erp_image(src_path: Path, dst_path: Path, pitch_shift_deg: float) -> None:
    from PIL import Image

    image = np.asarray(Image.open(src_path).convert("RGB"))
    height, width = image.shape[:2]
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    lon = (xs / float(width)) * (2.0 * math.pi) - math.pi
    lat = (math.pi / 2.0) - (ys / float(height)) * math.pi
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)
    vectors = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    rad = math.radians(float(pitch_shift_deg))
    inv_rot = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(-rad), -math.sin(-rad)],
            [0.0, math.sin(-rad), math.cos(-rad)],
        ],
        dtype=np.float64,
    )
    source_vectors = vectors @ inv_rot.T
    src_x = source_vectors[:, 0]
    src_y = np.clip(source_vectors[:, 1], -1.0, 1.0)
    src_z = source_vectors[:, 2]
    src_lon = np.arctan2(src_x, src_z)
    src_lat = np.arcsin(src_y)
    map_x = (((src_lon + math.pi) / (2.0 * math.pi)) * width).astype(np.float32).reshape(height, width)
    map_y = (((math.pi / 2.0 - src_lat) / math.pi) * height).astype(np.float32).reshape(height, width)
    x0 = np.floor(map_x).astype(np.int32) % width
    x1 = (x0 + 1) % width
    y0 = np.clip(np.floor(map_y).astype(np.int32), 0, height - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    wx = map_x - np.floor(map_x)
    wy = map_y - np.floor(map_y)

    top_left = image[y0, x0]
    top_right = image[y0, x1]
    bottom_left = image[y1, x0]
    bottom_right = image[y1, x1]
    top = top_left * (1.0 - wx[..., None]) + top_right * wx[..., None]
    bottom = bottom_left * (1.0 - wx[..., None]) + bottom_right * wx[..., None]
    rotated = top * (1.0 - wy[..., None]) + bottom * wy[..., None]
    rotated = np.clip(rotated, 0, 255).astype(np.uint8)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rotated).save(dst_path)


def transformed_bbox(entity: Entity, scene: SceneMetadata, *, yaw_shift_deg: float = 0.0, pitch_shift_deg: float = 0.0) -> Tuple[List[float], bool]:
    width = int(scene.erp_width or 0)
    height = int(scene.erp_height or 0)
    if width <= 0 or height <= 0 or len(entity.bbox_erp) != 4:
        return list(entity.bbox_erp), False
    x1, y1, x2, y2 = [float(v) for v in entity.bbox_erp]
    xs = np.linspace(x1, x2, num=5)
    ys = np.linspace(y1, y2, num=5)
    sample_points = [(sx, sy) for sx in xs for sy in ys]

    rotated_xs: List[float] = []
    rotated_ys: List[float] = []
    for sx, sy in sample_points:
        yaw_old = erp_x_to_yaw(sx, width)
        pitch_old = erp_y_to_pitch(sy, height)
        yaw_new, pitch_new = rotate_yaw_pitch(
            yaw_old,
            pitch_old,
            yaw_shift_deg=yaw_shift_deg,
            pitch_shift_deg=pitch_shift_deg,
        )
        rotated_xs.append(yaw_to_erp_x(yaw_new, width) % float(width))
        rotated_ys.append(min(float(height), max(0.0, pitch_to_erp_y(pitch_new, height))))

    if not rotated_xs or not rotated_ys:
        return list(entity.bbox_erp), False

    rotated_xs.sort()
    largest_gap = -1.0
    largest_gap_index = 0
    for idx in range(len(rotated_xs)):
        current = rotated_xs[idx]
        nxt = rotated_xs[(idx + 1) % len(rotated_xs)]
        gap = (nxt - current) if idx + 1 < len(rotated_xs) else (nxt + float(width) - current)
        if gap > largest_gap:
            largest_gap = gap
            largest_gap_index = idx

    start = rotated_xs[(largest_gap_index + 1) % len(rotated_xs)]
    end = rotated_xs[largest_gap_index]
    seam_cross = start > end
    y_min = min(rotated_ys)
    y_max = max(rotated_ys)

    if seam_cross:
        left_span = end
        right_span = float(width) - start
        if right_span >= left_span:
            return [start, y_min, float(width), y_max], True
        return [0.0, y_min, end, y_max], True
    return [start, y_min, end, y_max], False


def bfov_extents_from_item(item: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    entity_bfov = item.get("entity_bfov")
    if isinstance(entity_bfov, list) and len(entity_bfov) == 4:
        try:
            return float(entity_bfov[2]), float(entity_bfov[3])
        except (TypeError, ValueError):
            pass
    bfov = item.get("bfov", {}) or {}
    x_fov = bfov.get("x_fov_deg")
    y_fov = bfov.get("y_fov_deg")
    try:
        return (None if x_fov is None else float(x_fov), None if y_fov is None else float(y_fov))
    except (TypeError, ValueError):
        return (None, None)


def xyz_camera_from_yaw_pitch_depth(yaw_deg: float, pitch_deg: float, depth: Optional[float]) -> Optional[List[float]]:
    if depth is None:
        return None
    depth_value = float(depth)
    yaw_rad = math.radians(float(yaw_deg))
    pitch_rad = math.radians(float(pitch_deg))
    x = depth_value * math.cos(pitch_rad) * math.sin(yaw_rad)
    y = depth_value * math.sin(-pitch_rad)
    z = depth_value * math.cos(pitch_rad) * math.cos(yaw_rad)
    return [x, y, z]


def derived_image_path(scene: SceneMetadata, suffix: str) -> Path:
    original = Path(scene.erp_image_path)
    return original.parent / f"{original.stem}__{suffix}{original.suffix}"


def build_rotated_scene(
    scene: SceneMetadata,
    *,
    yaw_shift_deg: float = 0.0,
    pitch_shift_deg: float = 0.0,
    suffix: str,
    output_dir: Path,
) -> Optional[SceneMetadata]:
    raw = copy.deepcopy(scene.raw)
    raw["scene_id"] = f"{scene.scene_id}__{suffix}"
    raw["erp_width"] = int(scene.erp_width or 0)
    raw["erp_height"] = int(scene.erp_height or 0)
    rotated_image = derived_image_path(scene, suffix)
    if not Path(scene.erp_image_path).exists():
        return None
    if abs(pitch_shift_deg) > 1e-6:
        write_pitch_rotated_erp_image(Path(scene.erp_image_path), rotated_image, pitch_shift_deg)
    else:
        write_yaw_shifted_erp_image(Path(scene.erp_image_path), rotated_image, yaw_shift_deg)
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
        else:
            item.pop("entity_bfov", None)
        bbox, seam_cross = transformed_bbox(
            Entity.from_dict(entity),
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
        # Masks are no longer aligned with the derived image after spherical rotation.
        item["mask_rle"] = {}
        transformed_entities.append(item)

    raw["entities"] = transformed_entities
    derived_metadata_dir = output_dir / "derived_metadata"
    derived_metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = derived_metadata_dir / f"{raw['scene_id']}.json"
    metadata_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return SceneMetadata.from_dict(raw)


def choose_pitch_shift_for_polar(scene: SceneMetadata, entity: Entity) -> Optional[float]:
    bfov = entity.resolved_bfov
    if bfov is None:
        return None
    yaw_deg = float(bfov[0])
    pitch_current = float(bfov[1])
    target_lat = POLAR_TARGET_LAT_MIN_DEG + (
        (stable_hash(f"polar_target_lat:{scene.scene_id}:{entity.entity_id}") % 1001) / 1000.0
    ) * (POLAR_TARGET_LAT_MAX_DEG - POLAR_TARGET_LAT_MIN_DEG)
    best: Optional[Tuple[float, float, float]] = None
    candidates = [float(shift) for shift in range(-89, 90) if shift != 0]
    for shift in candidates:
        _, new_pitch = rotate_yaw_pitch(yaw_deg, pitch_current, pitch_shift_deg=shift)
        lat_abs = abs(-new_pitch)
        if POLAR_TARGET_LAT_MIN_DEG <= lat_abs <= POLAR_TARGET_LAT_MAX_DEG:
            score = abs(lat_abs - target_lat)
            candidate = (score, abs(shift), shift)
            if best is None or candidate < best:
                best = candidate
    return None if best is None else best[2]


def choose_yaw_shift_for_absolute_direction(entity: Entity) -> float:
    yaw = yaw_deg_360(entity)
    centers = sorted(
        ABSOLUTE_DIRECTION_CHALLENGE_CENTERS.values(),
        key=lambda center: (
            abs(wrapped_delta_deg(yaw - center)),
            abs(center - 180.0),
        ),
    )
    target_center = centers[0]
    return wrapped_delta_deg(yaw - target_center)


def choose_yaw_shift_for_absolute_direction_sector(entity: Entity, sector: str) -> float:
    center = ABSOLUTE_DIRECTION_CHALLENGE_CENTERS[sector]
    return wrapped_delta_deg(yaw_deg_360(entity) - center)


def absolute_direction_target_counts(target_per_task: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    running_total = 0
    ordered = list(ABSOLUTE_DIRECTION_PUBLIC_WEIGHTS.items())
    for idx, (sector, weight) in enumerate(ordered):
        if idx == len(ordered) - 1:
            counts[sector] = max(0, target_per_task - running_total)
        else:
            value = int(round(target_per_task * weight))
            counts[sector] = value
            running_total += value
    return counts


def weighted_target_counts(target_per_task: int, weights: Dict[str, float]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    running_total = 0
    ordered = list(weights.items())
    for idx, (label, weight) in enumerate(ordered):
        if idx == len(ordered) - 1:
            counts[label] = max(0, target_per_task - running_total)
        else:
            value = int(round(target_per_task * weight))
            counts[label] = value
            running_total += value
    return counts


def derived_candidate_entities(
    scene: SceneMetadata,
    *,
    predicate: Optional[Any] = None,
    max_entities: int = 24,
) -> List[Entity]:
    label_counts = Counter(entity.label for entity in scene.entities)
    filtered: List[Entity] = []
    for entity in scene.entities:
        if not anchor_pool_entity_eligible(entity):
            continue
        if not reference_is_resolvable(scene, entity):
            continue
        if predicate is not None and not predicate(entity):
            continue
        filtered.append(entity)
    filtered.sort(
        key=lambda entity: (
            -float(score_entity(entity, label_counts, scene)),
            stable_hash(f"derived:{scene.scene_id}:{entity.entity_id}"),
        )
    )
    return filtered[:max_entities]


def augment_representation_stress_candidates(
    scenes: Sequence[SceneMetadata],
    all_candidates: List[Dict[str, Any]],
    scene_infos: Dict[str, SceneSideInfo],
    output_dir: Path,
    *,
    target_public_per_task: int,
    seed: int,
    enabled_tasks: Optional[set[str]],
) -> List[Dict[str, Any]]:
    by_task = Counter(row["task_id"] for row in all_candidates)
    derived_rows: List[Dict[str, Any]] = []
    existing_ids = {row["item_id"] for row in all_candidates}

    def add_row(row: Optional[Dict[str, Any]], derived_scene: SceneMetadata, info: SceneSideInfo) -> bool:
        if not row or row["item_id"] in existing_ids:
            return False
        row["diagnostic_slices"] = sorted(set(list(row.get("diagnostic_slices", [])) + ["derived_rotation"]))
        row.setdefault("metadata", {})
        row["metadata"]["derived_rotation"] = {
            "source_scene_id": info.scene_id,
            "derived_scene_id": derived_scene.scene_id,
        }
        derived_rows.append(row)
        existing_ids.add(row["item_id"])
        scene_infos[derived_scene.scene_id] = SceneSideInfo(
            scene_id=derived_scene.scene_id,
            group_id=f"{info.group_id}__derived",
            source_id=info.source_id,
            domain=info.domain,
            split_lock=info.split_lock,
        )
        return True

    absolute_targets = absolute_direction_target_counts(max(DERIVED_MIN_ABSOLUTE_DIRECTION_CANDIDATES, target_public_per_task))
    absolute_existing_rows = [row for row in all_candidates if row["task_id"] == "absolute_direction_mc"]
    absolute_existing_by_sector = Counter((row.get("metadata") or {}).get("sector") for row in absolute_existing_rows)
    absolute_needed_by_sector = {
        sector: max(0, absolute_targets[sector] - absolute_existing_by_sector.get(sector, 0))
        for sector in ABSOLUTE_DIRECTION_CHALLENGE_SECTORS
    }
    if task_enabled("absolute_direction_mc", enabled_tasks) and any(value > 0 for value in absolute_needed_by_sector.values()):
        produced_by_sector: Counter[str] = Counter()
        for scene in scenes:
            if not any(absolute_needed_by_sector[sector] > produced_by_sector[sector] for sector in ABSOLUTE_DIRECTION_CHALLENGE_SECTORS):
                break
            produced_for_scene = 0
            used_target_ids: set[str] = set()
            info = scene_infos.get(scene.scene_id) or build_scene_side_info(scene, {})
            targets = derived_candidate_entities(
                scene,
                predicate=direction_task_entity_eligible,
            )
            for target in targets:
                if produced_for_scene >= DERIVED_MAX_PER_SCENE_PER_TASK:
                    break
                if target.entity_id in used_target_ids:
                    continue
                for sector in sorted(
                    ABSOLUTE_DIRECTION_CHALLENGE_SECTORS,
                    key=lambda item: (
                        -(absolute_needed_by_sector[item] - produced_by_sector[item]),
                        abs(wrapped_delta_deg(yaw_deg_360(target) - ABSOLUTE_DIRECTION_CHALLENGE_CENTERS[item])),
                    ),
                ):
                    if produced_by_sector[sector] >= absolute_needed_by_sector[sector]:
                        continue
                    shift = choose_yaw_shift_for_absolute_direction_sector(target, sector)
                    suffix = f"absolute_{sector}_yaw_{int(round(shift))}_{target.entity_id}"
                    derived_scene = build_rotated_scene(scene, yaw_shift_deg=shift, suffix=suffix, output_dir=output_dir)
                    if derived_scene is None:
                        continue
                    derived_target = find_entity(derived_scene, target.entity_id)
                    if derived_target is None:
                        continue
                    quality = float(score_entity(derived_target, Counter(e.label for e in derived_scene.entities), derived_scene))
                    row = build_absolute_direction_mc(derived_scene, derived_target, 0, quality)
                    if row and (row.get("metadata") or {}).get("sector") != sector:
                        continue
                    if add_row(row, derived_scene, info):
                        produced_by_sector[sector] += 1
                        used_target_ids.add(target.entity_id)
                        produced_for_scene += 1
                        break

    seam_needed = max(0, max(DERIVED_MIN_SEAM_CANDIDATES, target_public_per_task) - by_task.get("seam_continuity_mc", 0))
    if task_enabled("seam_continuity_mc", enabled_tasks) and seam_needed > 0:
        produced = 0
        for scene in scenes:
            if produced >= seam_needed:
                break
            produced_for_scene = 0
            used_target_ids: set[str] = set()
            info = scene_infos.get(scene.scene_id) or build_scene_side_info(scene, {})
            targets = derived_candidate_entities(scene)
            for target in targets:
                if produced >= seam_needed or produced_for_scene >= DERIVED_MAX_PER_SCENE_PER_TASK:
                    break
                if target.entity_id in used_target_ids:
                    continue
                # Move the target to the ERP seam, not to the front center.
                shift = wrapped_delta_deg(yaw_deg_360(target) - 180.0)
                suffix = f"seam_yaw_{int(round(shift))}_{target.entity_id}"
                derived_scene = build_rotated_scene(scene, yaw_shift_deg=shift, suffix=suffix, output_dir=output_dir)
                if derived_scene is None:
                    continue
                derived_target = find_entity(derived_scene, target.entity_id)
                if derived_target is None:
                    continue
                quality = float(score_entity(derived_target, Counter(e.label for e in derived_scene.entities), derived_scene))
                for row in build_seam_continuity_items(derived_scene, derived_target, [], 0, quality):
                    if add_row(row, derived_scene, info):
                        produced += 1
                        used_target_ids.add(target.entity_id)
                if target.entity_id in used_target_ids:
                    produced_for_scene += 1

    polar_task_ids = [
        "polar_shape_recovery_mc",
        "polar_shape_matching_mc",
        "polar_cross_latitude_matching_mc",
    ]
    polar_needed_by_task = {
        task_id: max(0, max(DERIVED_MIN_POLAR_CANDIDATES, target_public_per_task) - by_task.get(task_id, 0))
        for task_id in polar_task_ids
    }
    if any(task_enabled(task_id, enabled_tasks) and value > 0 for task_id, value in polar_needed_by_task.items()):
        produced_by_task: Counter[str] = Counter()
        for scene in scenes:
            if all((not task_enabled(task_id, enabled_tasks)) or (produced_by_task[task_id] >= polar_needed_by_task[task_id]) for task_id in polar_task_ids):
                break
            produced_for_scene = 0
            used_target_ids: set[str] = set()
            info = scene_infos.get(scene.scene_id) or build_scene_side_info(scene, {})
            targets = derived_candidate_entities(scene)
            for target in targets:
                if all((not task_enabled(task_id, enabled_tasks)) or (produced_by_task[task_id] >= polar_needed_by_task[task_id]) for task_id in polar_task_ids):
                    break
                if produced_for_scene >= DERIVED_MAX_PER_SCENE_PER_TASK:
                    break
                if target.entity_id in used_target_ids:
                    continue
                if shape_value(target) is None:
                    continue
                shift = choose_pitch_shift_for_polar(scene, target)
                if shift is None:
                    continue
                suffix = f"polar_pitch_{int(round(shift))}_{target.entity_id}"
                derived_scene = build_rotated_scene(scene, pitch_shift_deg=shift, suffix=suffix, output_dir=output_dir)
                if derived_scene is None:
                    continue
                derived_target = find_entity(derived_scene, target.entity_id)
                if derived_target is None:
                    continue
                quality = float(score_entity(derived_target, Counter(e.label for e in derived_scene.entities), derived_scene))
                produced_any = False
                for row in [
                    build_polar_shape_recovery(derived_scene, derived_target, 0, quality),
                    build_polar_shape_matching(derived_scene, derived_target, 0, quality),
                    build_polar_cross_latitude_matching(derived_scene, derived_target, 0, quality),
                ]:
                    if row is None:
                        continue
                    task_id = str(row["task_id"])
                    if not task_enabled(task_id, enabled_tasks):
                        continue
                    if produced_by_task[task_id] >= polar_needed_by_task[task_id]:
                        continue
                    if add_row(row, derived_scene, info):
                        produced_by_task[task_id] += 1
                        produced_any = True
                if produced_any:
                    used_target_ids.add(target.entity_id)
                    produced_for_scene += 1

    return derived_rows


def build_referring_grounding_bfov(scene: SceneMetadata, target: Entity, anchors: Sequence[Dict[str, Any]], anchor_index: int, quality: float) -> Optional[Dict[str, Any]]:
    if target.resolved_bfov is None:
        return None
    if not reference_is_resolvable(scene, target):
        return None
    target_ref = contextual_entity_ref(scene, target)
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
    if not direction_task_entity_eligible(target):
        return None
    if not reference_is_resolvable(scene, target):
        return None
    sector = absolute_sector_8way(target)
    if sector not in ABSOLUTE_DIRECTION_CHALLENGE_SECTORS:
        return None
    margin = effective_direction_margin(target, absolute_sector_margin(target))
    if margin < ABSOLUTE_DIRECTION_MIN_MARGIN_DEG:
        return None
    neighbors = sector_distractors(sector)
    options = [sector] + neighbors[:3]
    choices, answer_key = shuffled_choice_rows(options, sector, f"{scene.scene_id}:{target.entity_id}:absolute_direction")
    target_ref = contextual_entity_ref(scene, target)
    question = pick_template("absolute_direction_mc", f"{scene.scene_id}:{target.entity_id}").format(target_ref=target_ref)
    return benchmark_item(
        scene=scene,
        task_id="absolute_direction_mc",
        item_id=f"{scene.scene_id}_absolute_direction_mc_{target.entity_id}",
        question=question,
        answer=answer_key,
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
    # For absolute-direction benchmark items we avoid nearest-neighbor sectors,
    # because labels such as "left" vs "front-left" remain linguistically
    # plausible even when the target is filtered away from the exact boundary.
    # We instead prefer sectors that are clearly separated in panoramic angle.
    order = [
        ABSOLUTE_SECTORS_8[(idx + 4) % 8],
        ABSOLUTE_SECTORS_8[(idx + 2) % 8],
        ABSOLUTE_SECTORS_8[(idx - 2) % 8],
        ABSOLUTE_SECTORS_8[(idx + 3) % 8],
    ]
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


def relation_pair_boundary_metadata(scene: SceneMetadata, reference: Entity, target: Entity) -> Dict[str, Any]:
    reference_side = seam_primary_side(reference, scene)
    target_side = seam_primary_side(target, scene)
    boundary_pair = reference_side is not None and target_side is not None
    cross_boundary_pair = boundary_pair and reference_side != target_side
    return {
        "reference_boundary_side": reference_side or "",
        "target_boundary_side": target_side or "",
        "boundary_pair": bool(boundary_pair),
        "cross_boundary_pair": bool(cross_boundary_pair),
    }


def relation_selection_sort_key(row: Dict[str, Any], seed: int) -> Tuple[Any, ...]:
    metadata = row.get("metadata") or {}
    cross_boundary = bool(metadata.get("cross_boundary_pair"))
    boundary_pair = bool(metadata.get("boundary_pair"))
    return (
        0 if cross_boundary else 1 if boundary_pair else 2,
        -float(row.get("quality_score", 0.0)),
        stable_hash(f"{seed}:{row['item_id']}"),
    )


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
    "same object at the same place in the 360 scene",
    "same object at different places in the 360 scene",
    "different objects at the same place in the 360 scene",
    "different objects at different places in the 360 scene",
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
        "In the 360 scene, the left-edge and right-edge appearances of {target_ref} are best described as:",
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
    if not reference_is_resolvable(scene, target):
        return None
    bundles = seam_wrap_candidate_sets(scene, target)
    if bundles is None:
        return None
    correct = bundles["correct"]
    lure = bundles["lure"]
    distractors = bundles["distractors"]
    target_side = bundles["target_side"]

    option_entities = [correct, lure] + distractors
    item_key = f"{scene.scene_id}:seam_nearest_neighbor:{target.entity_id}:{correct.entity_id}:{lure.entity_id}"
    target_ref = contextual_entity_ref(scene, target)
    correct_ref = contextual_entity_ref(scene, correct)
    lure_ref = contextual_entity_ref(scene, lure)
    distractor_refs = [contextual_entity_ref(scene, entity) for entity in distractors]
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
    if not reference_is_resolvable(scene, target):
        return None
    bundles = seam_wrap_candidate_sets(scene, target)
    if bundles is None:
        return None
    correct = bundles["correct"]
    item_key = f"{scene.scene_id}:seam_relative_direction:{target.entity_id}:{correct.entity_id}"
    neighbor_ref = contextual_entity_ref(scene, correct)
    target_ref = contextual_entity_ref(scene, target)
    question = pick_variant(
        SEAM_SUBTYPE_TEMPLATES["relative_direction"],
        f"{item_key}:question",
    ).format(
        neighbor_ref=neighbor_ref,
        target_ref=target_ref,
    )
    answer_text = "adjacent across the boundary"
    choices, answer_key = shuffled_choice_rows(
        SEAM_RELATION_OPTIONS,
        answer_text,
        f"seam_continuity_mc:relative_direction:{item_key}",
    )
    return benchmark_item(
        scene=scene,
        task_id="seam_continuity_mc",
        item_id=f"{scene.scene_id}_seam_continuity_mc_relative_direction_{target.entity_id}_{correct.entity_id}",
        question=question,
        answer=answer_key,
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
    if not reference_is_resolvable(scene, target):
        return None
    side = seam_contact_side(target, scene)
    if side is None:
        return None
    if not bool(target.seam_crossing_flag):
        return None
    item_key = f"{scene.scene_id}:seam_dedup_count:{target.entity_id}"
    target_ref = contextual_entity_ref(scene, target)
    question = pick_variant(
        SEAM_SUBTYPE_TEMPLATES["dedup_count"],
        f"{item_key}:question",
    ).format(
        target_ref=target_ref,
    )
    answer_text = "one continuous object"
    choices, answer_key = shuffled_choice_rows(
        SEAM_DEDUP_OPTIONS,
        answer_text,
        f"seam_continuity_mc:dedup_count:{item_key}",
    )
    return benchmark_item(
        scene=scene,
        task_id="seam_continuity_mc",
        item_id=f"{scene.scene_id}_seam_continuity_mc_dedup_count_{target.entity_id}",
        question=question,
        answer=answer_key,
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
    if not reference_is_resolvable(scene, target):
        return None
    side = seam_contact_side(target, scene)
    if side is None or not bool(target.seam_crossing_flag):
        return None
    if not seam_structure_like(target):
        return None
    item_key = f"{scene.scene_id}:seam_structure_continuity:{target.entity_id}"
    target_ref = contextual_entity_ref(scene, target)
    question = pick_variant(
        SEAM_SUBTYPE_TEMPLATES["structure_continuity"],
        f"{item_key}:question",
    ).format(
        target_ref=target_ref,
    )
    answer_text = "one continuous structure"
    choices, answer_key = shuffled_choice_rows(
        SEAM_STRUCTURE_OPTIONS,
        answer_text,
        f"seam_continuity_mc:structure_continuity:{item_key}",
    )
    return benchmark_item(
        scene=scene,
        task_id="seam_continuity_mc",
        item_id=f"{scene.scene_id}_seam_continuity_mc_structure_continuity_{target.entity_id}",
        question=question,
        answer=answer_key,
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
    if not reference_is_resolvable(scene, target):
        return None
    side = seam_contact_side(target, scene)
    if side is None or not bool(target.seam_crossing_flag):
        return None
    item_key = f"{scene.scene_id}:seam_same_entity:{target.entity_id}"
    target_ref = contextual_entity_ref(scene, target)
    question = pick_variant(
        SEAM_SUBTYPE_TEMPLATES["same_entity_judgement"],
        f"{item_key}:question",
    ).format(
        target_ref=target_ref,
    )
    answer_text = "same object at the same place in the 360 scene"
    choices, answer_key = shuffled_choice_rows(
        SEAM_SAME_ENTITY_OPTIONS,
        answer_text,
        f"seam_continuity_mc:same_entity:{item_key}",
    )
    return benchmark_item(
        scene=scene,
        task_id="seam_continuity_mc",
        item_id=f"{scene.scene_id}_seam_continuity_mc_same_entity_{target.entity_id}",
        question=question,
        answer=answer_key,
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
    if not direction_task_entity_eligible(reference) or not direction_task_entity_eligible(target):
        return None
    if not reference_is_resolvable(scene, reference) or not reference_is_resolvable(scene, target):
        return None
    delta = wrapped_delta_deg(yaw_deg_360(target) - yaw_deg_360(reference))
    relation = panoramic_relation_from_delta(delta, opposite_label="opposite")
    if relation is None:
        return None
    raw_margin = closest_boundary_margin(abs(delta), [15.0, 90.0, 150.0, 180.0])
    half_ref = abs(float((reference.resolved_bfov or (0.0, 0.0, 0.0, 0.0))[2])) / 2.0
    half_target = abs(float((target.resolved_bfov or (0.0, 0.0, 0.0, 0.0))[2])) / 2.0
    margin = raw_margin - max(half_ref, half_target)
    if margin < RELATION_MIN_MARGIN_DEG:
        return None
    item_key = f"{scene.scene_id}:relative_direction_mc:{reference.entity_id}:{target.entity_id}"
    reference_ref = contextual_entity_ref(scene, reference)
    target_ref = contextual_entity_ref(scene, target)
    boundary_meta = relation_pair_boundary_metadata(scene, reference, target)
    question = pick_template("relative_direction_mc", f"{scene.scene_id}:{reference.entity_id}:{target.entity_id}").format(
        reference_ref=reference_ref,
        target_ref=target_ref,
    )
    choices, answer_key = shuffled_choice_rows(
        PANORAMIC_RELATION_LABELS,
        relation,
        f"{scene.scene_id}:{reference.entity_id}:{target.entity_id}:relative_direction",
    )
    return benchmark_item(
        scene=scene,
        task_id="relative_direction_mc",
        item_id=f"{scene.scene_id}_relative_direction_mc_{reference.entity_id}_{target.entity_id}",
        question=question,
        answer=answer_key,
        answer_text=relation,
        options=choices,
        target_entities=[reference.entity_id, target.entity_id],
        metadata={
            "reference_ref": reference_ref,
            "target_ref": target_ref,
            "reference_yaw_deg": round(yaw_deg_360(reference), 1),
            "target_yaw_deg": round(yaw_deg_360(target), 1),
            "delta_yaw_deg": round(delta, 2),
            **boundary_meta,
        },
        difficulty=difficulty_from_margin(margin),
        quality_score=(quality + float(score_entity(target, Counter(e.label for e in scene.entities), scene))) / 2.0,
        diagnostic_slices=dedupe_keep_order(
            slices_for_entity(reference)
            + slices_for_entity(target)
            + (["boundary_pair"] if boundary_meta["boundary_pair"] else [])
        ),
    )


def build_camera_rotation_transform_mc(scene: SceneMetadata, target: Entity, anchor_index: int, partner_index: int, quality: float) -> Optional[Dict[str, Any]]:
    if not direction_task_entity_eligible(target):
        return None
    if not reference_is_resolvable(scene, target):
        return None
    rotation_direction, angle_deg = ROTATION_OPTIONS[stable_hash(f"{scene.scene_id}:{target.entity_id}:rotation") % len(ROTATION_OPTIONS)]
    relation = camera_rotation_relation(target, rotation_direction, angle_deg)
    if relation is None:
        return None
    raw_delta = wrapped_delta_deg(yaw_deg_360(target) - ((float(angle_deg) if rotation_direction == "right" else -float(angle_deg)) % 360.0))
    margin = effective_direction_margin(target, closest_boundary_margin(abs(raw_delta), [15.0, 90.0, 150.0, 180.0]))
    if margin < RELATION_MIN_MARGIN_DEG:
        return None
    item_key = f"{scene.scene_id}:camera_rotation_transform_mc:{target.entity_id}:{rotation_direction}:{angle_deg}"
    target_ref = contextual_entity_ref(scene, target)
    question = pick_template("camera_rotation_transform_mc", f"{scene.scene_id}:{target.entity_id}:{rotation_direction}:{angle_deg}").format(
        angle_deg=angle_deg,
        turn_direction=rotation_direction,
        target_ref=target_ref,
    )
    choices, answer_key = shuffled_choice_rows(
        REORIENTED_RELATION_LABELS,
        relation,
        f"{scene.scene_id}:{target.entity_id}:{rotation_direction}:{angle_deg}:camera_rotation",
    )
    return benchmark_item(
        scene=scene,
        task_id="camera_rotation_transform_mc",
        item_id=f"{scene.scene_id}_camera_rotation_transform_mc_{target.entity_id}_{rotation_direction}_{angle_deg}",
        question=question,
        answer=answer_key,
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
    if not direction_task_entity_eligible(facing) or not direction_task_entity_eligible(target):
        return None
    if not reference_is_resolvable(scene, facing) or not reference_is_resolvable(scene, target):
        return None
    delta = wrapped_delta_deg(yaw_deg_360(target) - yaw_deg_360(facing))
    relation = panoramic_relation_from_delta(delta, opposite_label="behind")
    if relation is None:
        return None
    raw_margin = closest_boundary_margin(abs(delta), [15.0, 90.0, 150.0, 180.0])
    half_facing = abs(float((facing.resolved_bfov or (0.0, 0.0, 0.0, 0.0))[2])) / 2.0
    half_target = abs(float((target.resolved_bfov or (0.0, 0.0, 0.0, 0.0))[2])) / 2.0
    margin = raw_margin - max(half_facing, half_target)
    if margin < RELATION_MIN_MARGIN_DEG:
        return None
    item_key = f"{scene.scene_id}:object_conditioned_reorientation_mc:{facing.entity_id}:{target.entity_id}"
    facing_ref = contextual_entity_ref(scene, facing)
    target_ref = contextual_entity_ref(scene, target)
    boundary_meta = relation_pair_boundary_metadata(scene, facing, target)
    question = pick_template("object_conditioned_reorientation_mc", f"{scene.scene_id}:{facing.entity_id}:{target.entity_id}").format(
        facing_ref=facing_ref,
        target_ref=target_ref,
    )
    choices, answer_key = shuffled_choice_rows(
        REORIENTED_RELATION_LABELS,
        relation,
        f"{scene.scene_id}:{facing.entity_id}:{target.entity_id}:object_conditioned_reorientation",
    )
    return benchmark_item(
        scene=scene,
        task_id="object_conditioned_reorientation_mc",
        item_id=f"{scene.scene_id}_object_conditioned_reorientation_mc_{facing.entity_id}_{target.entity_id}",
        question=question,
        answer=answer_key,
        answer_text=relation,
        options=choices,
        target_entities=[facing.entity_id, target.entity_id],
        metadata={
            "facing_ref": facing_ref,
            "target_ref": target_ref,
            "facing_yaw_deg": round(yaw_deg_360(facing), 1),
            "target_yaw_deg": round(yaw_deg_360(target), 1),
            "delta_yaw_deg": round(delta, 2),
            **boundary_meta,
        },
        difficulty=difficulty_from_margin(margin),
        quality_score=(quality + float(score_entity(target, Counter(e.label for e in scene.entities), scene))) / 2.0,
        diagnostic_slices=dedupe_keep_order(
            slices_for_entity(facing)
            + slices_for_entity(target)
            + (["boundary_pair"] if boundary_meta["boundary_pair"] else [])
        ),
    )


def build_polar_shape_recovery(scene: SceneMetadata, target: Entity, anchor_index: int, quality: float) -> Optional[Dict[str, Any]]:
    shape = shape_value(target)
    if not shape:
        return None
    if not polar_target_eligible(target):
        return None
    target_ref = polar_entity_ref(scene, target)
    distractors = SHAPE_DISTRACTOR_MAP.get(shape, [item for item in SHAPE_FALLBACK_POOL if item != shape][:3])
    if len(distractors) < 3:
        return None
    options = [shape] + distractors
    choices, answer_key = shuffled_choice_rows(options, shape, f"{scene.scene_id}:{target.entity_id}:polar_shape_recovery")
    question = pick_template("polar_shape_recovery_mc", f"{scene.scene_id}:{target.entity_id}").format(target_ref=target_ref)
    return benchmark_item(
        scene=scene,
        task_id="polar_shape_recovery_mc",
        item_id=f"{scene.scene_id}_polar_shape_recovery_mc_{target.entity_id}",
        question=question,
        answer=answer_key,
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


def build_polar_shape_matching(scene: SceneMetadata, target: Entity, anchor_index: int, quality: float) -> Optional[Dict[str, Any]]:
    target_shape = shape_value(target)
    if not target_shape or not polar_target_eligible(target):
        return None
    pool = polar_option_entities(scene, target, low_lat_only=False)
    correct_pool = [entity for entity in pool if shape_value(entity) == target_shape]
    if not correct_pool:
        return None
    preferred_distractor_shapes = SHAPE_DISTRACTOR_MAP.get(target_shape, [])
    distractor_pool = [entity for entity in pool if shape_value(entity) != target_shape]
    distractor_pool.sort(
        key=lambda entity: (
            0 if shape_value(entity) in preferred_distractor_shapes else 1,
            preferred_distractor_shapes.index(shape_value(entity)) if shape_value(entity) in preferred_distractor_shapes else 999,
        )
    )
    selected_entities = [correct_pool[0]]
    used_shapes = {target_shape}
    for entity in distractor_pool:
        entity_shape = shape_value(entity)
        if entity_shape is None or entity_shape in used_shapes:
            continue
        selected_entities.append(entity)
        used_shapes.add(entity_shape)
        if len(selected_entities) == 4:
            break
    if len(selected_entities) < 4:
        return None
    option_rows = unique_polar_entity_refs(scene, selected_entities)
    if len(option_rows) < 4:
        return None
    option_rows = option_rows[:4]
    option_texts = [ref for _, ref in option_rows]
    correct_text = option_rows[0][1]
    choices, answer_key = shuffled_choice_rows(option_texts, correct_text, f"{scene.scene_id}:{target.entity_id}:polar_shape_matching")
    target_ref = polar_entity_ref(scene, target)
    question = pick_template("polar_shape_matching_mc", f"{scene.scene_id}:{target.entity_id}").format(target_ref=target_ref)
    return benchmark_item(
        scene=scene,
        task_id="polar_shape_matching_mc",
        item_id=f"{scene.scene_id}_polar_shape_matching_mc_{target.entity_id}",
        question=question,
        answer=answer_key,
        answer_text=correct_text,
        options=choices,
        target_entities=[target.entity_id] + [entity.entity_id for entity, _ in option_rows],
        metadata={
            "target_ref": target_ref,
            "target_shape": target_shape,
            "target_bfov": list(target.resolved_bfov or ()),
            "candidate_entity_ids": [entity.entity_id for entity, _ in option_rows],
            "candidate_shapes": [shape_value(entity) for entity, _ in option_rows],
        },
        difficulty="hard",
        quality_score=quality,
        diagnostic_slices=dedupe_keep_order(slices_for_entity(target) + ["pole"]),
        requires_manual_review=True,
        review_notes=["Verify that the matching candidates remain visually distinguishable and that the target shape is recoverable under polar distortion."],
    )


def build_polar_cross_latitude_matching(scene: SceneMetadata, target: Entity, anchor_index: int, quality: float) -> Optional[Dict[str, Any]]:
    target_shape = shape_value(target)
    if not target_shape or not polar_target_eligible(target):
        return None
    pool = polar_option_entities(scene, target, low_lat_only=True)
    correct_pool = [entity for entity in pool if shape_value(entity) == target_shape]
    if not correct_pool:
        return None
    distractor_pool = [entity for entity in pool if shape_value(entity) != target_shape]
    selected_entities = [correct_pool[0]]
    used_shapes = {target_shape}
    for entity in distractor_pool:
        entity_shape = shape_value(entity)
        if entity_shape is None or entity_shape in used_shapes:
            continue
        selected_entities.append(entity)
        used_shapes.add(entity_shape)
        if len(selected_entities) == 4:
            break
    if len(selected_entities) < 4:
        return None
    option_rows = unique_polar_entity_refs(scene, selected_entities)
    if len(option_rows) < 4:
        return None
    option_rows = option_rows[:4]
    option_texts = [ref for _, ref in option_rows]
    correct_text = option_rows[0][1]
    choices, answer_key = shuffled_choice_rows(option_texts, correct_text, f"{scene.scene_id}:{target.entity_id}:polar_cross_latitude_matching")
    target_ref = polar_entity_ref(scene, target)
    question = pick_template("polar_cross_latitude_matching_mc", f"{scene.scene_id}:{target.entity_id}").format(target_ref=target_ref)
    return benchmark_item(
        scene=scene,
        task_id="polar_cross_latitude_matching_mc",
        item_id=f"{scene.scene_id}_polar_cross_latitude_matching_mc_{target.entity_id}",
        question=question,
        answer=answer_key,
        answer_text=correct_text,
        options=choices,
        target_entities=[target.entity_id] + [entity.entity_id for entity, _ in option_rows],
        metadata={
            "target_ref": target_ref,
            "target_shape": target_shape,
            "target_bfov": list(target.resolved_bfov or ()),
            "candidate_entity_ids": [entity.entity_id for entity, _ in option_rows],
            "candidate_shapes": [shape_value(entity) for entity, _ in option_rows],
            "candidate_abs_lat_deg": [round(abs(entity.lat_deg), 2) for entity, _ in option_rows],
        },
        difficulty="hard",
        quality_score=quality,
        diagnostic_slices=dedupe_keep_order(slices_for_entity(target) + ["pole"]),
        requires_manual_review=True,
        review_notes=["Verify that the correct option is clearly from a lower-distortion latitude band while still matching the target geometry."],
    )


def build_observer_distance_choice(scene: SceneMetadata, anchors: Sequence[Dict[str, Any]], anchor_index: int) -> Optional[Dict[str, Any]]:
    candidates = [item["entity"] for item in anchors if item["entity"].entity_center_depth is not None]
    if len(candidates) < 3:
        return None
    candidates = sorted(candidates, key=lambda entity: float(entity.entity_center_depth))[:6]
    selected = candidates[:4] if len(candidates) >= 4 else candidates[:3]
    if any(not reference_is_resolvable(scene, entity) for entity in selected):
        return None
    depths = [float(entity.entity_center_depth) for entity in selected]
    if min(abs(depths[i] - depths[i + 1]) for i in range(len(depths) - 1)) < 0.35:
        return None
    closest = min(selected, key=lambda entity: float(entity.entity_center_depth))
    item_key = f"{scene.scene_id}:observer_distance_choice:{selected[0].entity_id}:{selected[-1].entity_id}"
    option_texts = [contextual_entity_ref(scene, entity) for entity in selected]
    correct_text = option_texts[[entity.entity_id for entity in selected].index(closest.entity_id)]
    choices, answer_key = shuffled_choice_rows(option_texts, correct_text, item_key)
    question = pick_template("observer_distance_choice", f"{scene.scene_id}:{selected[0].entity_id}:{selected[-1].entity_id}")
    avg_quality = sum(float(score_entity(entity, Counter(e.label for e in scene.entities), scene)) for entity in selected) / len(selected)
    return benchmark_item(
        scene=scene,
        task_id="observer_distance_choice",
        item_id=f"{scene.scene_id}_observer_distance_choice_{selected[0].entity_id}_{selected[-1].entity_id}",
        question=question,
        answer=answer_key,
        answer_text=correct_text,
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
    if not reference_is_resolvable(scene, entity_a) or not reference_is_resolvable(scene, entity_b):
        return None
    relation, parts = relative_3d_relation(entity_a, entity_b)
    if not relation:
        return None
    item_key = f"{scene.scene_id}:relative_3d_position_mc:{entity_a.entity_id}:{entity_b.entity_id}"
    entity_a_ref = contextual_entity_ref(scene, entity_a)
    entity_b_ref = contextual_entity_ref(scene, entity_b)
    options = relative_3d_choices(entity_a, entity_b, relation, parts)
    choices, answer_key = shuffled_choice_rows(options, relation, item_key)
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
        answer=answer_key,
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
    def scene_cap_reached(counter: Counter[str], scene_id: str) -> bool:
        return max_per_scene_per_task > 0 and counter[scene_id] >= max_per_scene_per_task

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
        if task_id == "absolute_direction_mc":
            quota_by_sector = absolute_direction_target_counts(target_per_task)
            rows_by_sector: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for row in rows:
                sector = str((row.get("metadata") or {}).get("sector", ""))
                rows_by_sector[sector].append(row)

            per_scene_counter: Counter[str] = Counter()
            kept = 0
            kept_by_sector: Counter[str] = Counter()
            for sector in ["back-right", "back", "back-left", "left", "right"]:
                for row in rows_by_sector.get(sector, []):
                    if kept >= target_per_task or kept_by_sector[sector] >= quota_by_sector.get(sector, 0):
                        break
                    if scene_cap_reached(per_scene_counter, str(row["scene_id"])):
                        continue
                    selected.append(row)
                    per_scene_counter[row["scene_id"]] += 1
                    kept += 1
                    kept_by_sector[sector] += 1
            if kept < target_per_task:
                for row in rows:
                    if kept >= target_per_task:
                        break
                    if row in selected:
                        continue
                    if scene_cap_reached(per_scene_counter, str(row["scene_id"])):
                        continue
                    selected.append(row)
                    per_scene_counter[row["scene_id"]] += 1
                    kept += 1
            continue
        if task_id in {"relative_direction_mc", "object_conditioned_reorientation_mc"}:
            quota_by_label = weighted_target_counts(
                target_per_task,
                RELATIVE_DIRECTION_PUBLIC_WEIGHTS if task_id == "relative_direction_mc" else OBJECT_REORIENTATION_PUBLIC_WEIGHTS,
            )
            rows_by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for row in rows:
                answer_text = str(row.get("answer_text", ""))
                rows_by_label[answer_text].append(row)
            for label_rows in rows_by_label.values():
                label_rows.sort(key=lambda row: relation_selection_sort_key(row, seed))

            per_scene_counter = Counter()
            kept = 0
            kept_by_label: Counter[str] = Counter()
            ordered_labels = ["back-right", "opposite" if task_id == "relative_direction_mc" else "behind", "back-left", "left", "right"]
            for label in ordered_labels:
                for row in rows_by_label.get(label, []):
                    if kept >= target_per_task or kept_by_label[label] >= quota_by_label.get(label, 0):
                        break
                    if scene_cap_reached(per_scene_counter, str(row["scene_id"])):
                        continue
                    selected.append(row)
                    per_scene_counter[row["scene_id"]] += 1
                    kept += 1
                    kept_by_label[label] += 1
            if kept < target_per_task:
                rows.sort(key=lambda row: relation_selection_sort_key(row, seed))
                for row in rows:
                    if kept >= target_per_task:
                        break
                    if row in selected:
                        continue
                    if scene_cap_reached(per_scene_counter, str(row["scene_id"])):
                        continue
                    selected.append(row)
                    per_scene_counter[row["scene_id"]] += 1
                    kept += 1
            continue
        per_scene_counter: Counter[str] = Counter()
        kept = 0
        for row in rows:
            if kept >= target_per_task:
                break
            if scene_cap_reached(per_scene_counter, str(row["scene_id"])):
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
    skipped_invalid_metadata: Sequence[Dict[str, Any]],
    num_input_scenes: int,
) -> Dict[str, Any]:
    def task_counter(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        counts = Counter(row["task_id"] for row in rows)
        return {task_id: int(counts.get(task_id, 0)) for task_id in sorted(TASK_SPECS)}

    def group_counter(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        counts = Counter(row["ability_group"] for row in rows)
        return dict(sorted(counts.items()))

    def answer_key_counter(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        grouped: Dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            options = row.get("options") or []
            answer = str(row.get("answer") or "")
            if not options or not answer:
                continue
            grouped[str(row["task_id"])][answer] += 1
        return {
            task_id: {key: int(counter.get(key, 0)) for key in sorted(counter)}
            for task_id, counter in sorted(grouped.items())
        }

    derived_rows = [row for row in all_candidates if "derived_rotation" in (row.get("metadata") or {})]
    natural_rows = [row for row in all_candidates if "derived_rotation" not in (row.get("metadata") or {})]
    derived_scene_ids = {
        str(row["metadata"]["derived_rotation"]["derived_scene_id"])
        for row in derived_rows
        if "derived_rotation" in (row.get("metadata") or {})
    }

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
        "num_scenes": num_input_scenes,
        "derived_scene_count": len(derived_scene_ids),
        "skipped_invalid_metadata_count": len(skipped_invalid_metadata),
        "derived_rotation_count": len(derived_rows),
        "candidate_pool_size": len(all_candidates),
        "review_queue_size": len(review_queue),
        "benchmark_public_size": len(public_selected),
        "natural_candidate_per_task": task_counter(natural_rows),
        "derived_candidate_per_task": task_counter(derived_rows),
        "candidate_per_task": task_counter(all_candidates),
        "benchmark_public_per_task": task_counter(public_selected),
        "benchmark_public_per_group": group_counter(public_selected),
        "benchmark_public_answer_key_distribution": answer_key_counter(public_selected),
        "skipped_invalid_metadata_examples": list(skipped_invalid_metadata[:10]),
        "derived_rotation_examples": [
            {
                "item_id": row["item_id"],
                "task_id": row["task_id"],
                "source_scene_id": row["metadata"]["derived_rotation"]["source_scene_id"],
                "derived_scene_id": row["metadata"]["derived_rotation"]["derived_scene_id"],
            }
            for row in derived_rows[:10]
        ],
        "scene_metadata_overview": dict(sorted(split_counts.items())),
        "leakage_controls": {
            "training_overlap_expected": "no_scene_overlap_by_construction",
            "template_overlap_control": "benchmark_template_family_separate_from_training_templates",
            "selection_overlap_control": "optional_per_scene_per_task_cap_with_public_release_only",
            "manual_review_required_for_fragile_tasks": sorted(MANUAL_REVIEW_TASKS),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
