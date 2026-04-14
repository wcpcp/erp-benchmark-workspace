#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from erp_spatial_benchmark._vendor.schemas import SceneMetadata
from erp_spatial_benchmark.build_benchmark import (
    erp_x_to_yaw,
    erp_y_to_pitch,
    find_entity,
    pitch_to_erp_y,
    pick_variant,
    rotate_yaw_pitch,
    write_pitch_rotated_erp_image,
    write_yaw_shifted_erp_image,
    yaw_to_erp_x,
)

POLAR_VISUAL_TEMPLATES = [
    "What is the actual shape of the objects drawn in red box in this ERP panorama?",
    "Which shape best matches the real shape of the object drawn in the red box?",
]
ROTATION_PATTERNS = [
    re.compile(r"__observer_distance_rot_y(?P<yaw>-?\d+)_p(?P<pitch>-?\d+)_"),
    re.compile(r"__polar_pitch_(?P<pitch>-?\d+)_"),
    re.compile(r"__[^_].*?_yaw_(?P<yaw>-?\d+)_"),
]
SOURCE_SCENE_PATTERNS = [
    re.compile(r"^(?P<source>.+)__observer_distance_rot_y-?\d+_p-?\d+_E[^_]+$"),
    re.compile(r"^(?P<source>.+)__polar_pitch_-?\d+_E[^_]+$"),
    re.compile(r"^(?P<source>.+)__[^_].*?_yaw_-?\d+_E[^_]+$"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite polar_shape_recovery_mc items to use a red-box visual prompt instead of textual target references."
    )
    parser.add_argument("--input-jsonl", required=True, help="Filtered benchmark JSONL to rewrite.")
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
        help="Directory for generated red-box images. Defaults to <output-jsonl-dir>/polar_shape_visual_prompt.",
    )
    parser.add_argument(
        "--image-search-roots",
        nargs="*",
        default=[],
        help="Optional fallback roots to search by image filename when item.image_path is missing. Found files are copied back to the original expected path before rewriting.",
    )
    parser.add_argument("--box-color", default="#ff2d2d")
    parser.add_argument("--box-width", type=int, default=6)
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


def discover_scene_metadata_paths(roots: Sequence[Path]) -> Dict[str, List[Path]]:
    scene_index: Dict[str, List[Path]] = {}
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
                if scene_id:
                    scene_index.setdefault(scene_id, [])
                    if path not in scene_index[scene_id]:
                        scene_index[scene_id].append(path)
            except Exception:
                continue
    return scene_index


def load_scene_from_path(path: Path) -> SceneMetadata:
    return SceneMetadata.from_dict(load_json(path))


def scene_candidate_paths(scene_index: Dict[str, List[Path]], key: str) -> List[Path]:
    if not key:
        return []
    return list(scene_index.get(key, []))


def resolve_scene_metadata_path(item: Dict[str, Any], scene_index: Dict[str, List[Path]]) -> Optional[Path]:
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
        candidates = scene_candidate_paths(scene_index, key)
        if candidates:
            return candidates[0]
    return None


def discover_image_paths(roots: Sequence[Path]) -> Dict[str, Path]:
    image_index: Dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            candidates = [root]
        else:
            candidates = [path for path in root.rglob("*") if path.is_file()]
        for path in candidates:
            image_index.setdefault(path.name, path)
    return image_index


def summarize_search_roots(roots: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for root in roots:
        summary[str(root)] = {
            "exists": root.exists(),
            "is_file": root.is_file(),
            "is_dir": root.is_dir(),
        }
    return summary


def restore_missing_image(expected_path: Path, image_index: Dict[str, Path]) -> Optional[Path]:
    if expected_path.exists():
        return expected_path
    source = image_index.get(expected_path.name)
    if source is None or not source.exists():
        return None
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, expected_path)
    return expected_path


def parse_rotation_from_name(text: str) -> Optional[Tuple[float, float]]:
    for pattern in ROTATION_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        yaw = float(match.groupdict().get("yaw") or 0.0)
        pitch = float(match.groupdict().get("pitch") or 0.0)
        if abs(yaw) > 1e-6 or abs(pitch) > 1e-6:
            return yaw, pitch
    return None


def parse_source_scene_id_from_name(scene_name: str) -> str:
    for pattern in SOURCE_SCENE_PATTERNS:
        match = pattern.match(scene_name)
        if match is not None:
            return str(match.group("source"))
    return ""


def expected_x_center_from_scene_name(scene_name: str) -> Optional[float]:
    sector_map = {
        "front": 0.50,
        "front-right": 0.625,
        "right": 0.75,
        "back-right": 0.875,
        "back": 0.00,
        "back-left": 0.125,
        "left": 0.25,
        "front-left": 0.375,
    }
    match = re.search(r"__absolute_(?P<sector>front-right|front-left|back-right|back-left|front|right|back|left)_yaw_", scene_name)
    if match is None:
        if "__seam_yaw_" in scene_name:
            return 0.0
        return None
    return sector_map.get(match.group("sector"))


def wrap_distance(a: float, b: float) -> float:
    return min(abs(a - b), 1.0 - abs(a - b))


def box_center_x_norm(box: Sequence[float], width: float) -> float:
    x1, _, x2, _ = [float(v) for v in box]
    if x1 <= x2:
        center = (x1 + x2) / 2.0
    else:
        span = ((x2 + width) - x1) / 2.0
        center = (x1 + span) % width
    return center / width if width > 0 else 0.0


def shifted_bbox_yaw_only(entity_box: Sequence[float], scene: SceneMetadata, *, yaw_shift_deg: float) -> Optional[Sequence[float]]:
    width = int(scene.erp_width or 0)
    if width <= 0 or len(entity_box) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in entity_box]
    shift_px = (float(yaw_shift_deg) / 360.0) * float(width)
    new_x1 = (x1 - shift_px) % float(width)
    new_x2 = (x2 - shift_px) % float(width)
    return [new_x1, y1, new_x2, y2]


def bfov_box_from_rotated_entity(
    scene: SceneMetadata,
    entity,
    *,
    yaw_shift_deg: float = 0.0,
    pitch_shift_deg: float = 0.0,
) -> Optional[Sequence[float]]:
    bfov = entity.resolved_bfov
    width = int(scene.erp_width or 0)
    height = int(scene.erp_height or 0)
    if bfov is None or width <= 0 or height <= 0:
        return None
    yaw_deg, pitch_deg, x_fov_deg, y_fov_deg = [float(v) for v in bfov]
    yaw_new, pitch_new = rotate_yaw_pitch(
        yaw_deg,
        pitch_deg,
        yaw_shift_deg=yaw_shift_deg,
        pitch_shift_deg=pitch_shift_deg,
    )
    x_center = yaw_to_erp_x(yaw_new, width) % float(width)
    y_center = min(float(height - 1), max(0.0, pitch_to_erp_y(pitch_new, height)))
    half_w = (float(x_fov_deg) / 360.0) * float(width) / 2.0
    half_h = (float(y_fov_deg) / 180.0) * float(height) / 2.0
    x1 = (x_center - half_w) % float(width)
    x2 = (x_center + half_w) % float(width)
    y1 = max(0.0, y_center - half_h)
    y2 = min(float(height - 1), y_center + half_h)
    return [x1, y1, x2, y2]


def rebuild_rotated_image(src_image: Path, dst_image: Path, *, yaw_shift_deg: float, pitch_shift_deg: float) -> None:
    if abs(yaw_shift_deg) > 1e-6 and abs(pitch_shift_deg) > 1e-6:
        tmp_dir = dst_image.parent / "_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_image = tmp_dir / f"{dst_image.stem}__yaw_tmp{dst_image.suffix}"
        write_yaw_shifted_erp_image(src_image, tmp_image, yaw_shift_deg)
        write_pitch_rotated_erp_image(tmp_image, dst_image, pitch_shift_deg)
        tmp_image.unlink(missing_ok=True)
    elif abs(yaw_shift_deg) > 1e-6:
        write_yaw_shifted_erp_image(src_image, dst_image, yaw_shift_deg)
    elif abs(pitch_shift_deg) > 1e-6:
        write_pitch_rotated_erp_image(src_image, dst_image, pitch_shift_deg)
    else:
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_image, dst_image)


def rebuild_missing_derived_image(
    expected_path: Path,
    item: Dict[str, Any],
    scene_index: Dict[str, List[Path]],
    image_index: Dict[str, Path],
) -> Tuple[Optional[Path], Optional[str]]:
    metadata = item.get("metadata") or {}
    source_scene_id = str((metadata.get("derived_rotation") or {}).get("source_scene_id", "")).strip()
    if not source_scene_id:
        return None, "no_source_scene_id"
    source_candidates = scene_candidate_paths(scene_index, source_scene_id)
    if not source_candidates:
        return None, "source_scene_metadata_not_found"
    source_scene = load_scene_from_path(source_candidates[0])
    source_image = Path(str(source_scene.erp_image_path))
    if not source_image.exists():
        restored = restore_missing_image(source_image, image_index)
        if restored is not None:
            source_image = restored
        else:
            return None, "source_image_not_found"
    rotation = parse_rotation_from_name(str(item.get("scene_id", ""))) or parse_rotation_from_name(expected_path.stem)
    if rotation is None:
        return None, "rotation_suffix_not_parsed"
    yaw_shift_deg, pitch_shift_deg = rotation
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    rebuild_rotated_image(source_image, expected_path, yaw_shift_deg=yaw_shift_deg, pitch_shift_deg=pitch_shift_deg)
    return expected_path, "rebuilt_from_source_scene"


def transformed_bbox_visual(entity_box: Sequence[float], scene: SceneMetadata, *, yaw_shift_deg: float = 0.0, pitch_shift_deg: float = 0.0) -> Optional[Sequence[float]]:
    width = int(scene.erp_width or 0)
    height = int(scene.erp_height or 0)
    if width <= 0 or height <= 0 or len(entity_box) != 4:
        return None

    x1, y1, x2, y2 = [float(v) for v in entity_box]
    xs = np.linspace(x1, x2, num=5)
    ys = np.linspace(y1, y2, num=5)
    sample_points = [(sx, sy) for sx in xs for sy in ys]

    rotated_xs: list[float] = []
    rotated_ys: list[float] = []
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
        return None

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
    y_min = min(rotated_ys)
    y_max = max(rotated_ys)
    if start > end:
        # Preserve seam wrap for visualization by returning x1 > x2.
        return [start, y_min, end, y_max]
    return [start, y_min, end, y_max]


def choose_best_source_scene(
    item: Dict[str, Any],
    target_id: str,
    scene_index: Dict[str, List[Path]],
) -> Optional[SceneMetadata]:
    metadata = item.get("metadata") or {}
    source_scene_id = str((metadata.get("derived_rotation") or {}).get("source_scene_id", "")).strip() or parse_source_scene_id_from_name(str(item.get("scene_id", "")))
    if not source_scene_id:
        return None
    candidates = scene_candidate_paths(scene_index, source_scene_id)
    if not candidates:
        return None

    rotation = parse_rotation_from_name(str(item.get("scene_id", ""))) or parse_rotation_from_name(str(Path(str(item.get("image_path", ""))).stem))
    expected_x = expected_x_center_from_scene_name(str(item.get("scene_id", "")))
    scored: List[Tuple[float, int, SceneMetadata]] = []
    for idx, path in enumerate(candidates):
        scene = load_scene_from_path(path)
        target = find_entity(scene, target_id)
        if target is None or len(target.bbox_erp) != 4:
            continue
        score = 1000.0
        if rotation is not None:
            if abs(rotation[1]) < 1e-6:
                visual_box = shifted_bbox_yaw_only(
                    target.bbox_erp,
                    scene,
                    yaw_shift_deg=rotation[0],
                )
            elif target.resolved_bfov is not None:
                visual_box = bfov_box_from_rotated_entity(
                    scene,
                    target,
                    yaw_shift_deg=rotation[0],
                    pitch_shift_deg=rotation[1],
                )
            else:
                visual_box = transformed_bbox_visual(
                    target.bbox_erp,
                    scene,
                    yaw_shift_deg=rotation[0],
                    pitch_shift_deg=rotation[1],
                )
            if visual_box is not None and expected_x is not None and scene.erp_width:
                center_x = box_center_x_norm(visual_box, float(scene.erp_width))
                score = wrap_distance(center_x, expected_x)
            elif visual_box is not None:
                score = 0.5
        scored.append((score, idx, scene))

    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[0][2]


def resolve_target_box(
    item: Dict[str, Any],
    scene: SceneMetadata,
    scene_path: Path,
    target_id: str,
    scene_index: Dict[str, List[Path]],
) -> Tuple[Optional[Sequence[float]], str]:
    target = find_entity(scene, target_id)
    if target is None:
        return None, "target_entity_not_found"

    item_scene_id = str(item.get("scene_id", "")).strip()
    metadata = item.get("metadata") or {}
    derived_scene_id = str((metadata.get("derived_rotation") or {}).get("derived_scene_id", "")).strip()
    source_scene_id = str((metadata.get("derived_rotation") or {}).get("source_scene_id", "")).strip() or parse_source_scene_id_from_name(item_scene_id)
    effective_derived_id = derived_scene_id or item_scene_id

    rotation = parse_rotation_from_name(item_scene_id) or parse_rotation_from_name(str(Path(str(item.get("image_path", ""))).stem))

    # If this item is derived and we know its source scene, prefer recomputing a
    # wrap-preserving box from the source scene even when derived metadata exists.
    if source_scene_id and rotation is not None:
        source_scene = choose_best_source_scene(item, target_id, scene_index)
        if source_scene is not None:
            source_target = find_entity(source_scene, target_id)
            if source_target is not None and len(source_target.bbox_erp) == 4:
                yaw_shift_deg, pitch_shift_deg = rotation
                if abs(pitch_shift_deg) < 1e-6:
                    visual_box = shifted_bbox_yaw_only(
                        source_target.bbox_erp,
                        source_scene,
                        yaw_shift_deg=yaw_shift_deg,
                    )
                    source_name = "bbox_shifted_from_source_scene_yaw_only"
                elif source_target.resolved_bfov is not None:
                    visual_box = bfov_box_from_rotated_entity(
                        source_scene,
                        source_target,
                        yaw_shift_deg=yaw_shift_deg,
                        pitch_shift_deg=pitch_shift_deg,
                    )
                    source_name = "bbox_recomputed_from_source_scene_bfov"
                else:
                    visual_box = transformed_bbox_visual(
                        source_target.bbox_erp,
                        source_scene,
                        yaw_shift_deg=yaw_shift_deg,
                        pitch_shift_deg=pitch_shift_deg,
                    )
                    source_name = "bbox_recomputed_from_source_scene_rotation_visual"
                if visual_box is not None:
                    return visual_box, source_name

    # If we actually loaded derived-scene metadata, use its bbox directly.
    if scene.scene_id == item_scene_id or (effective_derived_id and scene.scene_id == effective_derived_id):
        if len(target.bbox_erp) == 4:
            return target.bbox_erp, "bbox_erp_current_scene"
        return None, "target_bbox_missing_in_loaded_scene"

    # If this looks like a derived item but we only resolved source-scene metadata,
    # use the source-scene box as a last resort.
    if source_scene_id and scene.scene_id == source_scene_id:
        if len(target.bbox_erp) == 4:
            return target.bbox_erp, "bbox_erp_source_scene_fallback"
        return None, "target_bbox_missing_after_source_scene_fallback"

    if len(target.bbox_erp) == 4:
        return target.bbox_erp, "bbox_erp_generic_fallback"
    return None, "target_bbox_missing_generic"


def clip_box(box: Sequence[float], width: int, height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return (
        max(0.0, min(float(width - 1), x1)),
        max(0.0, min(float(height - 1), y1)),
        max(0.0, min(float(width - 1), x2)),
        max(0.0, min(float(height - 1), y2)),
    )


def draw_wrap_rect(draw: ImageDraw.ImageDraw, box: Tuple[float, float, float, float], img_w: int, color: str, width: int) -> None:
    x1, y1, x2, y2 = box
    if x2 >= x1:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        return
    draw.rectangle([0.0, y1, x2, y2], outline=color, width=width)
    draw.rectangle([x1, y1, float(img_w - 1), y2], outline=color, width=width)


def annotate_target_box(src_image: Path, dst_image: Path, bbox: Sequence[float], color: str, width: int) -> None:
    image = Image.open(src_image).convert("RGB")
    box = clip_box(bbox, image.width, image.height)
    draw = ImageDraw.Draw(image)
    draw_wrap_rect(draw, box, image.width, color, width)
    dst_image.parent.mkdir(parents=True, exist_ok=True)
    if dst_image.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(dst_image, quality=95)
    else:
        image.save(dst_image)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_jsonl = Path(args.output_jsonl)
    output_root = Path(args.output_root) if args.output_root else output_jsonl.parent / "polar_shape_visual_prompt"
    output_root.mkdir(parents=True, exist_ok=True)
    image_dir = output_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    metadata_roots = [Path(path) for path in args.metadata_roots]
    image_search_roots = [Path(path) for path in args.image_search_roots]
    scene_index = discover_scene_metadata_paths(metadata_roots)
    image_index = discover_image_paths(image_search_roots)
    image_search_root_status = summarize_search_roots(image_search_roots)
    output_rows = []
    report: Counter[str] = Counter()
    failures = []

    for row in iter_jsonl(input_path):
        item = copy.deepcopy(row)
        if str(item.get("task_id", "")) != "polar_shape_recovery_mc":
            output_rows.append(item)
            continue

        scene_path = resolve_scene_metadata_path(item, scene_index)
        if scene_path is None:
            report["skipped"] += 1
            failures.append({"item_id": item.get("item_id", ""), "reason": "scene_metadata_not_found"})
            output_rows.append(item)
            continue

        scene = load_scene_from_path(scene_path)
        target_entities = item.get("target_entities") or []
        if not target_entities:
            report["skipped"] += 1
            failures.append({"item_id": item.get("item_id", ""), "reason": "missing_target_entity"})
            output_rows.append(item)
            continue
        target_id = str(target_entities[0])
        target_box, target_box_source = resolve_target_box(item, scene, scene_path, target_id, scene_index)
        if target_box is None:
            report["skipped"] += 1
            failures.append({"item_id": item.get("item_id", ""), "reason": target_box_source})
            output_rows.append(item)
            continue

        src_image = Path(str(item.get("image_path", "")))
        if not src_image.exists():
            restored = restore_missing_image(src_image, image_index)
            if restored is not None:
                src_image = restored
                report["restored_missing_source_image"] += 1
            else:
                source_image_path = str(((item.get("metadata") or {}).get("visual_prompt") or {}).get("source_image_path", "")).strip()
                if source_image_path:
                    candidate = Path(source_image_path)
                    if candidate.exists():
                        src_image = candidate
        if not src_image.exists():
            rebuilt, reason = rebuild_missing_derived_image(src_image, item, scene_index, image_index)
            if rebuilt is not None:
                src_image = rebuilt
                report["rebuilt_missing_derived_image"] += 1
            elif reason:
                report[f"image_resolution_{reason}"] += 1
        if not src_image.exists():
            report["skipped"] += 1
            failures.append(
                {
                    "item_id": item.get("item_id", ""),
                    "reason": "image_not_found",
                    "expected_image_path": str(Path(str(item.get("image_path", "")))),
                    "scene_id": str(item.get("scene_id", "")),
                    "scene_metadata_path": str(scene_path),
                }
            )
            output_rows.append(item)
            continue

        dst_image = image_dir / f"{item.get('item_id', src_image.stem)}{src_image.suffix}"
        annotate_target_box(src_image, dst_image, target_box, str(args.box_color), int(args.box_width))

        item["image_path"] = str(dst_image)
        item["question"] = pick_variant(
            POLAR_VISUAL_TEMPLATES,
            f"polar_shape_recovery_mc:{item.get('item_id', '')}",
        )
        item.setdefault("metadata", {})
        item["metadata"].pop("target_ref", None)
        item["metadata"]["derived_metadata_path"] = str(scene_path)
        item["metadata"]["visual_prompt"] = {
            "type": "red_box",
            "target_entity_id": target_id,
            "bbox_erp": [round(float(value), 2) for value in target_box],
            "bbox_source": target_box_source,
            "source_image_path": str(src_image),
        }
        item["diagnostic_slices"] = sorted(set(item.get("diagnostic_slices") or []) | {"visual_prompt"})
        report["rewritten"] += 1
        output_rows.append(item)

    write_jsonl(output_jsonl, output_rows)
    report_path = output_jsonl.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(
            {
                "input_jsonl": str(input_path),
                "output_jsonl": str(output_jsonl),
                "output_root": str(output_root),
                "scene_index_size": len(scene_index),
                "image_index_size": len(image_index),
                "image_search_roots": [str(path) for path in image_search_roots],
                "image_search_root_status": image_search_root_status,
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
