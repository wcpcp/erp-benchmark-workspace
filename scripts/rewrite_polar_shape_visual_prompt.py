#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from erp_spatial_benchmark._vendor.schemas import SceneMetadata
from erp_spatial_benchmark.build_benchmark import find_entity, pick_variant

POLAR_VISUAL_TEMPLATES = [
    "What is the actual shape of the objects drawn in red box in this ERP panorama?",
    "Which shape best matches the real shape of the object drawn in the red box?",
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

    scene_index = discover_scene_metadata_paths([Path(path) for path in args.metadata_roots])
    output_rows = []
    report = {"rewritten": 0, "skipped": 0}
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
        target = find_entity(scene, target_id)
        if target is None or len(target.bbox_erp) != 4:
            report["skipped"] += 1
            failures.append({"item_id": item.get("item_id", ""), "reason": "target_bbox_missing"})
            output_rows.append(item)
            continue

        src_image = Path(str(item.get("image_path", "")))
        if not src_image.exists():
            report["skipped"] += 1
            failures.append({"item_id": item.get("item_id", ""), "reason": "image_not_found"})
            output_rows.append(item)
            continue

        dst_image = image_dir / f"{item.get('item_id', src_image.stem)}{src_image.suffix}"
        annotate_target_box(src_image, dst_image, target.bbox_erp, str(args.box_color), int(args.box_width))

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
            "bbox_erp": [round(float(value), 2) for value in target.bbox_erp],
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
                "counts": report,
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
                "counts": report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
