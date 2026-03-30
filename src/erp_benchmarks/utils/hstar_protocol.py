from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ACTION_RE = re.compile(r"(rotate|submit)\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", re.IGNORECASE)
ANGLE_RE = re.compile(
    r"(?:yaw\s*[:=]\s*)?(-?\d+(?:\.\d+)?)\s*[,，]\s*(?:pitch\s*[:=]\s*)?(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
PAREN_ANGLE_RE = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*\)")


@dataclass
class ParsedAction:
    name: str
    yaw: float
    pitch: float


def normalize_yaw(angle: float) -> float:
    return angle % 360


def wrap_signed_delta(source_yaw: float, target_yaw: float) -> float:
    source = normalize_yaw(source_yaw)
    target = normalize_yaw(target_yaw)
    delta = (target - source + 540) % 360 - 180
    return delta


def yaw_in_range(yaw: float, target_range: list[float]) -> bool:
    yaw = normalize_yaw(yaw)
    start = normalize_yaw(target_range[0])
    end = normalize_yaw(target_range[1])
    if start <= end:
        return start <= yaw <= end
    return yaw >= start or yaw <= end


def pitch_in_range(pitch: float, target_range: list[float]) -> bool:
    return float(target_range[0]) <= float(pitch) <= float(target_range[1])


def yaw_interval_span(target_range: list[float]) -> float:
    start = normalize_yaw(target_range[0])
    end = normalize_yaw(target_range[1])
    return (end - start) % 360


def yaw_interval_center(target_range: list[float]) -> float:
    start = normalize_yaw(target_range[0])
    span = yaw_interval_span(target_range)
    return normalize_yaw(start + span / 2.0)


def yaw_distance_to_range(yaw: float, target_range: list[float]) -> float:
    if yaw_in_range(yaw, target_range):
        return 0.0
    start = normalize_yaw(target_range[0])
    end = normalize_yaw(target_range[1])
    yaw = normalize_yaw(yaw)
    return min(abs(wrap_signed_delta(yaw, start)), abs(wrap_signed_delta(yaw, end)))


def pitch_distance_to_range(pitch: float, target_range: list[float]) -> float:
    if pitch_in_range(pitch, target_range):
        return 0.0
    low, high = float(target_range[0]), float(target_range[1])
    return min(abs(pitch - low), abs(pitch - high))


def canonical_direction(target_yaw: list[float], target_pitch: list[float]) -> str:
    yaw = int(normalize_yaw(round(yaw_interval_center(target_yaw))))
    pitch = int(round((float(target_pitch[0]) + float(target_pitch[1])) / 2.0))
    return f"({yaw},{pitch})"


def parse_action(text: Any) -> ParsedAction | None:
    if text is None:
        return None
    raw = str(text)
    match = ACTION_RE.search(raw)
    if match:
        return ParsedAction(
            name=match.group(1).lower(),
            yaw=float(match.group(2)),
            pitch=float(match.group(3)),
        )
    match = PAREN_ANGLE_RE.search(raw) or ANGLE_RE.search(raw)
    if match:
        return ParsedAction(
            name="angle",
            yaw=float(match.group(1)),
            pitch=float(match.group(2)),
        )
    return None


def extract_hstar_archives(raw_dir: Path, extract_root: Path) -> dict[str, Path]:
    extract_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for archive_name in ("hos_bench.zip", "hps_bench.zip"):
        archive_path = raw_dir / archive_name
        if not archive_path.exists():
            continue
        task_name = archive_name.replace(".zip", "")
        target_dir = extract_root / task_name
        marker = target_dir / ".extracted"
        if not marker.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(target_dir)
            marker.write_text("ok\n", encoding="utf-8")
        outputs[task_name] = target_dir
    return outputs


def _select_scene_image(scene_dir: Path) -> Path | None:
    image_candidates = (
        sorted(scene_dir.glob("*.png"))
        + sorted(scene_dir.glob("*.jpg"))
        + sorted(scene_dir.glob("*.jpeg"))
    )
    return image_candidates[0] if image_candidates else None


def _normalize_pitch_range(item: dict[str, Any]) -> list[float]:
    pitch = item.get("pitch")
    if isinstance(pitch, list) and len(pitch) >= 2:
        return [float(pitch[0]), float(pitch[1])]
    return [-90.0, 90.0]


def _normalize_level(value: Any) -> int:
    if isinstance(value, list):
        if not value:
            return 0
        value = value[0]
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _build_direct_question(task: str) -> str:
    task = str(task).strip()
    if task and task[-1] not in ".!?":
        task = f"{task}."
    return f"{task} Return only the target direction angles in this ERP panorama as (yaw,pitch)."


def _scene_key(scene_dir: Path, extract_root: Path) -> str:
    relative = scene_dir.relative_to(extract_root)
    return "__".join(relative.parts)


def _official_hstar_roots(extract_root: Path) -> list[Path]:
    roots = []
    for name in ("hos_bench", "hps_bench"):
        candidate = extract_root / name
        if candidate.exists() and candidate.is_dir():
            roots.append(candidate)
    return roots or [extract_root]


def iter_official_hstar_entries(extract_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for benchmark_root in _official_hstar_roots(extract_root):
        for annotation_path in sorted(benchmark_root.rglob("annotation.json")):
            scene_dir = annotation_path.parent
            image_path = _select_scene_image(scene_dir)
            if image_path is None:
                continue

            task_family = "hps" if benchmark_root.name == "hps_bench" else "hos"
            scene_name = scene_dir.name
            scene_path = str(scene_dir.relative_to(extract_root))
            scene_key = _scene_key(scene_dir, extract_root)
            items = json.loads(annotation_path.read_text(encoding="utf-8"))
            if not isinstance(items, list):
                continue

            for item_index, item in enumerate(items):
                if not isinstance(item, dict) or "task" not in item or "yaw" not in item:
                    continue
                target_yaw = [float(item["yaw"][0]), float(item["yaw"][1])]
                target_pitch = _normalize_pitch_range(item)
                question = _build_direct_question(str(item["task"]))
                record_id = f"{task_family}::{scene_key}::{item_index}"
                entries.append(
                    {
                        "id": record_id,
                        "protocol": "erp_direct",
                        "task_variant": "direct_submit",
                        "task_family": task_family,
                        "scene_name": scene_name,
                        "scene_path": scene_path,
                        "image_path": str(image_path),
                        "annotation_path": str(annotation_path),
                        "instruction": str(item["task"]).strip(),
                        "target_yaw": target_yaw,
                        "target_pitch": target_pitch,
                        "target_center_yaw": yaw_interval_center(target_yaw),
                        "target_center_pitch": (target_pitch[0] + target_pitch[1]) / 2.0,
                        "level": _normalize_level(item.get("level", 0)),
                        "question": question,
                        "prompt": question,
                        "answer": canonical_direction(target_yaw, target_pitch),
                        "success_rule": "submitted yaw/pitch falls inside the official target window",
                    }
                )
    return entries


def build_hstar_protocol_records(extract_root: Path) -> dict[str, list[dict[str, Any]]]:
    return {"erp_direct_submit": iter_official_hstar_entries(extract_root)}
