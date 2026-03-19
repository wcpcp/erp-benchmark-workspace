from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


ACTION_RE = re.compile(r"(rotate|submit)\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", re.IGNORECASE)


@dataclass
class ParsedAction:
    name: str
    yaw: int
    pitch: int


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


def canonical_submit(target_yaw: list[float], target_pitch: list[float]) -> str:
    yaw = int(round(yaw_interval_center(target_yaw)))
    pitch = int(round((float(target_pitch[0]) + float(target_pitch[1])) / 2.0))
    return f"submit({yaw},{pitch})"


def parse_action(text: Any) -> ParsedAction | None:
    if text is None:
        return None
    match = ACTION_RE.search(str(text))
    if not match:
        return None
    return ParsedAction(
        name=match.group(1).lower(),
        yaw=int(match.group(2)),
        pitch=int(match.group(3)),
    )


def rotate_relative_yaw(angle: float, start_yaw: float) -> float:
    return normalize_yaw(float(angle) - float(start_yaw))


def rotate_yaw_range(target_range: list[float], start_yaw: float) -> list[float]:
    return [
        rotate_relative_yaw(float(target_range[0]), start_yaw),
        rotate_relative_yaw(float(target_range[1]), start_yaw),
    ]


def build_rotated_erp_image(source_path: Path, output_path: Path, start_yaw: float) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path

    with Image.open(source_path) as image:
        image = image.convert("RGB")
        width, _ = image.size
        shift_px = int(round((float(start_yaw) % 360.0) / 360.0 * width)) % width
        if shift_px == 0:
            image.save(output_path)
            return output_path

        left = image.crop((shift_px, 0, width, image.height))
        right = image.crop((0, 0, shift_px, image.height))
        rotated = Image.new(image.mode, image.size)
        rotated.paste(left, (0, 0))
        rotated.paste(right, (width - shift_px, 0))
        rotated.save(output_path)
    return output_path


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


def iter_episode_sources(extract_root: Path) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for annotation_path in sorted(extract_root.rglob("annotation.json")):
        scene_dir = annotation_path.parent
        image_candidates = sorted(scene_dir.glob("*.png")) + sorted(scene_dir.glob("*.jpg")) + sorted(scene_dir.glob("*.jpeg"))
        if not image_candidates:
            continue
        image_path = image_candidates[0]
        task_family = "hps" if "hps" in str(scene_dir).lower() else "hos"
        scene_name = scene_dir.name
        items = json.loads(annotation_path.read_text(encoding="utf-8"))
        for item_index, item in enumerate(items):
            initial_yaws = item.get("initial yaw")
            if isinstance(initial_yaws, list) and initial_yaws:
                levels = item.get("level", [0] * len(initial_yaws))
                zipped = zip(initial_yaws, levels)
            else:
                zipped = zip([0, 90, 180, 270], [0, 0, 0, 0])

            for seed_index, (start_yaw, level) in enumerate(zipped):
                target_yaw = [float(item["yaw"][0]), float(item["yaw"][1])]
                target_pitch = (
                    [float(item["pitch"][0]), float(item["pitch"][1])]
                    if "pitch" in item
                    else [-90.0, 90.0]
                )
                record_id = f"{task_family}::{scene_name}::{item_index}::{seed_index}"
                episodes.append(
                    {
                        "id": record_id,
                        "task_family": task_family,
                        "scene_name": scene_name,
                        "image_path": str(image_path),
                        "annotation_path": str(annotation_path),
                        "instruction": item["task"],
                        "start_yaw": float(start_yaw),
                        "start_pitch": 0.0,
                        "target_yaw": target_yaw,
                        "target_pitch": target_pitch,
                        "target_center_yaw": yaw_interval_center(target_yaw),
                        "target_center_pitch": (target_pitch[0] + target_pitch[1]) / 2.0,
                        "level": int(level),
                    }
                )
    return episodes


def build_hstar_protocol_records(extract_root: Path) -> dict[str, list[dict[str, Any]]]:
    episodes = iter_episode_sources(extract_root)
    perspective_records: list[dict[str, Any]] = []
    erp_submit_records: list[dict[str, Any]] = []
    rotated_root = extract_root.parent / "rotated_erp"

    for episode in episodes:
        perspective_records.append(
            {
                **episode,
                "protocol": "perspective_multiturn",
                "question": (
                    "Use perspective observations rendered from the ERP panorama and "
                    "solve the task with multi-turn actions."
                ),
                "prompt": episode["instruction"],
                "allowed_actions": ["rotate(yaw,pitch)", "submit(yaw,pitch)"],
                "render_config": {
                    "projection": "perspective",
                    "fov": 100,
                    "initial_yaw": episode["start_yaw"],
                    "initial_pitch": episode["start_pitch"],
                    "resolution": [1920, 1080],
                },
                "success_rule": "submit inside target yaw/pitch range",
                "preferred_submit": canonical_submit(episode["target_yaw"], episode["target_pitch"]),
            }
        )
        source_path = Path(str(episode["image_path"]))
        rotated_name = (
            f"{episode['task_family']}__{episode['scene_name']}__yaw"
            f"{int(round(float(episode['start_yaw']))):03d}{source_path.suffix.lower()}"
        )
        rotated_image_path = build_rotated_erp_image(
            source_path,
            rotated_root / rotated_name,
            float(episode["start_yaw"]),
        )
        rotated_target_yaw = rotate_yaw_range(episode["target_yaw"], episode["start_yaw"])

        erp_submit_records.append(
            {
                **episode,
                "protocol": "erp_rotated",
                "task_variant": "rotated_submit",
                "original_image_path": episode["image_path"],
                "image_path": str(rotated_image_path),
                "rotation_delta_yaw": float(episode["start_yaw"]),
                "answer_coordinate_system": "rotated_erp_image_coordinates",
                "target_yaw_original": list(episode["target_yaw"]),
                "target_yaw": rotated_target_yaw,
                "question": (
                    f"You are given one full ERP panorama whose forward direction has been re-centered to the current initial yaw. "
                    f"Human instruction: {episode['instruction']} Output the final target direction in this ERP image's coordinate "
                    "system as submit(yaw,pitch)."
                ),
                "prompt": (
                    f"Rotated ERP panorama input. Task: {episode['instruction']}. "
                    "Return the final target direction in the current ERP image coordinates as submit(yaw,pitch)."
                ),
                "answer": canonical_submit(rotated_target_yaw, episode["target_pitch"]),
                "success_rule": "submitted yaw/pitch falls inside the rotated target window",
            }
        )

    return {
        "perspective_multiturn": perspective_records,
        "erp_rotated_submit": erp_submit_records,
    }
