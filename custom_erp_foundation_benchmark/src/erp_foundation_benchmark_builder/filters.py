from __future__ import annotations

import json
from math import pi
from pathlib import Path
from typing import Any, Iterable, List

from .runtime import DATA_GEN_SRC  # noqa: F401
from erp_data_generation.schemas import Entity


ROOT = Path(__file__).resolve().parents[2]
FILTER_PATH = ROOT / "config" / "filter_rules.json"


def load_filter_rules() -> dict[str, Any]:
    return json.loads(FILTER_PATH.read_text(encoding="utf-8"))


def entity_is_eligible(entity: Entity) -> bool:
    rules = load_filter_rules()["entity"]
    semantic_conf = entity.semantic.confidence if entity.semantic.confidence is not None else 0.7
    return (
        float(entity.confidence) >= rules["min_confidence"]
        and float(semantic_conf) >= rules["min_semantic_confidence"]
        and rules["min_area_ratio"] <= float(entity.area_ratio) <= rules["max_area_ratio"]
    )


def depth_pair_is_eligible(entity_a: Entity, entity_b: Entity) -> bool:
    if entity_a.entity_center_depth is None or entity_b.entity_center_depth is None:
        return False
    rules = load_filter_rules()["pair"]
    return abs(float(entity_a.entity_center_depth) - float(entity_b.entity_center_depth)) >= rules["min_depth_margin"]


def direction_margin_is_eligible(delta_deg: float) -> bool:
    rules = load_filter_rules()["pair"]
    return abs(delta_deg) >= rules["min_direction_margin_deg"]


def yaw_label_from_lon(lon_rad: float) -> str:
    bins = load_filter_rules()["direction_bins"]
    normalized = (lon_rad + pi) / (2 * pi)
    index = int(normalized * len(bins)) % len(bins)
    return bins[index]


def rotate_yaw_label(label: str, rotation_deg: int) -> str:
    bins = load_filter_rules()["direction_bins"]
    if label not in bins:
        return label
    step = int(round(rotation_deg / (360 / len(bins)))) % len(bins)
    idx = bins.index(label)
    return bins[(idx + step) % len(bins)]


def depth_bucket(depth_value: float) -> str:
    rules = load_filter_rules()["distance_buckets"]
    if depth_value <= rules["near_max"]:
        return "near"
    if depth_value <= rules["medium_max"]:
        return "medium"
    return "far"


def relation_label_from_delta(delta_deg: float) -> str:
    delta = ((delta_deg + 180.0) % 360.0) - 180.0
    if -45.0 <= delta <= 45.0:
        return "front"
    if 45.0 < delta <= 135.0:
        return "right"
    if -135.0 <= delta < -45.0:
        return "left"
    return "back"


def search_turn_from_yaw(label: str) -> str:
    mapping = {
        "front": "go_forward",
        "front_right": "turn_right",
        "right": "turn_right",
        "back_right": "turn_back",
        "back": "turn_back",
        "back_left": "turn_back",
        "left": "turn_left",
        "front_left": "turn_left",
    }
    return mapping.get(label, "go_forward")


def strongest_3d_relation(delta_xyz: list[float]) -> str:
    x, y, z = [float(v) for v in delta_xyz]
    axes = {
        "right_of": abs(x),
        "above": abs(y),
        "behind": abs(z),
    }
    dominant = max(axes, key=axes.get)
    if dominant == "right_of":
        return "right_of" if x > 0 else "left_of"
    if dominant == "above":
        return "above" if y > 0 else "below"
    return "behind" if z > 0 else "in_front_of"


def slices_for_entity(entity: Entity) -> List[str]:
    slices = []
    if entity.seam_crossing_flag:
        slices.append("seam")
    if entity.pole_proximity_flag:
        slices.append("pole")
    return slices


def option_key(index: int) -> str:
    return chr(ord("A") + index)
