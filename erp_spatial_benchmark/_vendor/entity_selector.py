from __future__ import annotations

from collections import Counter
from math import pi
from typing import Dict, Iterable, List

from .schemas import Entity, SceneMetadata


POLE_ABS_LAT_DEG_MIN = 55.0
SEAM_MARGIN_RATIO = 0.03


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def infer_seam_adjacency(entity: Entity, scene: SceneMetadata) -> bool:
    if entity.seam_crossing_flag is not None:
        return bool(entity.seam_crossing_flag)
    if len(entity.bbox_erp) == 4 and scene.erp_width:
        x1, _, x2, _ = entity.bbox_erp
        margin = max(12.0, scene.erp_width * SEAM_MARGIN_RATIO)
        return x1 <= margin or x2 >= (scene.erp_width - margin)
    return abs(entity.lon_deg) >= 165.0


def infer_pole_proximity(entity: Entity) -> bool:
    if entity.pole_proximity_flag is not None:
        return bool(entity.pole_proximity_flag)
    return abs(entity.lat_deg) >= POLE_ABS_LAT_DEG_MIN


def angular_gap_deg(entity_a: Entity, entity_b: Entity) -> float:
    raw_lon_gap = abs(entity_a.lon_deg - entity_b.lon_deg) % 360.0
    lon_gap = min(raw_lon_gap, 360.0 - raw_lon_gap)
    lat_gap = abs(entity_a.lat_deg - entity_b.lat_deg)
    return lon_gap + lat_gap


def _area_score(area_ratio: float) -> float:
    if area_ratio <= 0:
        return 0.0
    if area_ratio < 0.001:
        return 0.15
    if area_ratio < 0.005:
        return 0.5
    if area_ratio < 0.05:
        return 1.0
    if area_ratio < 0.2:
        return 0.8
    return 0.4


def _uniqueness_score(entity: Entity, label_counts: Counter) -> float:
    if entity.entity_uniqueness_score is not None:
        return float(entity.entity_uniqueness_score)
    count = max(label_counts[entity.label], 1)
    return 1.0 / count


def _geometry_usefulness(entity: Entity, scene: SceneMetadata) -> float:
    score = 0.45
    if infer_seam_adjacency(entity, scene):
        score += 0.2
    if infer_pole_proximity(entity):
        score += 0.15
    if abs(entity.lon_deg) > 120:
        score += 0.1
    if abs(entity.lat_deg) > 35:
        score += 0.1
    return _clamp(score, 0.0, 1.0)


def _semantic_verification_score(entity: Entity) -> float:
    return 1.0 if entity.verified_semantics else 0.35


def _depth_quality_score(entity: Entity) -> float:
    if entity.depth_quality_score is not None:
        return _clamp(float(entity.depth_quality_score), 0.0, 1.0)
    return 1.0 if entity.has_depth else 0.4


def _reground_score(entity: Entity) -> float:
    if entity.local_reground_pred_score is None:
        return 0.0
    return _clamp(float(entity.local_reground_pred_score), 0.0, 1.0)


def score_entity(entity: Entity, label_counts: Counter, scene: SceneMetadata) -> float:
    confidence = float(entity.best_score or entity.confidence or 0.0)
    reground = _reground_score(entity)
    support_views = _clamp(entity.support_views / 3.0, 0.0, 1.0)
    area_score = _area_score(entity.area_ratio)
    uniqueness = _uniqueness_score(entity, label_counts)
    geometry = _geometry_usefulness(entity, scene)
    semantic_verification = _semantic_verification_score(entity)
    depth_quality = _depth_quality_score(entity)
    score = (
        0.24 * confidence
        + 0.20 * reground
        + 0.10 * support_views
        + 0.14 * area_score
        + 0.12 * uniqueness
        + 0.10 * geometry
        + 0.10 * semantic_verification
        + 0.00 * depth_quality
    )
    return round(score, 4)


def _yaw_bin(entity: Entity, bins: int = 4) -> int:
    lon = entity.lon_lat[0]
    normalized = (lon + pi) / (2 * pi)
    return min(bins - 1, max(0, int(normalized * bins)))


def _pitch_bin(entity: Entity, bins: int = 3) -> int:
    lat = entity.lon_lat[1]
    normalized = (lat + (pi / 2.0)) / pi
    return min(bins - 1, max(0, int(normalized * bins)))


def select_anchor_entities(scene: SceneMetadata, max_anchors: int = 6) -> List[Dict[str, object]]:
    label_counts = Counter(entity.label for entity in scene.entities)
    scored = []
    for entity in scene.entities:
        selection_score = score_entity(entity, label_counts, scene)
        scored.append(
            {
                "entity": entity,
                "selection_score": selection_score,
                "yaw_bin": _yaw_bin(entity),
                "pitch_bin": _pitch_bin(entity),
                "seam_adjacent": infer_seam_adjacency(entity, scene),
                "pole_adjacent": infer_pole_proximity(entity),
                "depth_bucket": entity.depth_bucket,
            }
        )
    scored.sort(key=lambda item: item["selection_score"], reverse=True)

    selected: List[Dict[str, object]] = []
    used_bins = set()

    for item in scored:
        bin_key = (item["yaw_bin"], item["pitch_bin"])
        if bin_key not in used_bins:
            selected.append(item)
            used_bins.add(bin_key)
        if len(selected) >= max_anchors:
            return selected

    for item in scored:
        if any(sel["entity"].entity_id == item["entity"].entity_id for sel in selected):
            continue
        selected.append(item)
        if len(selected) >= max_anchors:
            break

    return selected


def choose_relation_partners(anchor: Entity, scene: SceneMetadata, max_partners: int = 3) -> List[Dict[str, object]]:
    candidate_payloads: List[Dict[str, object]] = []
    for entity in scene.entities:
        if entity.entity_id == anchor.entity_id:
            continue
        gap = angular_gap_deg(anchor, entity)
        depth_gap = 0.0
        if anchor.entity_center_depth is not None and entity.entity_center_depth is not None:
            depth_gap = abs(anchor.entity_center_depth - entity.entity_center_depth)
        candidate_payloads.append(
            {
                "entity": entity,
                "angular_gap_deg": round(gap, 2),
                "depth_gap": round(depth_gap, 3),
                "same_label": entity.label == anchor.label,
                "verified_semantics": entity.verified_semantics,
            }
        )

    selected: List[Dict[str, object]] = []
    used_ids = set()

    def _take_best(candidates: List[Dict[str, object]], role: str) -> None:
        for item in candidates:
            entity = item["entity"]
            if entity.entity_id in used_ids:
                continue
            enriched = dict(item)
            enriched["role"] = role
            selected.append(enriched)
            used_ids.add(entity.entity_id)
            return

    nearest = sorted(
        candidate_payloads,
        key=lambda item: (
            item["angular_gap_deg"],
            -int(item["verified_semantics"]),
            item["depth_gap"],
        ),
    )
    _take_best(nearest, "nearest_neighbor")

    same_category = sorted(
        [item for item in candidate_payloads if item["same_label"]],
        key=lambda item: (
            item["angular_gap_deg"],
            -int(item["verified_semantics"]),
        ),
    )
    _take_best(same_category, "same_category_distractor")

    far_contrast = sorted(
        [item for item in candidate_payloads if not item["same_label"]],
        key=lambda item: (
            -(item["angular_gap_deg"] + item["depth_gap"] * 10.0),
            -int(item["verified_semantics"]),
        ),
    )
    _take_best(far_contrast, "far_contrast")

    fallback = sorted(
        candidate_payloads,
        key=lambda item: (
            -int(item["verified_semantics"]),
            item["angular_gap_deg"],
            -item["depth_gap"],
        ),
    )
    for item in fallback:
        if len(selected) >= max_partners:
            break
        if item["entity"].entity_id in used_ids:
            continue
        enriched = dict(item)
        enriched["role"] = "fallback"
        selected.append(enriched)
        used_ids.add(item["entity"].entity_id)

    return selected[:max_partners]


def summarize_label_distribution(entities: Iterable[Entity]) -> Dict[str, int]:
    return dict(Counter(entity.label for entity in entities))
