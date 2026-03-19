from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .runtime import DATA_GEN_SRC  # noqa: F401
from erp_data_generation.entity_selector import choose_relation_partners, score_entity, select_anchor_entities
from erp_data_generation.schemas import Entity, SceneMetadata

from .filters import (
    depth_bucket,
    depth_pair_is_eligible,
    direction_margin_is_eligible,
    entity_is_eligible,
    option_key,
    relation_label_from_delta,
    rotate_yaw_label,
    search_turn_from_yaw,
    slices_for_entity,
    strongest_3d_relation,
    yaw_label_from_lon,
)
from .schemas import BenchmarkChoice, BenchmarkItem
from .templates import render_question


ROOT = Path(__file__).resolve().parents[2]
BLUEPRINT_PATH = ROOT / "config" / "task_blueprint.json"


def load_scene_metadata(input_path: str) -> SceneMetadata:
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    return SceneMetadata.from_dict(data)


def load_blueprint() -> dict[str, Any]:
    return json.loads(BLUEPRINT_PATH.read_text(encoding="utf-8"))


def generate_scene_candidates(scene: SceneMetadata, include_extension: bool = True) -> list[dict[str, Any]]:
    blueprint = load_blueprint()
    task_meta = {task["task_id"]: task for task in blueprint["tasks"]}
    label_counts = Counter(entity.label for entity in scene.entities)
    anchors = [item for item in select_anchor_entities(scene, max_anchors=6) if entity_is_eligible(item["entity"])]

    items: List[BenchmarkItem] = []
    for idx, anchor_item in enumerate(anchors):
        entity = anchor_item["entity"]
        quality = score_entity(entity, label_counts, scene)
        items.extend(_entity_identify(scene, entity, idx, task_meta["entity_identify"], quality))
        items.extend(_attribute_understanding(scene, entity, idx, task_meta["attribute_understanding"], quality))
        items.extend(_referring_grounding(scene, anchors, entity, idx, task_meta["referring_grounding"], quality))
        items.extend(_absolute_direction(scene, entity, idx, task_meta["absolute_direction"], quality))
        items.extend(_rotation_consistency(scene, entity, idx, task_meta["rotation_consistency"], quality))
        items.extend(_distance_bucket(scene, entity, idx, task_meta["distance_bucket"], quality))
        items.extend(_object_search_initial_turn(scene, entity, idx, task_meta["object_search_initial_turn"], quality))
        items.extend(_seam_continuity(scene, entity, idx, task_meta["seam_continuity"], quality))
        items.extend(_polar_distortion(scene, entity, idx, task_meta["polar_distortion_awareness"], quality))

        partners = choose_relation_partners(entity, scene, max_partners=3)
        for p_idx, partner_item in enumerate(partners):
            partner = partner_item["entity"]
            items.extend(_relative_direction(scene, entity, partner, idx, p_idx, task_meta["relative_direction"], quality))
            items.extend(_depth_ordering(scene, entity, partner, idx, p_idx, task_meta["depth_ordering"], quality))
            if include_extension:
                items.extend(_relative_3d_position(scene, entity, partner, idx, p_idx, task_meta["relative_3d_position"], quality))

    items.extend(_existence(scene, anchors, task_meta["existence"]))
    items.extend(_counting(scene, task_meta["counting"]))
    items.extend(_scene_composition(scene, anchors, task_meta["scene_composition"]))
    if include_extension:
        items.extend(_path_search_initial_action(scene, task_meta["path_search_initial_action"]))

    return [item.to_dict() for item in items]


def _make_item(scene: SceneMetadata, task: dict[str, Any], item_id: str, question: str, answer: Any, answer_format: str, difficulty: str, quality: float, target_entities: list[str], options: list[BenchmarkChoice] | None = None, erp_slices: list[str] | None = None, notes: list[str] | None = None) -> BenchmarkItem:
    return BenchmarkItem(
        item_id=item_id,
        scene_id=scene.scene_id,
        task_id=task["task_id"],
        ability_group=task["ability_group"],
        release_phase=task["release_phase"],
        question=question,
        answer=answer,
        answer_format=answer_format,
        difficulty=difficulty,
        image_path=scene.erp_image_path,
        options=options or [],
        target_entities=target_entities,
        erp_slices=erp_slices or [],
        metadata_tier=task["metadata_tier"],
        quality_score=quality,
        gt_sources=list(task["gt_sources"]),
        notes=notes or [],
    )


def _difficulty_from_quality(quality: float) -> str:
    if quality >= 0.82:
        return "easy"
    if quality >= 0.72:
        return "medium"
    return "hard"


def _collect_distractor_labels(scene: SceneMetadata, target: Entity, limit: int = 3) -> list[str]:
    labels = []
    for entity in scene.entities:
        if entity.entity_id == target.entity_id:
            continue
        if entity.label != target.label and entity.label not in labels:
            labels.append(entity.label)
        if len(labels) >= limit:
            break
    fallback = ["chair", "table", "window", "door", "cabinet", "lamp"]
    for label in fallback:
        if label != target.label and label not in labels:
            labels.append(label)
        if len(labels) >= limit:
            break
    return labels[:limit]


def _entity_identify(scene: SceneMetadata, entity: Entity, index: int, task: dict[str, Any], quality: float) -> list[BenchmarkItem]:
    distractors = _collect_distractor_labels(scene, entity)
    options = [entity.label] + distractors
    choices = [BenchmarkChoice(key=option_key(i), text=value) for i, value in enumerate(options)]
    question = render_question(task["task_id"], index, entity_ref=entity.semantic.reground_query or entity.semantic.caption_brief or entity.label)
    return [
        _make_item(
            scene,
            task,
            f"{scene.scene_id}_{task['task_id']}_{entity.entity_id}",
            question,
            choices[0].key,
            task["answer_format"],
            _difficulty_from_quality(quality),
            quality,
            [entity.entity_id],
            choices,
            slices_for_entity(entity),
        )
    ]


def _attribute_understanding(scene: SceneMetadata, entity: Entity, index: int, task: dict[str, Any], quality: float) -> list[BenchmarkItem]:
    attrs = entity.semantic.attributes or {}
    for attribute_name in ("color", "material", "shape", "condition"):
        if attribute_name in attrs and attrs[attribute_name]:
            value = str(attrs[attribute_name])
            pool = _attribute_distractors(attribute_name, value)
            options = [value] + pool[:3]
            choices = [BenchmarkChoice(key=option_key(i), text=opt) for i, opt in enumerate(options)]
            question = render_question(task["task_id"], index, attribute_name=attribute_name, entity_ref=entity.semantic.reground_query or entity.label)
            return [
                _make_item(
                    scene,
                    task,
                    f"{scene.scene_id}_{task['task_id']}_{entity.entity_id}_{attribute_name}",
                    question,
                    choices[0].key,
                    task["answer_format"],
                    _difficulty_from_quality(quality),
                    quality,
                    [entity.entity_id],
                    choices,
                    slices_for_entity(entity),
                )
            ]
    return []


def _referring_grounding(scene: SceneMetadata, anchors: list[dict[str, Any]], entity: Entity, index: int, task: dict[str, Any], quality: float) -> list[BenchmarkItem]:
    candidates = [item["entity"] for item in anchors if item["entity"].entity_id != entity.entity_id][:3]
    if len(candidates) < 3:
        return []
    options = [entity] + candidates
    choices = [
        BenchmarkChoice(
            key=option_key(i),
            text=opt.semantic.caption_brief or opt.semantic.reground_query or opt.label,
            entity_id=opt.entity_id,
            bbox_erp=opt.bbox_erp,
        )
        for i, opt in enumerate(options)
    ]
    question = render_question(task["task_id"], index, entity_ref=entity.semantic.reground_query or entity.label)
    return [
        _make_item(
            scene,
            task,
            f"{scene.scene_id}_{task['task_id']}_{entity.entity_id}",
            question,
            choices[0].key,
            task["answer_format"],
            _difficulty_from_quality(quality),
            quality,
            [entity.entity_id],
            choices,
            slices_for_entity(entity),
        )
    ]


def _existence(scene: SceneMetadata, anchors: list[dict[str, Any]], task: dict[str, Any]) -> list[BenchmarkItem]:
    if not anchors:
        return []
    label_counts = Counter(entity.label for entity in scene.entities)
    label = label_counts.most_common(1)[0][0]
    positive = _make_item(
        scene,
        task,
        f"{scene.scene_id}_{task['task_id']}_{label}_pos",
        render_question(task["task_id"], 0, category_name=label),
        "yes",
        task["answer_format"],
        "easy",
        0.9,
        [],
        [],
    )
    absent = next((candidate for candidate in ["sink", "microwave", "bathtub", "fireplace"] if candidate not in label_counts), None)
    negatives = []
    if absent:
        negatives.append(
            _make_item(
                scene,
                task,
                f"{scene.scene_id}_{task['task_id']}_{absent}_neg",
                render_question(task["task_id"], 1, category_name=absent),
                "no",
                task["answer_format"],
                "medium",
                0.88,
                [],
                [],
            )
        )
    return [positive] + negatives


def _counting(scene: SceneMetadata, task: dict[str, Any]) -> list[BenchmarkItem]:
    counts = Counter(entity.label for entity in scene.entities)
    for label, count in counts.most_common():
        if count <= 5:
            options = ["0", "1", "2", "3", "4", "5+"]
            answer = "5+" if count >= 5 else str(count)
            choices = [BenchmarkChoice(key=option_key(i), text=opt) for i, opt in enumerate(options)]
            answer_key = next(choice.key for choice in choices if choice.text == answer)
            return [
                _make_item(
                    scene,
                    task,
                    f"{scene.scene_id}_{task['task_id']}_{label}",
                    render_question(task["task_id"], 0, category_name=label),
                    answer_key,
                    task["answer_format"],
                    "medium" if count > 1 else "easy",
                    0.86,
                    [],
                    choices,
                )
            ]
    return []


def _scene_composition(scene: SceneMetadata, anchors: list[dict[str, Any]], task: dict[str, Any]) -> list[BenchmarkItem]:
    if len(anchors) < 2:
        return []
    labels = [item["entity"].label for item in anchors[:3]]
    true_statement = f"The scene prominently contains {', '.join(labels[:2])}."
    distractors = [
        f"The scene prominently contains a bathtub and a stove.",
        f"The panorama mainly shows outdoor vehicles and traffic signs.",
        f"The scene contains no clear objects and is visually empty."
    ]
    choices = [BenchmarkChoice(key=option_key(i), text=opt) for i, opt in enumerate([true_statement] + distractors)]
    return [
        _make_item(
            scene,
            task,
            f"{scene.scene_id}_{task['task_id']}",
            render_question(task["task_id"], 0),
            choices[0].key,
            task["answer_format"],
            "medium",
            0.78,
            [item["entity"].entity_id for item in anchors[:2]],
            choices,
        )
    ]


def _absolute_direction(scene: SceneMetadata, entity: Entity, index: int, task: dict[str, Any], quality: float) -> list[BenchmarkItem]:
    label = yaw_label_from_lon(entity.lon_lat[0])
    options = ["front", "right", "back", "left"]
    mapped = _compress_direction(label)
    choices = [BenchmarkChoice(key=option_key(i), text=opt) for i, opt in enumerate(options)]
    answer_key = next(choice.key for choice in choices if choice.text == mapped)
    return [
        _make_item(
            scene,
            task,
            f"{scene.scene_id}_{task['task_id']}_{entity.entity_id}",
            render_question(task["task_id"], index, entity_ref=entity.semantic.reground_query or entity.label),
            answer_key,
            task["answer_format"],
            _difficulty_from_quality(quality),
            quality,
            [entity.entity_id],
            choices,
            slices_for_entity(entity),
        )
    ]


def _relative_direction(scene: SceneMetadata, entity_a: Entity, entity_b: Entity, index: int, partner_index: int, task: dict[str, Any], quality: float) -> list[BenchmarkItem]:
    delta_deg = entity_a.lon_deg - entity_b.lon_deg
    if not direction_margin_is_eligible(delta_deg):
        return []
    label = relation_label_from_delta(delta_deg)
    choices = [BenchmarkChoice(key=option_key(i), text=opt) for i, opt in enumerate(["front", "right", "back", "left"])]
    answer_key = next(choice.key for choice in choices if choice.text == label)
    return [
        _make_item(
            scene,
            task,
            f"{scene.scene_id}_{task['task_id']}_{entity_a.entity_id}_{entity_b.entity_id}_{partner_index}",
            render_question(task["task_id"], index, entity_a=entity_a.semantic.reground_query or entity_a.label, entity_b=entity_b.semantic.reground_query or entity_b.label),
            answer_key,
            task["answer_format"],
            "medium",
            quality,
            [entity_a.entity_id, entity_b.entity_id],
            choices,
            sorted(set(slices_for_entity(entity_a) + slices_for_entity(entity_b))),
        )
    ]


def _seam_continuity(scene: SceneMetadata, entity: Entity, index: int, task: dict[str, Any], quality: float) -> list[BenchmarkItem]:
    if not entity.seam_crossing_flag:
        return []
    choices = [BenchmarkChoice(key="A", text="yes"), BenchmarkChoice(key="B", text="no")]
    return [
        _make_item(
            scene,
            task,
            f"{scene.scene_id}_{task['task_id']}_{entity.entity_id}",
            render_question(task["task_id"], index, entity_ref=entity.semantic.reground_query or entity.label),
            "A",
            task["answer_format"],
            "hard",
            quality,
            [entity.entity_id],
            choices,
            ["seam"],
            ["manual_verification_required"],
        )
    ]


def _rotation_consistency(scene: SceneMetadata, entity: Entity, index: int, task: dict[str, Any], quality: float) -> list[BenchmarkItem]:
    original = _compress_direction(yaw_label_from_lon(entity.lon_lat[0]))
    rotated = _compress_direction(rotate_yaw_label(yaw_label_from_lon(entity.lon_lat[0]), 90))
    choices = [BenchmarkChoice(key=option_key(i), text=opt) for i, opt in enumerate(["front", "right", "back", "left"])]
    answer_key = next(choice.key for choice in choices if choice.text == rotated)
    return [
        _make_item(
            scene,
            task,
            f"{scene.scene_id}_{task['task_id']}_{entity.entity_id}",
            render_question(task["task_id"], index, entity_ref=entity.semantic.reground_query or entity.label, rotation_deg=90),
            answer_key,
            task["answer_format"],
            "hard",
            quality,
            [entity.entity_id],
            choices,
            sorted(set(slices_for_entity(entity) + ["rotation"])),
            [f"original_direction={original}"],
        )
    ]


def _polar_distortion(scene: SceneMetadata, entity: Entity, index: int, task: dict[str, Any], quality: float) -> list[BenchmarkItem]:
    if not entity.pole_proximity_flag:
        return []
    choices = [BenchmarkChoice(key="A", text="yes"), BenchmarkChoice(key="B", text="no")]
    return [
        _make_item(
            scene,
            task,
            f"{scene.scene_id}_{task['task_id']}_{entity.entity_id}",
            render_question(task["task_id"], index, entity_ref=entity.semantic.reground_query or entity.label),
            "A",
            task["answer_format"],
            "hard",
            quality,
            [entity.entity_id],
            choices,
            ["pole"],
            ["manual_verification_required"],
        )
    ]


def _depth_ordering(scene: SceneMetadata, entity_a: Entity, entity_b: Entity, index: int, partner_index: int, task: dict[str, Any], quality: float) -> list[BenchmarkItem]:
    if not depth_pair_is_eligible(entity_a, entity_b):
        return []
    closer = entity_a if float(entity_a.entity_center_depth) < float(entity_b.entity_center_depth) else entity_b
    choices = [
        BenchmarkChoice(key="A", text=entity_a.semantic.reground_query or entity_a.label),
        BenchmarkChoice(key="B", text=entity_b.semantic.reground_query or entity_b.label),
    ]
    answer = "A" if closer.entity_id == entity_a.entity_id else "B"
    return [
        _make_item(
            scene,
            task,
            f"{scene.scene_id}_{task['task_id']}_{entity_a.entity_id}_{entity_b.entity_id}_{partner_index}",
            render_question(task["task_id"], index, entity_a=entity_a.semantic.reground_query or entity_a.label, entity_b=entity_b.semantic.reground_query or entity_b.label),
            answer,
            task["answer_format"],
            "medium",
            quality,
            [entity_a.entity_id, entity_b.entity_id],
            choices,
        )
    ]


def _distance_bucket(scene: SceneMetadata, entity: Entity, index: int, task: dict[str, Any], quality: float) -> list[BenchmarkItem]:
    if entity.entity_center_depth is None:
        return []
    bucket = depth_bucket(float(entity.entity_center_depth))
    choices = [BenchmarkChoice(key=option_key(i), text=opt) for i, opt in enumerate(["near", "medium", "far"])]
    answer_key = next(choice.key for choice in choices if choice.text == bucket)
    return [
        _make_item(
            scene,
            task,
            f"{scene.scene_id}_{task['task_id']}_{entity.entity_id}",
            render_question(task["task_id"], index, entity_ref=entity.semantic.reground_query or entity.label),
            answer_key,
            task["answer_format"],
            "medium",
            quality,
            [entity.entity_id],
            choices,
        )
    ]


def _relative_3d_position(scene: SceneMetadata, entity_a: Entity, entity_b: Entity, index: int, partner_index: int, task: dict[str, Any], quality: float) -> list[BenchmarkItem]:
    if entity_a.entity_xyz_camera is None or entity_b.entity_xyz_camera is None:
        return []
    delta = [a - b for a, b in zip(entity_a.entity_xyz_camera, entity_b.entity_xyz_camera)]
    label = strongest_3d_relation(delta)
    options = ["left_of", "right_of", "above", "below", "in_front_of", "behind"]
    choices = [BenchmarkChoice(key=option_key(i), text=opt) for i, opt in enumerate(options)]
    answer_key = next(choice.key for choice in choices if choice.text == label)
    return [
        _make_item(
            scene,
            task,
            f"{scene.scene_id}_{task['task_id']}_{entity_a.entity_id}_{entity_b.entity_id}_{partner_index}",
            render_question(task["task_id"], index, entity_a=entity_a.semantic.reground_query or entity_a.label, entity_b=entity_b.semantic.reground_query or entity_b.label),
            answer_key,
            task["answer_format"],
            "hard",
            quality,
            [entity_a.entity_id, entity_b.entity_id],
            choices,
            notes=["manual_verification_recommended"],
        )
    ]


def _object_search_initial_turn(scene: SceneMetadata, entity: Entity, index: int, task: dict[str, Any], quality: float) -> list[BenchmarkItem]:
    yaw_label = _compress_direction(yaw_label_from_lon(entity.lon_lat[0]))
    action = search_turn_from_yaw(yaw_label)
    options = ["turn_left", "turn_right", "go_forward", "turn_back"]
    choices = [BenchmarkChoice(key=option_key(i), text=opt) for i, opt in enumerate(options)]
    answer_key = next(choice.key for choice in choices if choice.text == action)
    return [
        _make_item(
            scene,
            task,
            f"{scene.scene_id}_{task['task_id']}_{entity.entity_id}",
            render_question(task["task_id"], index, entity_ref=entity.semantic.reground_query or entity.label),
            answer_key,
            task["answer_format"],
            "medium",
            quality,
            [entity.entity_id],
            choices,
        )
    ]


def _path_search_initial_action(scene: SceneMetadata, task: dict[str, Any]) -> list[BenchmarkItem]:
    if not scene.openings:
        return []
    opening = scene.openings[0]
    answer = opening.get("preferred_action", "go_forward")
    options = ["turn_left", "turn_right", "go_forward", "turn_back"]
    if answer not in options:
        return []
    choices = [BenchmarkChoice(key=option_key(i), text=opt) for i, opt in enumerate(options)]
    answer_key = next(choice.key for choice in choices if choice.text == answer)
    return [
        _make_item(
            scene,
            task,
            f"{scene.scene_id}_{task['task_id']}",
            render_question(task["task_id"], 0),
            answer_key,
            task["answer_format"],
            "hard",
            0.72,
            [],
            choices,
            notes=["manual_verification_required"],
        )
    ]


def _attribute_distractors(attribute_name: str, value: str) -> list[str]:
    pools = {
        "color": ["white", "black", "brown", "gray", "blue", "red", "green"],
        "material": ["wood", "metal", "plastic", "glass", "fabric", "stone"],
        "shape": ["rectangular", "round", "square", "elongated", "flat"],
        "condition": ["open", "closed", "blurred", "clean", "worn"],
    }
    pool = [item for item in pools.get(attribute_name, []) if item != value]
    return pool


def _compress_direction(label: str) -> str:
    if "front" in label and "left" in label:
        return "left"
    if "front" in label and "right" in label:
        return "right"
    if "back" in label and "left" in label:
        return "back"
    if "back" in label and "right" in label:
        return "back"
    return label
