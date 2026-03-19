from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BLUEPRINT_PATH = ROOT / "config" / "task_blueprint.json"
PROTOCOL_PATH = ROOT / "config" / "benchmark_protocol.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assemble_benchmark_pool(candidates: list[dict[str, Any]], phase: str = "core_v1", target_per_task: int | None = None) -> list[dict[str, Any]]:
    blueprint = load_json(BLUEPRINT_PATH)
    protocol = load_json(PROTOCOL_PATH)
    task_meta = {task["task_id"]: task for task in blueprint["tasks"] if phase == "all" or task["release_phase"] == phase}
    max_per_scene_per_task = int(protocol["balancing_rules"]["max_items_from_single_scene_per_task"])

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        task_id = item.get("task_id")
        if task_id in task_meta:
            grouped[item["task_id"]].append(item)

    selected: list[dict[str, Any]] = []
    for task_id, items in grouped.items():
        items.sort(key=lambda row: (-float(row.get("quality_score", 0.0)), row["scene_id"], row["id"]))
        wanted = target_per_task or int(task_meta[task_id]["target_samples"])
        per_scene_counts: dict[str, int] = defaultdict(int)
        count = 0
        for item in items:
            scene_id = item["scene_id"]
            if per_scene_counts[scene_id] >= max_per_scene_per_task:
                continue
            selected.append(item)
            per_scene_counts[scene_id] += 1
            count += 1
            if count >= wanted:
                break
    selected.sort(key=lambda row: (row["task_id"], row["scene_id"], row["id"]))
    return selected
