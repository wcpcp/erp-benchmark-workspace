#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a benchmark pool JSONL and select up to a target quota per task, with optional incremental top-up from an existing selected set."
    )
    parser.add_argument(
        "--pool-jsonl",
        nargs="+",
        required=True,
        help="One or more pool JSONL files. These can be merged candidate/public files collected over time.",
    )
    parser.add_argument(
        "--existing-selected",
        default="",
        help="Optional previously selected JSONL. Items in this file are preserved and count toward each task quota.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where selected.jsonl, remaining_pool.jsonl, and selection_report.json will be written.",
    )
    parser.add_argument(
        "--target-per-task",
        type=int,
        default=250,
        help="Desired number of items per task in the selected subset.",
    )
    parser.add_argument(
        "--allow-manual-review",
        action="store_true",
        help="If set, review-required items may be used while filling deficits. By default they are considered after clean items.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def dedupe_by_item_id(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    best: Dict[str, Dict[str, Any]] = {}
    collisions: List[Dict[str, Any]] = []
    for row in rows:
        item_id = str(row.get("item_id", ""))
        if not item_id:
            continue
        existing = best.get(item_id)
        if existing is None:
            best[item_id] = row
            continue
        prev_score = float(existing.get("quality_score", 0.0))
        new_score = float(row.get("quality_score", 0.0))
        if new_score > prev_score:
            collisions.append({"item_id": item_id, "kept_quality": new_score, "dropped_quality": prev_score})
            best[item_id] = row
        else:
            collisions.append({"item_id": item_id, "kept_quality": prev_score, "dropped_quality": new_score})
    deduped = sorted(best.values(), key=lambda row: (str(row.get("task_id", "")), str(row.get("scene_id", "")), str(row.get("item_id", ""))))
    return deduped, collisions


def row_is_derived(row: Dict[str, Any]) -> bool:
    return "derived_rotation" in (row.get("metadata") or {})


def row_sort_key(row: Dict[str, Any], allow_manual_review: bool) -> Tuple[Any, ...]:
    requires_review = bool(row.get("requires_manual_review"))
    derived = row_is_derived(row)
    return (
        0 if (allow_manual_review or not requires_review) else 1,
        1 if requires_review else 0,
        1 if derived else 0,
        -float(row.get("quality_score", 0.0)),
        str(row.get("scene_id", "")),
        str(row.get("item_id", "")),
    )


def count_answer_keys(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counter = Counter()
    for row in rows:
        answer = str(row.get("answer", "") or "")
        options = row.get("options") or []
        if options and answer:
            counter[answer] += 1
    return {key: int(counter[key]) for key in sorted(counter)}


def task_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    derived = [row for row in rows if row_is_derived(row)]
    natural = [row for row in rows if not row_is_derived(row)]
    difficulty = Counter(str(row.get("difficulty", "unknown")) for row in rows)
    slices = Counter(slice_name for row in rows for slice_name in (row.get("diagnostic_slices") or []))
    return {
        "count": len(rows),
        "natural_count": len(natural),
        "derived_count": len(derived),
        "manual_review_count": sum(1 for row in rows if row.get("requires_manual_review")),
        "difficulty": {key: int(difficulty[key]) for key in sorted(difficulty)},
        "answer_keys": count_answer_keys(rows),
        "diagnostic_slices": {key: int(slices[key]) for key in sorted(slices)},
    }


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pool_rows_raw: List[Dict[str, Any]] = []
    for raw_path in args.pool_jsonl:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Pool JSONL not found: {path}")
        pool_rows_raw.extend(iter_jsonl(path))

    pool_rows, collisions = dedupe_by_item_id(pool_rows_raw)

    existing_rows: List[Dict[str, Any]] = []
    if args.existing_selected:
        existing_path = Path(args.existing_selected)
        if not existing_path.exists():
            raise FileNotFoundError(f"Existing selected JSONL not found: {existing_path}")
        existing_rows, existing_collisions = dedupe_by_item_id(list(iter_jsonl(existing_path)))
        collisions.extend(existing_collisions)

    selected_by_id: Dict[str, Dict[str, Any]] = {str(row["item_id"]): row for row in existing_rows}
    pool_by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    task_ids = sorted({str(row.get("task_id", "")) for row in pool_rows} | {str(row.get("task_id", "")) for row in existing_rows})

    for row in pool_rows:
        item_id = str(row["item_id"])
        if item_id in selected_by_id:
            continue
        pool_by_task[str(row["task_id"])].append(row)

    for task_id in pool_by_task:
        pool_by_task[task_id].sort(key=lambda row: row_sort_key(row, allow_manual_review=bool(args.allow_manual_review)))

    selected_rows: List[Dict[str, Any]] = list(existing_rows)
    selected_count_by_task: Counter[str] = Counter(str(row.get("task_id", "")) for row in selected_rows)
    additions_by_task: Counter[str] = Counter()

    for task_id in task_ids:
        deficit = max(0, int(args.target_per_task) - int(selected_count_by_task.get(task_id, 0)))
        if deficit == 0:
            continue
        candidates = pool_by_task.get(task_id, [])
        to_add = candidates[:deficit]
        selected_rows.extend(to_add)
        additions_by_task[task_id] += len(to_add)
        selected_count_by_task[task_id] += len(to_add)

    selected_ids = {str(row["item_id"]) for row in selected_rows}
    remaining_rows = [row for row in pool_rows if str(row["item_id"]) not in selected_ids]

    selected_rows.sort(key=lambda row: (str(row.get("task_id", "")), str(row.get("scene_id", "")), str(row.get("item_id", ""))))
    remaining_rows.sort(key=lambda row: (str(row.get("task_id", "")), str(row.get("scene_id", "")), str(row.get("item_id", ""))))

    selected_path = output_dir / "selected.jsonl"
    remaining_path = output_dir / "remaining_pool.jsonl"
    report_path = output_dir / "selection_report.json"
    write_jsonl(selected_path, selected_rows)
    write_jsonl(remaining_path, remaining_rows)

    pool_task_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    selected_task_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    remaining_task_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        pool_task_groups[str(row.get("task_id", ""))].append(row)
    for row in selected_rows:
        selected_task_groups[str(row.get("task_id", ""))].append(row)
    for row in remaining_rows:
        remaining_task_groups[str(row.get("task_id", ""))].append(row)

    per_task_report: Dict[str, Any] = {}
    for task_id in task_ids:
        selected_for_task = selected_task_groups.get(task_id, [])
        pool_for_task = pool_task_groups.get(task_id, [])
        remaining_for_task = remaining_task_groups.get(task_id, [])
        existing_count = sum(1 for row in existing_rows if str(row.get("task_id", "")) == task_id)
        final_count = len(selected_for_task)
        per_task_report[task_id] = {
            "target": int(args.target_per_task),
            "pool": task_summary(pool_for_task),
            "selected": task_summary(selected_for_task),
            "remaining_pool": task_summary(remaining_for_task),
            "existing_selected_count": existing_count,
            "added_this_run": int(additions_by_task.get(task_id, 0)),
            "final_selected_count": final_count,
            "deficit_after_selection": max(0, int(args.target_per_task) - final_count),
            "overflow_after_selection": max(0, final_count - int(args.target_per_task)),
        }

    underfilled_tasks = {
        task_id: info["deficit_after_selection"]
        for task_id, info in per_task_report.items()
        if info["deficit_after_selection"] > 0
    }
    report = {
        "target_per_task": int(args.target_per_task),
        "pool_files": [str(Path(p)) for p in args.pool_jsonl],
        "existing_selected": str(args.existing_selected) if args.existing_selected else "",
        "allow_manual_review": bool(args.allow_manual_review),
        "pool_rows_raw": len(pool_rows_raw),
        "pool_rows_unique": len(pool_rows),
        "existing_rows": len(existing_rows),
        "selected_rows": len(selected_rows),
        "remaining_rows": len(remaining_rows),
        "dedupe_collision_count": len(collisions),
        "dedupe_collision_examples": collisions[:20],
        "underfilled_tasks": underfilled_tasks,
        "selected_path": str(selected_path),
        "remaining_pool_path": str(remaining_path),
        "per_task": per_task_report,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "selected_path": str(selected_path),
        "remaining_pool_path": str(remaining_path),
        "report_path": str(report_path),
        "underfilled_tasks": underfilled_tasks,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
