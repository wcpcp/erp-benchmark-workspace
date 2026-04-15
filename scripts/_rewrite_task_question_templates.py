#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


RenderFn = Callable[[Dict[str, Any], str], Optional[str]]


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input-jsonl", required=True, help="Input benchmark JSONL to rewrite.")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--max-failure-examples",
        type=int,
        default=50,
        help="Maximum number of skipped example records to keep in the report.",
    )
    return parser


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


def stable_template(templates: Sequence[str], key: str) -> str:
    if not templates:
        raise ValueError("templates must not be empty")
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return list(templates)[int(digest[:8], 16) % len(templates)]


def report_path_for(output_jsonl: Path) -> Path:
    if output_jsonl.suffix:
        return output_jsonl.with_suffix(output_jsonl.suffix + ".report.json")
    return output_jsonl.parent / f"{output_jsonl.name}.report.json"


def rewrite_task_questions(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    task_id: str,
    templates: Sequence[str],
    render_question: RenderFn,
    max_failure_examples: int = 50,
) -> Dict[str, Any]:
    output_rows: List[Dict[str, Any]] = []
    failure_examples: List[Dict[str, Any]] = []
    counts = {
        "rewritten": 0,
        "skipped": 0,
        "non_target_rows": 0,
        "task_rows_seen": 0,
    }

    for row in iter_jsonl(input_jsonl):
        if str(row.get("task_id", "")) != task_id:
            counts["non_target_rows"] += 1
            output_rows.append(row)
            continue

        counts["task_rows_seen"] += 1
        item_id = str(row.get("item_id", "")).strip()
        template = stable_template(templates, item_id or str(counts["task_rows_seen"]))
        question = render_question(row, template)
        if not question:
            counts["skipped"] += 1
            output_rows.append(row)
            if len(failure_examples) < max_failure_examples:
                failure_examples.append(
                    {
                        "item_id": item_id,
                        "reason": "missing_required_metadata",
                        "metadata_keys": sorted(list((row.get("metadata") or {}).keys())),
                    }
                )
            continue

        rewritten = copy.deepcopy(row)
        rewritten["question"] = question
        output_rows.append(rewritten)
        counts["rewritten"] += 1

    write_jsonl(output_jsonl, output_rows)
    report = {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "task_id": task_id,
        "templates": list(templates),
        "counts": counts,
        "failure_examples": failure_examples,
    }
    report_path_for(output_jsonl).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def metadata_str(row: Dict[str, Any], key: str) -> str:
    metadata = row.get("metadata") or {}
    value = metadata.get(key)
    if value is None:
        return ""
    return str(value).strip()
