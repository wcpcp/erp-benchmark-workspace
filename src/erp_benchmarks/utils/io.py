from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


ID_KEYS = ("id", "question_id", "sample_id", "uid", "episode_id")
ANSWER_KEYS = ("answer", "label", "target", "ground_truth", "gt_answer")
PRED_KEYS = ("prediction", "pred", "answer", "response", "output")


def load_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "samples", "records", "items"):
                if isinstance(payload.get(key), list):
                    return payload[key]
            return [payload]

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    raise ValueError(f"Unsupported file format: {path}")


def dump_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def first_existing(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def infer_record_id(record: dict[str, Any], fallback_index: int) -> str:
    value = first_existing(record, ID_KEYS)
    return str(value if value is not None else fallback_index)


def infer_answer(record: dict[str, Any]) -> Any:
    return first_existing(record, ANSWER_KEYS)


def infer_prediction(record: dict[str, Any]) -> Any:
    return first_existing(record, PRED_KEYS)
