#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple


INDOOR_KEYWORDS = (
    "indoor",
    "interior",
    "inside",
    "room",
    "corridor",
    "hallway",
    "kitchen",
    "bedroom",
    "bathroom",
    "living room",
    "office",
    "warehouse",
    "mall",
    "shop interior",
    "station hall",
)

OUTDOOR_KEYWORDS = (
    "traffic_sign",
    "road",
    "street",
    "sidewalk",
    "crossing",
    "highway",
    "cycleway",
    "bridge",
    "lane",
    "parking",
    "bus_stop",
    "crosswalk",
)


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON on line {line_number} in {path}") from exc


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "item_id": row.get("item_id"),
        "collection_id": row.get("collection_id"),
        "item_datetime": row.get("item_datetime"),
        "lon": row.get("lon"),
        "lat": row.get("lat"),
        "projection_type": row.get("projection_type"),
        "field_of_view": row.get("field_of_view"),
        "license": row.get("license"),
        "erp_status": row.get("erp_status"),
        "outdoor_status": row.get("outdoor_status"),
        "keep": row.get("keep"),
        "filter_reasons": ";".join(row.get("filter_reasons", [])),
        "hd_url": (row.get("assets") or {}).get("hd"),
        "sd_url": (row.get("assets") or {}).get("sd"),
        "thumb_url": (row.get("assets") or {}).get("thumb"),
    }


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    fieldnames = [
        "item_id",
        "collection_id",
        "item_datetime",
        "lon",
        "lat",
        "projection_type",
        "field_of_view",
        "license",
        "erp_status",
        "outdoor_status",
        "keep",
        "filter_reasons",
        "hd_url",
        "sd_url",
        "thumb_url",
    ]
    materialized = [csv_row(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if materialized:
            writer.writerows(materialized)


def is_equirectangular(row: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    projection_type = str(row.get("projection_type") or "").strip().lower()
    if projection_type == "equirectangular":
        reasons.append("projection_type=equirectangular")
        return True, reasons

    field_of_view = row.get("field_of_view")
    sensor_width = row.get("sensor_width")
    sensor_height = row.get("sensor_height")

    ratio_ok = False
    if sensor_width and sensor_height:
        ratio = float(sensor_width) / float(sensor_height)
        ratio_ok = abs(ratio - 2.0) <= 0.15
        if ratio_ok:
            reasons.append(f"sensor_ratio~2:1({ratio:.3f})")

    if field_of_view is not None and float(field_of_view) >= 359.0 and ratio_ok:
        reasons.append(f"field_of_view={field_of_view}")
        return True, reasons

    return False, reasons


def flatten_text_fields(row: Dict[str, Any]) -> str:
    tokens: List[str] = []
    for key in (
        "collection_title",
        "collection_description",
        "original_file_name",
    ):
        value = row.get(key)
        if value:
            tokens.append(str(value))
    for key in (
        "provider_names",
        "collection_provider_names",
        "collection_keywords",
        "item_semantics",
        "collection_semantics",
        "annotation_semantics",
    ):
        values = row.get(key) or []
        tokens.extend(str(value) for value in values if value)
    return " ".join(tokens).lower()


def outdoor_status(row: Dict[str, Any]) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    text_blob = flatten_text_fields(row)

    indoor_hits = [keyword for keyword in INDOOR_KEYWORDS if keyword in text_blob]
    if indoor_hits:
        reasons.append("indoor_keywords=" + ",".join(indoor_hits[:5]))
        return "rejected", reasons

    score = 0
    if row.get("lon") is not None and row.get("lat") is not None:
        score += 1
        reasons.append("has_gps")
    if row.get("view_azimuth") is not None:
        score += 1
        reasons.append("has_view_azimuth")
    if row.get("horizontal_accuracy") is not None:
        score += 1
        reasons.append("has_horizontal_accuracy")
    collection_length_km = row.get("collection_length_km")
    if collection_length_km is not None and float(collection_length_km) > 0:
        score += 1
        reasons.append("collection_length_km>0")

    outdoor_hits = [keyword for keyword in OUTDOOR_KEYWORDS if keyword in text_blob]
    if outdoor_hits:
        score += 1
        reasons.append("outdoor_keywords=" + ",".join(outdoor_hits[:5]))

    if score >= 2:
        return "accepted", reasons
    if score == 1:
        return "review", reasons
    reasons.append("missing_outdoor_signals")
    return "rejected", reasons


def classify_row(row: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(row)
    reasons: List[str] = []

    erp_ok, erp_reasons = is_equirectangular(row)
    reasons.extend(erp_reasons)
    result["erp_status"] = "accepted" if erp_ok else "rejected"
    if not erp_ok:
        reasons.append("not_equirectangular")
        result["outdoor_status"] = "rejected"
        result["keep"] = False
        result["filter_reasons"] = reasons
        return result

    outdoor, outdoor_reasons = outdoor_status(row)
    reasons.extend(outdoor_reasons)
    result["outdoor_status"] = outdoor
    result["keep"] = outdoor == "accepted"
    result["filter_reasons"] = reasons
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter a Panoramax manifest down to equirectangular outdoor candidates"
    )
    parser.add_argument("--input", required=True, help="Input JSONL manifest from download_panoramax.py crawl")
    parser.add_argument("--accepted-jsonl", required=True, help="Output JSONL for accepted rows")
    parser.add_argument("--review-jsonl", required=True, help="Output JSONL for review rows")
    parser.add_argument("--rejected-jsonl", required=True, help="Output JSONL for rejected rows")
    parser.add_argument("--accepted-csv", help="Optional CSV export for accepted rows")
    args = parser.parse_args()

    accepted: List[Dict[str, Any]] = []
    review: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for row in iter_jsonl(Path(args.input)):
        classified = classify_row(row)
        if classified["keep"]:
            accepted.append(classified)
        elif classified["outdoor_status"] == "review":
            review.append(classified)
        else:
            rejected.append(classified)

    write_jsonl(Path(args.accepted_jsonl), accepted)
    write_jsonl(Path(args.review_jsonl), review)
    write_jsonl(Path(args.rejected_jsonl), rejected)
    if args.accepted_csv:
        write_csv(Path(args.accepted_csv), accepted)

    print(
        json.dumps(
            {
                "accepted": len(accepted),
                "review": len(review),
                "rejected": len(rejected),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
