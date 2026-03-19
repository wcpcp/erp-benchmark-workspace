#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TEMPLATES = ROOT / "templates"
EXAMPLES = ROOT / "examples"


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_prediction_template(rows: list[dict]) -> list[dict]:
    template = []
    for row in rows:
        if "answer" in row:
            prediction = row["answer"]
        else:
            prediction = ""
        template.append({"id": row["id"], "prediction": prediction})
    return template


def build_omnispatial_smoke() -> dict:
    raw_root = DATA / "omnispatial" / "raw"
    manifest_path = DATA / "omnispatial" / "manifests" / "smoke_20.jsonl"
    template_path = TEMPLATES / "predictions_omnispatial_smoke_template.jsonl"

    if manifest_path.exists():
        rows = load_jsonl(manifest_path)
        save_jsonl(template_path, make_prediction_template(rows))
        return {
            "status": "ok",
            "num_rows": len(rows),
            "smoke_manifest": str(manifest_path),
            "template": str(template_path),
            "source": "existing_smoke_manifest",
        }

    full_manifest = DATA / "omnispatial" / "manifests" / "test.jsonl"
    if full_manifest.exists():
        rows = load_jsonl(full_manifest)[:20]
        save_jsonl(manifest_path, rows)
        save_jsonl(template_path, make_prediction_template(rows))
        return {
            "status": "ok",
            "num_rows": len(rows),
            "smoke_manifest": str(manifest_path),
            "template": str(template_path),
            "source": "local_test_manifest",
        }

    parquet_path = raw_root / "test-00000-of-00001.parquet"
    if parquet_path.exists():
        from datasets import load_dataset

        rows = []
        dataset = load_dataset("parquet", data_files=str(parquet_path), split="train[:20]")
        for row_index, row in enumerate(dataset):
            rel_image_path = str(row["image_path"])
            local_image = raw_root / rel_image_path
            options = list(row["options"])
            answer_index = int(row["answer"])
            answer_letter = str(row.get("gt", "")).strip().upper()
            answer_text = options[answer_index] if 0 <= answer_index < len(options) else answer_letter
            raw_id = str(row["id"])
            rows.append(
                {
                    "id": f"{row_index}::{raw_id}",
                    "raw_id": raw_id,
                    "row_index": row_index,
                    "image_path": str(local_image),
                    "question": row["question"],
                    "prompt": row["question"],
                    "options": options,
                    "answer": answer_text,
                    "answer_index": answer_index,
                    "answer_letter": answer_letter,
                    "task_type": row.get("task_type"),
                    "sub_task_type": row.get("sub_task_type"),
                }
            )

        save_jsonl(manifest_path, rows)
        save_jsonl(template_path, make_prediction_template(rows))
        return {
            "status": "ok",
            "num_rows": len(rows),
            "smoke_manifest": str(manifest_path),
            "template": str(template_path),
            "source": "local_parquet",
        }

    return {
        "status": "skipped",
        "reason": (
            "OmniSpatial local data not found. Expected one of: "
            "data/omnispatial/manifests/smoke_20.jsonl, "
            "data/omnispatial/manifests/test.jsonl, or "
            "data/omnispatial/raw/test-00000-of-00001.parquet."
        ),
    }


def build_subset(src: Path, dst: Path, limit: int = 20) -> int:
    rows = load_jsonl(src)[:limit]
    save_jsonl(dst, rows)
    return len(rows)


def main() -> int:
    report: dict[str, dict] = {}

    tasks = [
        (
            "osr-bench",
            DATA / "osr-bench" / "manifests" / "test.jsonl",
            DATA / "osr-bench" / "manifests" / "smoke_20.jsonl",
            TEMPLATES / "predictions_osr_smoke_template.jsonl",
            20,
        ),
        (
            "panoenv",
            DATA / "panoenv" / "manifests" / "test.jsonl",
            DATA / "panoenv" / "manifests" / "smoke_20.jsonl",
            TEMPLATES / "predictions_panoenv_smoke_template.jsonl",
            20,
        ),
        (
            "hstar-bench-erp-rotated-submit",
            DATA / "hstar-bench-erp" / "manifests" / "erp_rotated_submit.jsonl",
            DATA / "hstar-bench-erp" / "manifests" / "smoke_rotated_submit_20.jsonl",
            TEMPLATES / "predictions_hstar_erp_rotated_submit_smoke_template.jsonl",
            20,
        ),
    ]

    for name, src, dst, template_path, limit in tasks:
        if src.exists():
            rows = load_jsonl(src)[:limit]
        else:
            if name == "hstar-bench-erp-rotated-submit":
                rows = load_jsonl(EXAMPLES / "hstar_erp_submit_manifest.jsonl")
            else:
                report[name] = {"status": "skipped", "reason": f"missing source manifest: {src}"}
                continue
        save_jsonl(dst, rows)
        save_jsonl(template_path, make_prediction_template(rows))
        report[name] = {
            "status": "ok",
            "num_rows": len(rows),
            "smoke_manifest": str(dst),
            "template": str(template_path),
        }

    report["omnispatial"] = build_omnispatial_smoke()

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
