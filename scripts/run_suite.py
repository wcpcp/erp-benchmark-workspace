#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from erp_benchmarks.data import BENCHMARK_DATASETS, get_dataset_adapter
from erp_benchmarks.models import create_model_adapter
from erp_benchmarks.utils.io import dump_json, load_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a unified benchmark suite.")
    parser.add_argument(
        "--benchmarks",
        default="osr-bench,panoenv",
        help="Comma-separated benchmark ids or 'all'.",
    )
    parser.add_argument(
        "--model",
        default="mock",
        choices=["mock", "mlx-qwen-vl", "transformers-vlm", "vllm-openai", "openai-api"],
        help="Model adapter.",
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-timeout", type=float, default=120.0)
    parser.add_argument(
        "--data-root",
        default=str(ROOT / "data"),
    )
    parser.add_argument(
        "--pred-root",
        default=str(ROOT / "preds"),
    )
    parser.add_argument(
        "--report-root",
        default=str(ROOT / "reports"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample cap per benchmark for smoke testing.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use existing local manifests/data only and do not call ensure_data().",
    )
    parser.add_argument(
        "--skip-unsupported",
        action="store_true",
        help="Skip benchmarks that do not support direct model generation.",
    )
    return parser


def resolve_benchmarks(raw: str) -> list[str]:
    if raw == "all":
        return list(BENCHMARK_DATASETS.keys())
    return [item.strip() for item in raw.split(",") if item.strip()]


def save_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = build_parser().parse_args()
    data_root = Path(args.data_root).resolve()
    pred_root = Path(args.pred_root).resolve()
    report_root = Path(args.report_root).resolve()

    model = create_model_adapter(args)
    suite_summary = []

    try:
        for benchmark_id in resolve_benchmarks(args.benchmarks):
            adapter = get_dataset_adapter(benchmark_id)
            if not args.skip_download:
                adapter.ensure_data(data_root)
            manifest_path = adapter.build_manifest(data_root, split="test")

            if not adapter.supported_model_generation:
                status = {
                    "benchmark": benchmark_id,
                    "status": "skipped" if args.skip_unsupported else "external_predictions_required",
                    "reason": "This benchmark currently uses evaluation-only adapters in the unified suite.",
                    "manifest_path": str(manifest_path),
                }
                report_path = report_root / args.model / f"{benchmark_id}.json"
                dump_json(report_path, status)
                suite_summary.append(status)
                if args.skip_unsupported:
                    continue
                continue

            records = load_records(manifest_path)
            if args.limit is not None:
                records = records[: args.limit]

            predictions = []
            for row in records:
                predictions.append(
                    {
                        "id": row["id"],
                        "prediction": model.generate(row),
                    }
                )

            predictions_path = pred_root / args.model / benchmark_id / "predictions.jsonl"
            report_path = report_root / args.model / f"{benchmark_id}.json"
            active_manifest_path = manifest_path
            if args.limit is not None:
                active_manifest_path = pred_root / args.model / benchmark_id / "manifest_subset.jsonl"
                save_predictions(active_manifest_path, records)
            save_predictions(predictions_path, predictions)
            report = adapter.evaluate(active_manifest_path, predictions_path, report_path)
            report["predictions_path"] = str(predictions_path)
            report["manifest_path"] = str(active_manifest_path)
            suite_summary.append(report)

        summary_path = report_root / args.model / "suite_summary.json"
        dump_json(summary_path, {"model": args.model, "benchmarks": suite_summary})
        print(json.dumps({"model": args.model, "benchmarks": suite_summary}, ensure_ascii=False, indent=2))
        return 0
    finally:
        model.close()


if __name__ == "__main__":
    raise SystemExit(main())
