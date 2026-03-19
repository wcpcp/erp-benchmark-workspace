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
from erp_benchmarks.utils.io import load_records


SUPPORTED_BENCHMARKS = {
    benchmark_id
    for benchmark_id, adapter in BENCHMARK_DATASETS.items()
    if adapter.supported_model_generation
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate predictions for one benchmark."
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=sorted(SUPPORTED_BENCHMARKS),
        help="Benchmark id.",
    )
    parser.add_argument(
        "--model",
        default="mock",
        choices=["mock", "mlx-qwen-vl", "transformers-vlm", "vllm-openai", "openai-api"],
        help="Model adapter.",
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--model-name", default=None, help="Served model name for API-based inference.")
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
        "--references",
        default=None,
        help="Optional manifest path. If omitted, use the benchmark's local test manifest.",
    )
    parser.add_argument(
        "--data-root",
        default=str(ROOT / "data"),
        help="Benchmark data root.",
    )
    parser.add_argument(
        "--predictions-out",
        required=True,
        help="Output predictions JSONL path.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split when using local manifests.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of samples for smoke testing.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not call ensure_data(). Use existing local files only.",
    )
    return parser


def save_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = build_parser().parse_args()

    data_root = Path(args.data_root).resolve()
    adapter = get_dataset_adapter(args.benchmark)
    if not args.skip_download:
        adapter.ensure_data(data_root)

    if args.references:
        manifest_path = Path(args.references).resolve()
    else:
        manifest_path = adapter.build_manifest(data_root, split=args.split)

    rows = load_records(manifest_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    model = create_model_adapter(args)
    predictions: list[dict[str, str]] = []
    try:
        for row in rows:
            predictions.append(
                {
                    "id": str(row["id"]),
                    "prediction": model.generate(row),
                }
            )
    finally:
        model.close()

    output_path = Path(args.predictions_out).resolve()
    save_predictions(output_path, predictions)
    print(
        json.dumps(
            {
                "benchmark": args.benchmark,
                "manifest_path": str(manifest_path),
                "predictions_out": str(output_path),
                "num_predictions": len(predictions),
                "model": args.model,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
