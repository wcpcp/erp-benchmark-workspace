from __future__ import annotations

import argparse
import json
from typing import Any

from .benchmarks import (
    HabitatNavBenchmark,
    HstarBenchBenchmark,
    HstarBenchErpBenchmark,
    Loc360Benchmark,
    OmniSpatialBenchmark,
    OsrBenchBenchmark,
    PanoEnvBenchmark,
)
from .registry import get_benchmark, list_benchmarks


ADAPTERS = {
    "360loc": Loc360Benchmark(),
    "hstar-bench": HstarBenchBenchmark(),
    "hstar-bench-erp": HstarBenchErpBenchmark(),
    "omnispatial": OmniSpatialBenchmark(),
    "osr-bench": OsrBenchBenchmark(),
    "panoenv": PanoEnvBenchmark(),
    "habitat-nav": HabitatNavBenchmark(),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified ERP / 360 benchmark helper."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List benchmark registry entries.")

    describe_parser = subparsers.add_parser("describe", help="Show one benchmark entry.")
    describe_parser.add_argument("benchmark")

    eval_parser = subparsers.add_parser("evaluate", help="Run one benchmark adapter.")
    eval_parser.add_argument("--benchmark", required=True, choices=sorted(ADAPTERS.keys()))
    eval_parser.add_argument("--predictions", required=True)
    eval_parser.add_argument("--references", default=None)
    eval_parser.add_argument(
        "--dataset-source",
        choices=["local", "hf"],
        default="local",
        help="Load references from a local file or directly from Hugging Face.",
    )
    eval_parser.add_argument("--config", default=None)
    eval_parser.add_argument("--split", default="test")
    eval_parser.add_argument("--cache-dir", default=None)
    eval_parser.add_argument("--report", default=None)
    eval_parser.add_argument(
        "--coordinate-system",
        choices=["cartesian", "geographic"],
        default="cartesian",
        help="Used by 360Loc.",
    )
    eval_parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[1.0, 5.0, 10.0],
        help="Used by 360Loc success@threshold metrics.",
    )

    return parser


def _cmd_list() -> int:
    benchmarks = list_benchmarks()
    payload = []
    for benchmark_id, meta in benchmarks.items():
        payload.append(
            {
                "id": benchmark_id,
                "display_name": meta["display_name"],
                "status": meta["status"],
                "primary_focus": meta["primary_focus"],
            }
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_describe(benchmark_id: str) -> int:
    print(json.dumps(get_benchmark(benchmark_id), ensure_ascii=False, indent=2))
    return 0


def _cmd_evaluate(args: Any) -> int:
    report = ADAPTERS[args.benchmark].evaluate(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list":
        return _cmd_list()
    if args.command == "describe":
        return _cmd_describe(args.benchmark)
    if args.command == "evaluate":
        return _cmd_evaluate(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2
