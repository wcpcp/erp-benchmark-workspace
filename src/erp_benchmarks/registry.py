from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def registry_path() -> Path:
    return Path(__file__).resolve().parents[2] / "registry.yaml"


def load_registry() -> dict[str, Any]:
    with registry_path().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def list_benchmarks() -> dict[str, Any]:
    return load_registry()["benchmarks"]


def get_benchmark(benchmark_id: str) -> dict[str, Any]:
    benchmarks = list_benchmarks()
    if benchmark_id not in benchmarks:
        raise KeyError(f"Unknown benchmark: {benchmark_id}")
    return benchmarks[benchmark_id]
