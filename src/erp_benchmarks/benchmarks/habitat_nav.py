from __future__ import annotations

from typing import Any

from .base import BenchmarkAdapter
from ..utils.io import dump_json, load_records
from ..utils.metrics import navigation_report


class HabitatNavBenchmark(BenchmarkAdapter):
    def evaluate(self, args: Any) -> dict[str, Any]:
        records = load_records(args.predictions)
        report = navigation_report(records)
        report["benchmark"] = "habitat-nav"

        if args.report:
            dump_json(args.report, report)
        return report
