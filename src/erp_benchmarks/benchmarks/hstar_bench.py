from __future__ import annotations

from typing import Any

from .base import BenchmarkAdapter
from ..utils.io import dump_json, load_records
from ..utils.metrics import navigation_report


class HstarBenchBenchmark(BenchmarkAdapter):
    def evaluate(self, args: Any) -> dict[str, Any]:
        records = load_records(args.predictions)
        report = navigation_report(records)
        report["benchmark"] = "hstar-bench"
        report["note"] = (
            "This unified adapter aggregates search or navigation-style metrics "
            "exported by an H*Bench pipeline or an ERP-adapted HOS/HPS variant."
        )
        if args.report:
            dump_json(args.report, report)
        return report
