from __future__ import annotations

from typing import Any

from .base import BenchmarkAdapter
from ..utils.io import (
    dump_json,
    infer_answer,
    infer_prediction,
    infer_record_id,
    load_records,
)
from ..utils.metrics import exact_match_report


def _load_hf_records(dataset_id: str, config: str, split: str, cache_dir: str | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    kwargs = {"split": split, "cache_dir": cache_dir}
    if config:
        kwargs["name"] = config
    dataset = load_dataset(dataset_id, **kwargs)
    return [dict(row) for row in dataset]


class OsrBenchBenchmark(BenchmarkAdapter):
    dataset_id = "UUUserna/OSR-Bench"

    def evaluate(self, args: Any) -> dict[str, Any]:
        if args.dataset_source == "hf":
            reference_records = _load_hf_records(
                self.dataset_id, args.config, args.split, args.cache_dir
            )
        elif args.references:
            reference_records = load_records(args.references)
        else:
            raise ValueError("Provide --references or use --dataset-source hf")

        prediction_records = load_records(args.predictions)

        references = {
            infer_record_id(record, idx): infer_answer(record)
            for idx, record in enumerate(reference_records)
        }
        predictions = {
            infer_record_id(record, idx): infer_prediction(record)
            for idx, record in enumerate(prediction_records)
        }

        report = exact_match_report(references, predictions)
        report["benchmark"] = "osr-bench"
        report["config"] = args.config
        report["split"] = args.split

        if args.report:
            dump_json(args.report, report)
        return report
