from __future__ import annotations

from typing import Any

from .base import BenchmarkAdapter
from ..utils.io import dump_json, infer_record_id, load_records
from ..utils.metrics import localization_report


def _coerce_float(record: dict[str, Any], keys: tuple[str, ...]) -> float:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return float(record[key])
    raise KeyError(f"Missing coordinate field from choices: {keys}")


def _extract_ground_truth(record: dict[str, Any], coordinate_system: str) -> list[float]:
    if coordinate_system == "geographic":
        return [
            _coerce_float(record, ("gt_lat", "lat", "latitude", "ground_truth_lat")),
            _coerce_float(record, ("gt_lon", "lon", "longitude", "ground_truth_lon")),
        ]
    return [
        _coerce_float(record, ("gt_x", "x", "ground_truth_x")),
        _coerce_float(record, ("gt_y", "y", "ground_truth_y")),
        _coerce_float(record, ("gt_z", "z", "ground_truth_z")),
    ]


def _extract_prediction(record: dict[str, Any], coordinate_system: str) -> list[float]:
    if coordinate_system == "geographic":
        return [
            _coerce_float(record, ("pred_lat", "prediction_lat", "latitude_pred")),
            _coerce_float(record, ("pred_lon", "prediction_lon", "longitude_pred")),
        ]
    return [
        _coerce_float(record, ("pred_x", "prediction_x")),
        _coerce_float(record, ("pred_y", "prediction_y")),
        _coerce_float(record, ("pred_z", "prediction_z")),
    ]


class Loc360Benchmark(BenchmarkAdapter):
    def evaluate(self, args: Any) -> dict[str, Any]:
        prediction_records = load_records(args.predictions)

        if args.references:
            reference_records = load_records(args.references)
            references = {
                infer_record_id(record, idx): _extract_ground_truth(record, args.coordinate_system)
                for idx, record in enumerate(reference_records)
            }
            predictions = {
                infer_record_id(record, idx): _extract_prediction(record, args.coordinate_system)
                for idx, record in enumerate(prediction_records)
            }
            pairs = []
            for sample_id, gt in references.items():
                if sample_id in predictions:
                    pairs.append({"id": sample_id, "ground_truth": gt, "prediction": predictions[sample_id]})
        else:
            pairs = [
                {
                    "id": infer_record_id(record, idx),
                    "ground_truth": _extract_ground_truth(record, args.coordinate_system),
                    "prediction": _extract_prediction(record, args.coordinate_system),
                }
                for idx, record in enumerate(prediction_records)
            ]

        report = localization_report(pairs, args.thresholds, args.coordinate_system)
        report["benchmark"] = "360loc"
        report["coordinate_system"] = args.coordinate_system
        report["thresholds"] = args.thresholds

        if args.report:
            dump_json(args.report, report)
        return report
