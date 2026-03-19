from __future__ import annotations

from .habitat_nav import HabitatNavDataset
from .hstar_bench import HstarBenchDataset, HstarBenchErpDataset
from .loc360 import Loc360Dataset
from .omnispatial import OmniSpatialDataset
from .osr_bench import OsrBenchDataset
from .panoenv import PanoEnvDataset


BENCHMARK_DATASETS = {
    "360loc": Loc360Dataset(),
    "hstar-bench": HstarBenchDataset(),
    "hstar-bench-erp": HstarBenchErpDataset(),
    "omnispatial": OmniSpatialDataset(),
    "osr-bench": OsrBenchDataset(),
    "panoenv": PanoEnvDataset(),
    "habitat-nav": HabitatNavDataset(),
}


def get_dataset_adapter(benchmark_id: str):
    if benchmark_id not in BENCHMARK_DATASETS:
        raise KeyError(f"Unknown benchmark dataset: {benchmark_id}")
    return BENCHMARK_DATASETS[benchmark_id]
