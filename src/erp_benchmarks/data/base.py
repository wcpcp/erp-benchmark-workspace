from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class DatasetAdapter(ABC):
    benchmark_id: str
    task_type: str
    supported_model_generation: bool = True

    @abstractmethod
    def ensure_data(self, data_root: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_manifest(self, data_root: Path, split: str = "test") -> Path:
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        manifest_path: Path,
        predictions_path: Path,
        report_path: Path,
    ) -> dict[str, Any]:
        raise NotImplementedError
