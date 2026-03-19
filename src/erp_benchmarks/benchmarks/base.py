from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BenchmarkAdapter(ABC):
    @abstractmethod
    def evaluate(self, args: Any) -> dict[str, Any]:
        raise NotImplementedError
