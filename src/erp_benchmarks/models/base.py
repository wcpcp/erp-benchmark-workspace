from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModelAdapter(ABC):
    name: str

    @abstractmethod
    def generate(self, sample: dict[str, Any]) -> str:
        raise NotImplementedError

    def close(self) -> None:
        return None
