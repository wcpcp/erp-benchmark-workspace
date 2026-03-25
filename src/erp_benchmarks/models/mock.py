from __future__ import annotations

from .base import ModelAdapter


class MockModelAdapter(ModelAdapter):
    name = "mock"

    def generate(self, sample: dict[str, object]) -> str:
        return str(sample.get("answer", ""))
