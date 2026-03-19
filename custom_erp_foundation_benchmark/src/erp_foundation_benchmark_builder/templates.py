from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = ROOT / "config" / "question_templates.json"


def load_templates() -> dict[str, list[str]]:
    return json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))


def render_question(task_id: str, variant_index: int = 0, **kwargs: str) -> str:
    templates = load_templates()
    variants = templates[task_id]
    template = variants[variant_index % len(variants)]
    return template.format(**kwargs)
