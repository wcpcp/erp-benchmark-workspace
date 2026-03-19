from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkChoice:
    key: str
    text: str
    entity_id: Optional[str] = None
    bbox_erp: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"key": self.key, "text": self.text}
        if self.entity_id:
            payload["entity_id"] = self.entity_id
        if self.bbox_erp is not None:
            payload["bbox_erp"] = self.bbox_erp
        return payload


@dataclass
class BenchmarkItem:
    item_id: str
    scene_id: str
    task_id: str
    ability_group: str
    release_phase: str
    question: str
    answer: Any
    answer_format: str
    difficulty: str
    image_path: str
    options: List[BenchmarkChoice] = field(default_factory=list)
    target_entities: List[str] = field(default_factory=list)
    erp_slices: List[str] = field(default_factory=list)
    metadata_tier: str = "base"
    quality_score: float = 0.0
    gt_sources: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.item_id,
            "scene_id": self.scene_id,
            "task_id": self.task_id,
            "ability_group": self.ability_group,
            "release_phase": self.release_phase,
            "question": self.question,
            "answer": self.answer,
            "answer_format": self.answer_format,
            "difficulty": self.difficulty,
            "image_path": self.image_path,
            "options": [option.to_dict() for option in self.options],
            "target_entities": self.target_entities,
            "erp_slices": self.erp_slices,
            "metadata_tier": self.metadata_tier,
            "quality_score": round(self.quality_score, 4),
            "gt_sources": self.gt_sources,
            "notes": self.notes,
        }
