#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from _rewrite_task_question_templates import build_parser, metadata_str, rewrite_task_questions


TEMPLATES = [
    "In the current observer-centered 3D frame, which spatial relation best describes {entity_a_ref} relative to {entity_b_ref}?",
    "From the current observer position, which 3D spatial relation best matches {entity_a_ref} relative to {entity_b_ref}?",
]


def render_question(row, template: str):
    entity_a_ref = metadata_str(row, "entity_a_ref")
    entity_b_ref = metadata_str(row, "entity_b_ref")
    if not entity_a_ref or not entity_b_ref:
        return None
    return template.format(entity_a_ref=entity_a_ref, entity_b_ref=entity_b_ref)


def main() -> int:
    parser = build_parser("Rewrite relative_3d_position_mc questions with the updated observer-centered 3D templates.")
    args = parser.parse_args()
    rewrite_task_questions(
        input_jsonl=Path(args.input_jsonl),
        output_jsonl=Path(args.output_jsonl),
        task_id="relative_3d_position_mc",
        templates=TEMPLATES,
        render_question=render_question,
        max_failure_examples=int(args.max_failure_examples),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
