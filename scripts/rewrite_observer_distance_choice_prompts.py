#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from _rewrite_task_question_templates import build_parser, rewrite_task_questions


TEMPLATES = [
    "Which listed object is physically closest to the current observer in the 3D scene?",
    "From the current observer position, which listed object is nearest in 3D space?",
]


def render_question(row, template: str):
    return template


def main() -> int:
    parser = build_parser("Rewrite observer_distance_choice questions with the updated 3D-distance templates.")
    args = parser.parse_args()
    rewrite_task_questions(
        input_jsonl=Path(args.input_jsonl),
        output_jsonl=Path(args.output_jsonl),
        task_id="observer_distance_choice",
        templates=TEMPLATES,
        render_question=render_question,
        max_failure_examples=int(args.max_failure_examples),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
