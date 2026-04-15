#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from _rewrite_task_question_templates import build_parser, metadata_str, rewrite_task_questions


TEMPLATES = [
    "In the observer-centered ERP panorama, which direction sector contains the center of {target_ref}?",
    "Using the ERP image center as the current front direction, which panorama sector does the center of {target_ref} fall into?",
]


def render_question(row, template: str):
    target_ref = metadata_str(row, "target_ref")
    if not target_ref:
        return None
    return template.format(target_ref=target_ref)


def main() -> int:
    parser = build_parser("Rewrite absolute_direction_mc questions with the updated observer-centered templates.")
    args = parser.parse_args()
    rewrite_task_questions(
        input_jsonl=Path(args.input_jsonl),
        output_jsonl=Path(args.output_jsonl),
        task_id="absolute_direction_mc",
        templates=TEMPLATES,
        render_question=render_question,
        max_failure_examples=int(args.max_failure_examples),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
