#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from _rewrite_task_question_templates import build_parser, metadata_str, rewrite_task_questions


TEMPLATES = [
    "Without changing the current front direction, where does the center of {target_ref} lie relative to the center of {reference_ref} on the observer-centered panoramic ring?",
    "Keeping the current observer orientation fixed, what is the angular direction of the center of {target_ref} relative to the center of {reference_ref}?",
]


def render_question(row, template: str):
    target_ref = metadata_str(row, "target_ref")
    reference_ref = metadata_str(row, "reference_ref")
    if not target_ref or not reference_ref:
        return None
    return template.format(target_ref=target_ref, reference_ref=reference_ref)


def main() -> int:
    parser = build_parser("Rewrite relative_direction_mc questions with the updated observer-centered templates.")
    args = parser.parse_args()
    rewrite_task_questions(
        input_jsonl=Path(args.input_jsonl),
        output_jsonl=Path(args.output_jsonl),
        task_id="relative_direction_mc",
        templates=TEMPLATES,
        render_question=render_question,
        max_failure_examples=int(args.max_failure_examples),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
