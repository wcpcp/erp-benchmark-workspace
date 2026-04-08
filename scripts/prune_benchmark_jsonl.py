#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class DeleteRule:
    raw: str
    path_text: str
    basename: str
    stem: str
    is_prefix: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove benchmark items from JSONL files based on a delete.txt list of image-like filenames or prefixes."
    )
    parser.add_argument(
        "--jsonl",
        nargs="+",
        required=True,
        help="One or more JSONL files to prune, e.g. benchmark_public.jsonl benchmark_public_prompts.jsonl benchmark_public_references.jsonl",
    )
    parser.add_argument(
        "--delete-txt",
        required=True,
        help="Path to the delete.txt file. Each line can be an exact image path or a prefix ending with '*'.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input JSONL files in place. By default, writes sibling *.filtered.jsonl files.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="When used with --in-place, create a .bak copy before overwriting each JSONL file.",
    )
    return parser.parse_args()


def iter_delete_rules(path: Path) -> List[DeleteRule]:
    rules: List[DeleteRule] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        is_prefix = line.endswith("*")
        path_text = line[:-1] if is_prefix else line
        basename = Path(path_text).name
        stem = Path(basename).stem if Path(basename).suffix else basename
        rules.append(
            DeleteRule(
                raw=line,
                path_text=path_text,
                basename=basename,
                stem=stem,
                is_prefix=is_prefix,
            )
        )
    return rules


def row_match_tokens(row: Dict[str, object]) -> Dict[str, str]:
    item_id = str(row.get("item_id", "") or "")
    scene_id = str(row.get("scene_id", "") or "")
    image_path = str(row.get("image_path", "") or "")
    image_name = Path(image_path).name if image_path else ""
    image_stem = Path(image_name).stem if image_name else ""
    return {
        "item_id": item_id,
        "scene_id": scene_id,
        "image_path": image_path,
        "image_name": image_name,
        "image_stem": image_stem,
    }


def rule_matches_row(rule: DeleteRule, row: Dict[str, object]) -> bool:
    tokens = row_match_tokens(row)

    if rule.is_prefix:
        prefix = rule.basename or rule.stem
        if not prefix:
            return False
        return any(
            value.startswith(prefix)
            for value in (
                tokens["item_id"],
                tokens["scene_id"],
                tokens["image_name"],
                tokens["image_stem"],
                tokens["image_path"],
            )
            if value
        )

    exact_candidates = {
        tokens["image_path"],
        tokens["image_name"],
        tokens["image_stem"],
        tokens["item_id"],
        tokens["scene_id"],
    }
    if rule.path_text in exact_candidates:
        return True
    if rule.basename in exact_candidates:
        return True
    if rule.stem in exact_candidates:
        return True

    # If a user provided an exact source image filename, allow matching rows whose
    # source image basename is identical, even when item_id carries derived suffixes.
    basename_suffix = Path(rule.basename).suffix.lower()
    if basename_suffix in IMAGE_SUFFIXES and tokens["image_name"] == rule.basename:
        return True
    return False


def find_matching_rule(row: Dict[str, object], rules: Sequence[DeleteRule]) -> DeleteRule | None:
    for rule in rules:
        if rule_matches_row(rule, row):
            return rule
    return None


def output_path_for(path: Path, in_place: bool) -> Path:
    if in_place:
        return path
    return path.with_name(f"{path.stem}.filtered{path.suffix}")


def prune_jsonl(path: Path, rules: Sequence[DeleteRule], out_path: Path, backup: bool) -> Dict[str, object]:
    kept_lines: List[str] = []
    removed: List[Tuple[str, str]] = []
    total = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            matched_rule = find_matching_rule(row, rules)
            if matched_rule is not None:
                removed.append((str(row.get("item_id", "")), matched_rule.raw))
                continue
            kept_lines.append(line if line.endswith("\n") else f"{line}\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path == path and backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    with out_path.open("w", encoding="utf-8") as handle:
        handle.writelines(kept_lines)

    return {
        "path": str(path),
        "output_path": str(out_path),
        "total_rows": total,
        "removed_rows": len(removed),
        "kept_rows": len(kept_lines),
        "removed_examples": [{"item_id": item_id, "matched_rule": rule} for item_id, rule in removed[:20]],
    }


def main() -> int:
    args = parse_args()
    delete_path = Path(args.delete_txt)
    if not delete_path.exists():
        raise FileNotFoundError(f"Delete list not found: {delete_path}")

    rules = iter_delete_rules(delete_path)
    if not rules:
        raise ValueError(f"No usable delete rules found in: {delete_path}")

    summaries: List[Dict[str, object]] = []
    for raw_path in args.jsonl:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"JSONL file not found: {path}")
        out_path = output_path_for(path, in_place=bool(args.in_place))
        summary = prune_jsonl(path, rules, out_path, backup=bool(args.backup and args.in_place))
        summaries.append(summary)

    print(json.dumps({"delete_rule_count": len(rules), "files": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
