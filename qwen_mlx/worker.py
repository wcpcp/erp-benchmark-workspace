from __future__ import annotations

import argparse
import json
import sys

from mlx_vlm import generate, load


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent MLX-VLM worker.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model, processor = load(args.model_path)
    print(json.dumps({"status": "ready"}), flush=True)

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
            prompt = str(payload["prompt"]).strip()
            image_path = str(payload["image_path"])
            if args.system_prompt:
                prompt = f"{args.system_prompt}\n\n{prompt}"
            prediction = generate(
                model,
                processor,
                prompt=prompt,
                image=image_path,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(json.dumps({"prediction": prediction}, ensure_ascii=False), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
