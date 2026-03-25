from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any


def sample_prompt(sample: dict[str, Any]) -> str:
    prompt = str(sample.get("prompt", sample.get("question", ""))).strip()
    if not prompt:
        raise ValueError("Sample is missing prompt/question.")
    return prompt


def sample_image_path(sample: dict[str, Any]) -> str:
    image_path = sample.get("image_path")
    if not image_path:
        raise ValueError("Sample is missing image_path.")
    return str(image_path)


def build_messages(prompt: str, image_path: str, system_prompt: str | None = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    )
    return messages


def encode_image_as_data_url(image_path: str) -> str:
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "application/octet-stream"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    texts.append(str(item["text"]))
                elif "text" in item and item["text"]:
                    texts.append(str(item["text"]))
            elif item:
                texts.append(str(item))
        return "\n".join(texts).strip()
    return str(content).strip()
