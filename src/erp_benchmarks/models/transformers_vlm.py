from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import ModelAdapter
from .common import build_messages, sample_image_path, sample_prompt


_DTYPE_MAP = {
    "auto": "auto",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float32": "float32",
}


class TransformersVLMAdapter(ModelAdapter):
    def __init__(
        self,
        model_path: str,
        processor_path: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        device_map: str | None = "auto",
        torch_dtype: str = "auto",
        attn_implementation: str | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                "transformers-vlm requires `transformers` and `torch` in the active environment."
            ) from exc

        processor_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels

        model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if device_map:
            model_kwargs["device_map"] = device_map
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        dtype_key = _DTYPE_MAP.get(torch_dtype, torch_dtype)
        if dtype_key != "auto":
            model_kwargs["torch_dtype"] = getattr(torch, dtype_key)

        self._processor = AutoProcessor.from_pretrained(
            processor_path or model_path,
            **processor_kwargs,
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            **model_kwargs,
        )
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._torch = torch
        self.name = Path(model_path).name

    def generate(self, sample: dict[str, Any]) -> str:
        image_path = sample_image_path(sample)
        prompt = sample_prompt(sample)
        messages = build_messages(prompt, image_path, self._system_prompt)
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self._model.device)
        input_len = len(inputs.input_ids[0])
        generation_kwargs = {
            "max_new_tokens": self._max_tokens,
            "do_sample": self._temperature > 0,
        }
        if self._temperature > 0:
            generation_kwargs["temperature"] = self._temperature
        with self._torch.no_grad():
            output_ids = self._model.generate(**inputs, **generation_kwargs)
        text = self._processor.batch_decode(
            output_ids[:, input_len:],
            skip_special_tokens=True,
        )[0]
        return str(text).strip()
