from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests

from .base import ModelAdapter
from .common import encode_image_as_data_url, extract_text_content, sample_image_path, sample_prompt


class OpenAICompatibleAdapter(ModelAdapter):
    def __init__(
        self,
        *,
        adapter_name: str,
        model_name: str,
        api_base: str,
        api_key: str | None,
        api_key_env: str,
        system_prompt: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        self.name = adapter_name
        self._model_name = model_name
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key or os.getenv(api_key_env, "")
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout

    def generate(self, sample: dict[str, Any]) -> str:
        prompt = sample_prompt(sample)
        image_path = sample_image_path(sample)
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": encode_image_as_data_url(image_path)}},
        ]
        messages: list[dict[str, Any]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": content})

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model_name,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        response = requests.post(
            f"{self._api_base}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        body = response.json()
        try:
            message = body["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Unexpected OpenAI-compatible response: {body}") from exc
        return extract_text_content(message)


class VllmOpenAIAdapter(OpenAICompatibleAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(adapter_name="vllm-openai", **kwargs)


class OpenAIAPIAdapter(OpenAICompatibleAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(adapter_name="openai-api", **kwargs)
