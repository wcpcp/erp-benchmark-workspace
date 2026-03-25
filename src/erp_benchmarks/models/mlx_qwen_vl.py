from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .base import ModelAdapter


class MlxQwenVLAdapter(ModelAdapter):
    def __init__(
        self,
        model_path: str,
        system_prompt: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        qwen_project = Path(__file__).resolve().parents[3] / "qwen_mlx"
        worker_path = qwen_project / "worker.py"
        self._process = subprocess.Popen(
            [
                "uv",
                "run",
                "--project",
                str(qwen_project),
                "python",
                str(worker_path),
                "--model-path",
                model_path,
                "--max-tokens",
                str(max_tokens),
                "--temperature",
                str(temperature),
                *(
                    ["--system-prompt", system_prompt]
                    if system_prompt
                    else []
                ),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready = self._process.stdout.readline().strip()
        if ready != '{"status": "ready"}':
            stderr = self._process.stderr.read()
            raise RuntimeError(f"Failed to start MLX worker: {ready}\n{stderr}")
        self._system_prompt = system_prompt
        self.name = Path(model_path).name

    def generate(self, sample: dict[str, Any]) -> str:
        image_path = sample.get("image_path")
        prompt = str(sample.get("prompt", sample.get("question", ""))).strip()
        if not prompt:
            raise ValueError("Sample is missing prompt/question.")
        if not image_path:
            raise ValueError("Sample is missing image_path.")

        payload = {
            "prompt": prompt,
            "image_path": str(image_path),
        }
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self._process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline().strip()
        if not line:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise RuntimeError(f"MLX worker returned no output.\n{stderr}")
        message = json.loads(line)
        if "error" in message:
            raise RuntimeError(message["error"])
        return str(message["prediction"]).strip()

    def close(self) -> None:
        if self._process.poll() is None:
            if self._process.stdin:
                self._process.stdin.close()
            self._process.terminate()
