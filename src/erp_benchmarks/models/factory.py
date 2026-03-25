from __future__ import annotations

from argparse import Namespace

from .base import ModelAdapter
from .mlx_qwen_vl import MlxQwenVLAdapter
from .mock import MockModelAdapter
from .openai_compatible import OpenAIAPIAdapter, VllmOpenAIAdapter
from .transformers_vlm import TransformersVLMAdapter


def create_model_adapter(args: Namespace) -> ModelAdapter:
    if args.model == "mock":
        return MockModelAdapter()

    if args.model == "mlx-qwen-vl":
        if not args.model_path:
            raise ValueError("--model-path is required for mlx-qwen-vl")
        return MlxQwenVLAdapter(
            model_path=args.model_path,
            system_prompt=args.system_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    if args.model == "transformers-vlm":
        if not args.model_path:
            raise ValueError("--model-path is required for transformers-vlm")
        return TransformersVLMAdapter(
            model_path=args.model_path,
            processor_path=getattr(args, "processor_path", None),
            system_prompt=args.system_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            device_map=getattr(args, "device_map", "auto"),
            torch_dtype=getattr(args, "torch_dtype", "auto"),
            attn_implementation=getattr(args, "attn_implementation", None),
            min_pixels=getattr(args, "min_pixels", None),
            max_pixels=getattr(args, "max_pixels", None),
            trust_remote_code=getattr(args, "trust_remote_code", False),
        )

    if args.model == "vllm-openai":
        if not getattr(args, "model_name", None):
            raise ValueError("--model-name is required for vllm-openai")
        if not getattr(args, "api_base", None):
            raise ValueError("--api-base is required for vllm-openai")
        return VllmOpenAIAdapter(
            model_name=args.model_name,
            api_base=args.api_base,
            api_key=getattr(args, "api_key", None),
            api_key_env=getattr(args, "api_key_env", "OPENAI_API_KEY"),
            system_prompt=args.system_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=getattr(args, "api_timeout", 120.0),
        )

    if args.model == "openai-api":
        if not getattr(args, "model_name", None):
            raise ValueError("--model-name is required for openai-api")
        api_base = getattr(args, "api_base", None) or "https://api.openai.com/v1"
        return OpenAIAPIAdapter(
            model_name=args.model_name,
            api_base=api_base,
            api_key=getattr(args, "api_key", None),
            api_key_env=getattr(args, "api_key_env", "OPENAI_API_KEY"),
            system_prompt=args.system_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=getattr(args, "api_timeout", 120.0),
        )

    raise ValueError(f"Unsupported model adapter: {args.model}")
