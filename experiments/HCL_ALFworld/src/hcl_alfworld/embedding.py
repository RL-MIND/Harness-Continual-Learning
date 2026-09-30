from __future__ import annotations

import math
import os
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Protocol, Sequence


class EmbedderProtocol(Protocol):
    def embed(self, texts: Sequence[str]) -> List[List[float]]: ...

    def usage_snapshot(self) -> Dict[str, Any]: ...


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return numerator / (left_norm * right_norm)


class OpenAICompatibleEmbedder:
    """Small OpenAI-compatible embedding client for semantic-memory baselines."""

    def __init__(self, config: Dict[str, Any]):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the LLM extra with `pip install -e '.[llm]'`.") from exc

        api_key_env = str(config.get("api_key_env", "OPENAI_API_KEY"))
        configured_api_key = config.get("api_key")
        api_key = (
            str(configured_api_key)
            if configured_api_key is not None
            else os.environ.get(api_key_env)
        )
        if not api_key:
            raise RuntimeError(
                "Embedding retrieval requires embedding.api_key or "
                f"the {api_key_env} environment variable."
            )
        options: Dict[str, Any] = {
            "api_key": api_key,
            "timeout": float(config.get("request_timeout_seconds", 90.0)),
            "max_retries": int(config.get("transport_max_retries", 3)),
        }
        if config.get("base_url"):
            options["base_url"] = str(config["base_url"])
        self.client = OpenAI(**options)
        self.model = str(config["model"])
        self.expected_dimension = (
            int(config["dimension"]) if config.get("dimension") is not None else None
        )
        self.request_dimensions = (
            int(config["request_dimensions"])
            if config.get("request_dimensions") is not None
            else None
        )
        self.max_text_length = int(config.get("max_text_length", 4096))
        self.usage: Dict[str, Any] = {
            "requests": 0,
            "input_tokens": 0,
            "by_model": defaultdict(lambda: {"requests": 0, "input_tokens": 0}),
        }

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        normalized = [str(text)[: self.max_text_length] for text in texts]
        if not normalized:
            return []
        request: Dict[str, Any] = {"model": self.model, "input": normalized}
        # Fixed-dimension local models such as Qwen3-Embedding-4B reject
        # OpenAI's optional Matryoshka `dimensions` argument. `dimension`
        # therefore validates output only; an API parameter is sent solely
        # when the separate `request_dimensions` option is explicit.
        if self.request_dimensions is not None:
            request["dimensions"] = self.request_dimensions
        response = self.client.embeddings.create(**request)
        ordered = sorted(response.data, key=lambda item: int(item.index))
        vectors = [[float(value) for value in item.embedding] for item in ordered]
        if len(vectors) != len(normalized):
            raise RuntimeError(
                f"Embedding endpoint returned {len(vectors)} vectors for {len(normalized)} inputs."
            )
        if self.expected_dimension is not None:
            invalid_dimensions = [
                len(vector) for vector in vectors if len(vector) != self.expected_dimension
            ]
            if invalid_dimensions:
                raise RuntimeError(
                    "Embedding endpoint returned an unexpected vector dimension: "
                    f"expected {self.expected_dimension}, got {invalid_dimensions[0]}."
                )
        tokens = int(getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0)
        self.usage["requests"] += 1
        self.usage["input_tokens"] += tokens
        by_model = self.usage["by_model"][self.model]
        by_model["requests"] += 1
        by_model["input_tokens"] += tokens
        return vectors

    def usage_snapshot(self) -> Dict[str, Any]:
        return {
            "requests": int(self.usage["requests"]),
            "input_tokens": int(self.usage["input_tokens"]),
            "by_model": {model: dict(values) for model, values in self.usage["by_model"].items()},
        }


_LOCAL_EMBED_CACHE: Dict[tuple[str, str, str, str], tuple[Any, Any, threading.Lock]] = {}
_LOCAL_EMBED_CACHE_LOCK = threading.Lock()


class LocalTransformersEmbedder:
    """Produce dense vectors directly from a local encoder model."""

    def __init__(self, config: Dict[str, Any]):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the local extra with `pip install -e '.[local]'.") from exc

        model_path = Path(str(config.get("model_path", ""))).expanduser().resolve()
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(f"Local embedding model directory is incomplete: {model_path}")
        if config.get("request_dimensions") is not None:
            raise ValueError("Local embeddings do not support request_dimensions.")
        dtype_name = str(config.get("dtype", "float32"))
        if dtype_name not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"Unsupported local embedding dtype: {dtype_name}")
        device_map = str(config.get("device_map", "auto"))
        padding_side = config.get("padding_side")
        if padding_side is not None and padding_side not in {"left", "right"}:
            raise ValueError(f"Unsupported local embedding padding_side: {padding_side}")
        cache_key = (str(model_path), dtype_name, device_map, str(padding_side or ""))
        with _LOCAL_EMBED_CACHE_LOCK:
            if cache_key not in _LOCAL_EMBED_CACHE:
                tokenizer_options: Dict[str, Any] = {"local_files_only": True}
                if padding_side is not None:
                    tokenizer_options["padding_side"] = padding_side
                tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_options)
                model = AutoModel.from_pretrained(
                    model_path,
                    local_files_only=True,
                    dtype=getattr(torch, dtype_name),
                    device_map=device_map,
                ).eval()
                _LOCAL_EMBED_CACHE[cache_key] = (tokenizer, model, threading.Lock())
            self.tokenizer, self.model, self._model_lock = _LOCAL_EMBED_CACHE[cache_key]

        self.model_name = str(config.get("model") or model_path.name)
        self.expected_dimension = (
            int(config["dimension"]) if config.get("dimension") is not None else None
        )
        self.max_text_length = int(config.get("max_text_length", 4096))
        self.max_tokens = int(config.get("max_tokens", 8192))
        self.batch_size = int(config.get("batch_size", 8))
        if self.max_tokens <= 0 or self.batch_size <= 0:
            raise ValueError("Local embedding max_tokens and batch_size must be positive.")
        self.pooling = str(config.get("pooling", "cls")).lower()
        if self.pooling not in {"cls", "last_token", "mean"}:
            raise ValueError(f"Unknown local embedding pooling: {self.pooling}")
        self.normalize = bool(config.get("normalize", True))
        self.usage: Dict[str, Any] = {
            "requests": 0,
            "input_tokens": 0,
            "by_model": defaultdict(lambda: {"requests": 0, "input_tokens": 0}),
        }

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        import torch
        import torch.nn.functional as F

        normalized = [str(text)[: self.max_text_length] for text in texts]
        if not normalized:
            return []
        result: List[List[float]] = []
        token_count = 0
        with self._model_lock, torch.inference_mode():
            for start in range(0, len(normalized), self.batch_size):
                batch = normalized[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_tokens,
                    return_tensors="pt",
                )
                token_count += int(encoded["attention_mask"].sum().item())
                encoded = {
                    key: value.to(next(self.model.parameters()).device)
                    for key, value in encoded.items()
                }
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"]
                if self.pooling == "cls":
                    vectors = hidden[:, 0]
                elif self.pooling == "last_token":
                    last_indices = mask.shape[1] - 1 - mask.flip(1).int().argmax(1)
                    vectors = hidden[
                        torch.arange(hidden.shape[0], device=hidden.device), last_indices
                    ]
                else:
                    weights = mask.unsqueeze(-1).to(hidden.dtype)
                    vectors = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1)
                vectors = vectors.float()
                if self.normalize:
                    vectors = F.normalize(vectors, p=2, dim=-1)
                result.extend(vectors.cpu().tolist())

        if self.expected_dimension is not None and any(
            len(vector) != self.expected_dimension for vector in result
        ):
            raise RuntimeError(
                "Local embedding model returned an unexpected vector dimension: "
                f"expected {self.expected_dimension}, got {len(result[0])}."
            )
        self.usage["requests"] += 1
        self.usage["input_tokens"] += token_count
        by_model = self.usage["by_model"][self.model_name]
        by_model["requests"] += 1
        by_model["input_tokens"] += token_count
        return result

    def usage_snapshot(self) -> Dict[str, Any]:
        return {
            "requests": int(self.usage["requests"]),
            "input_tokens": int(self.usage["input_tokens"]),
            "by_model": {model: dict(values) for model, values in self.usage["by_model"].items()},
        }


def make_embedder(config: Dict[str, Any]) -> EmbedderProtocol:
    provider = str(config.get("provider", "openai")).lower()
    if provider == "openai":
        return OpenAICompatibleEmbedder(config)
    if provider == "transformers":
        return LocalTransformersEmbedder(config)
    raise ValueError(f"Unknown embedding provider: {provider}")
