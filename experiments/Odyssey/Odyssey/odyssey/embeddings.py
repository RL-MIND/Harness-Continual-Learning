from __future__ import annotations

import os
from typing import Any

import requests


class OpenAICompatibleEmbeddings:
    """Minimal LangChain-compatible client for an OpenAI embeddings endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60,
        batch_size: int = 32,
        query_instruction: str = "",
    ):
        self.endpoint = base_url.rstrip("/") + "/embeddings"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.batch_size = max(1, int(batch_size))
        self.query_instruction = query_instruction.strip()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(list(texts))

    def embed_query(self, text: str) -> list[float]:
        query = str(text)
        if self.query_instruction:
            query = f"Instruct: {self.query_instruction}\nQuery: {query}"
        return self._embed([query])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = requests.post(
                self.endpoint,
                headers=headers,
                json={"model": self.model, "input": batch},
                timeout=self.timeout,
            )
            try:
                response.raise_for_status()
                payload = response.json()
                entries = sorted(payload["data"], key=lambda item: int(item["index"]))
                batch_vectors = [entry["embedding"] for entry in entries]
            except Exception as exc:  # noqa: BLE001
                detail = response.text[:500] if response is not None else ""
                raise RuntimeError(
                    f"Embedding request failed for {self.model} at {self.endpoint}: {detail}"
                ) from exc
            if len(batch_vectors) != len(batch):
                raise RuntimeError(
                    f"Embedding endpoint returned {len(batch_vectors)} vectors for {len(batch)} texts"
                )
            vectors.extend(batch_vectors)
        return vectors


def build_embeddings(config: str | dict[str, Any]):
    """Build either the legacy local HuggingFace or an API embedding client."""
    if isinstance(config, dict):
        provider = str(config.get("provider", "")).strip().lower()
        if provider == "openai_compatible":
            api_key = str(config.get("api_key", ""))
            api_key_env = str(config.get("api_key_env", "")).strip()
            if api_key_env:
                api_key = os.getenv(api_key_env, api_key)
            return OpenAICompatibleEmbeddings(
                base_url=str(config["base_url"]),
                api_key=api_key,
                model=str(config["model"]),
                timeout=float(config.get("timeout", 60)),
                batch_size=int(config.get("batch_size", 32)),
                query_instruction=str(config.get("query_instruction", "")),
            )
        if provider not in {"", "huggingface"}:
            raise ValueError(f"Unsupported embedding provider: {provider}")
        model_name = str(config.get("model") or config.get("model_name") or "")
    else:
        model_name = str(config or "")

    from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=model_name)
