from __future__ import annotations

import json
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, Optional, Protocol


class StructuredLLMProtocol(Protocol):
    usage: Dict[str, Any]

    def complete_json(
        self,
        component: str,
        system_prompt: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]: ...

    def usage_snapshot(self) -> Dict[str, Any]: ...


class OpenAICompatibleLLM:
    """Shared structured-reasoning client used by every semantic HCL component."""

    def __init__(self, config: Dict[str, Any]):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the LLM extra with `pip install -e '.[llm]'`.") from exc

        api_key_env = str(config.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is required for the LLM-driven harness.")
        options: Dict[str, Any] = {"api_key": api_key}
        if config.get("base_url"):
            options["base_url"] = config["base_url"]
        options["timeout"] = float(config.get("request_timeout_seconds", 90.0))
        # Application-level retries below provide consistent behavior and logging
        # across OpenAI-compatible providers, so disable the SDK's hidden retries.
        options["max_retries"] = 0
        self.client = OpenAI(**options)
        self.default_model = str(config.get("model", "gpt-4.1-mini"))
        self.component_models = dict(config.get("component_models", {}))
        self.temperature = float(config.get("temperature", 0.0))
        self.component_temperatures = dict(config.get("component_temperatures", {}))
        self.thinking_enabled = config.get("thinking_enabled")
        self.component_thinking_enabled = dict(
            config.get("component_thinking_enabled", {})
        )
        self.reasoning_effort = config.get("reasoning_effort")
        self.component_reasoning_efforts = dict(
            config.get("component_reasoning_efforts", {})
        )
        raw_max_tokens = config.get("max_tokens")
        self.max_tokens = None if raw_max_tokens is None else int(raw_max_tokens)
        self.component_max_tokens = {
            str(component): int(value)
            for component, value in dict(config.get("component_max_tokens", {})).items()
        }
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")
        if any(value <= 0 for value in self.component_max_tokens.values()):
            raise ValueError("component_max_tokens values must be positive integers.")
        self.max_retries = int(config.get("max_retries", 2))
        self.transport_max_retries = max(
            0, int(config.get("transport_max_retries", 8))
        )
        self.transport_retry_base_seconds = max(
            0.0, float(config.get("transport_retry_base_seconds", 1.0))
        )
        self.transport_retry_max_seconds = max(
            self.transport_retry_base_seconds,
            float(config.get("transport_retry_max_seconds", 60.0)),
        )
        self.transport_retry_jitter = max(
            0.0, float(config.get("transport_retry_jitter", 0.25))
        )
        self.json_mode = bool(config.get("json_mode", True))
        self.usage: Dict[str, Any] = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "by_component": defaultdict(
                lambda: {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}
            ),
        }

    @staticmethod
    def _repair_missing_commas(content: str, max_repairs: int = 8) -> Dict[str, Any]:
        """Conservatively repair separators omitted between JSON values.

        Small local instruction-tuned models occasionally omit a comma between
        two otherwise valid object members. Only repair the exact decoder error
        and only when the surrounding characters form an unambiguous value/key
        boundary. Other malformed output is left for a structured retry.
        """
        candidate = content
        for _ in range(max_repairs):
            try:
                parsed = json.loads(candidate)
                if not isinstance(parsed, dict):
                    raise TypeError("LLM response must be a JSON object.")
                return parsed
            except json.JSONDecodeError as exc:
                if exc.msg != "Expecting ',' delimiter":
                    raise
                position = exc.pos
                previous = position - 1
                while previous >= 0 and candidate[previous].isspace():
                    previous -= 1
                following = position
                while following < len(candidate) and candidate[following].isspace():
                    following += 1
                if (
                    previous < 0
                    or following >= len(candidate)
                    or candidate[previous] not in {'"', "}", "]"}
                    or candidate[following] != '"'
                ):
                    raise
                candidate = candidate[:following] + "," + candidate[following:]
        return json.loads(candidate)

    @classmethod
    def _parse_json(cls, content: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.S)
            if not match:
                raise
            parsed = cls._repair_missing_commas(match.group(0))
        if not isinstance(parsed, dict):
            raise TypeError("LLM response must be a JSON object.")
        return parsed

    def complete_json(
        self,
        component: str,
        system_prompt: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        base_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            },
        ]
        messages = list(base_messages)
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            thinking_enabled = self.component_thinking_enabled.get(
                component, self.thinking_enabled
            )
            reasoning_effort = self.component_reasoning_efforts.get(
                component, self.reasoning_effort
            )
            max_tokens = self.component_max_tokens.get(component, self.max_tokens)
            request: Dict[str, Any] = {
                "model": str(self.component_models.get(component, self.default_model)),
                "messages": messages,
            }
            if max_tokens is not None:
                request["max_tokens"] = max_tokens
            # DeepSeek thinking mode ignores sampling parameters, so omit temperature
            # when thinking is explicitly enabled. Keeping it for other modes preserves
            # the behavior of generic OpenAI-compatible backends.
            if thinking_enabled is not True:
                request["temperature"] = float(
                    self.component_temperatures.get(component, self.temperature)
                )
            if thinking_enabled is not None:
                request["extra_body"] = {
                    "thinking": {
                        "type": "enabled" if bool(thinking_enabled) else "disabled"
                    }
                }
            if reasoning_effort is not None and thinking_enabled is not False:
                request["reasoning_effort"] = str(reasoning_effort)
            if self.json_mode:
                request["response_format"] = {"type": "json_object"}
            response = self._create_with_transport_retry(component, request)
            self._record_usage(component, response.usage)
            content = response.choices[0].message.content or "{}"
            try:
                return self._parse_json(content)
            except (json.JSONDecodeError, TypeError) as exc:
                last_error = exc
                # Do not echo the entire malformed response back to the model:
                # it grows the context and encourages the same syntax pattern.
                messages = list(base_messages)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous response was rejected by the JSON parser with: "
                            f"{exc}. Regenerate the object from scratch. Use double-quoted "
                            "keys and strings, put a comma between every object member and "
                            "array item, and return one concise JSON object only. "
                            f"This is structured retry {attempt + 1}."
                        ),
                    }
                )
        raise RuntimeError(f"{component} did not return valid JSON: {last_error}")

    @staticmethod
    def _status_code(exc: Exception) -> Optional[int]:
        status = getattr(exc, "status_code", None)
        if status is None:
            status = getattr(getattr(exc, "response", None), "status_code", None)
        try:
            return int(status) if status is not None else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _is_retryable_transport_error(cls, exc: Exception) -> bool:
        status = cls._status_code(exc)
        if status is not None:
            return status in {408, 409, 429} or 500 <= status <= 599
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return True
        # The SDK wraps httpx errors in these classes without a status code.
        return exc.__class__.__name__ in {
            "APIConnectionError",
            "APITimeoutError",
            "ConnectError",
            "ConnectTimeout",
            "ReadError",
            "ReadTimeout",
            "RemoteProtocolError",
            "WriteError",
            "WriteTimeout",
        }

    @staticmethod
    def _response_headers(exc: Exception) -> Any:
        response = getattr(exc, "response", None)
        return getattr(response, "headers", {}) if response is not None else {}

    def _retry_after_seconds(self, exc: Exception) -> Optional[float]:
        headers = self._response_headers(exc)
        retry_after_ms = headers.get("retry-after-ms") if headers else None
        if retry_after_ms is not None:
            try:
                return max(0.0, float(retry_after_ms) / 1000.0)
            except (TypeError, ValueError):
                pass
        retry_after = headers.get("retry-after") if headers else None
        if retry_after is None:
            return None
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            try:
                delay = parsedate_to_datetime(str(retry_after)).timestamp() - time.time()
                return max(0.0, delay)
            except (TypeError, ValueError, OverflowError):
                return None

    def _retry_delay(self, exc: Exception, retry_index: int) -> float:
        retry_after = self._retry_after_seconds(exc)
        if retry_after is not None:
            base_delay = retry_after
        else:
            base_delay = self.transport_retry_base_seconds * (2**retry_index)
        base_delay = min(base_delay, self.transport_retry_max_seconds)
        if base_delay == 0.0 or self.transport_retry_jitter == 0.0:
            return base_delay
        spread = base_delay * self.transport_retry_jitter
        return max(0.0, base_delay + random.uniform(-spread, spread))

    def _create_with_transport_retry(
        self, component: str, request: Dict[str, Any]
    ) -> Any:
        for retry_index in range(self.transport_max_retries + 1):
            try:
                return self.client.chat.completions.create(**request)
            except Exception as exc:
                if (
                    not self._is_retryable_transport_error(exc)
                    or retry_index >= self.transport_max_retries
                ):
                    raise
                delay = self._retry_delay(exc, retry_index)
                status = self._status_code(exc)
                reason = f"HTTP {status}" if status is not None else exc.__class__.__name__
                print(
                    f"[{component}] transient LLM error ({reason}); retry "
                    f"{retry_index + 1}/{self.transport_max_retries} in {delay:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

    def _record_usage(self, component: str, usage: Any) -> None:
        self.usage["requests"] += 1
        component_usage = self.usage["by_component"][component]
        component_usage["requests"] += 1
        if usage is None:
            return
        prompt_tokens = int(usage.prompt_tokens or 0)
        completion_tokens = int(usage.completion_tokens or 0)
        self.usage["prompt_tokens"] += prompt_tokens
        self.usage["completion_tokens"] += completion_tokens
        component_usage["prompt_tokens"] += prompt_tokens
        component_usage["completion_tokens"] += completion_tokens

    def usage_snapshot(self) -> Dict[str, Any]:
        return {
            "requests": self.usage["requests"],
            "prompt_tokens": self.usage["prompt_tokens"],
            "completion_tokens": self.usage["completion_tokens"],
            "by_component": {
                key: dict(value) for key, value in self.usage["by_component"].items()
            },
        }


_LOCAL_MODEL_CACHE: Dict[tuple[str, str, str], tuple[Any, Any, threading.Lock]] = {}
_LOCAL_MODEL_CACHE_LOCK = threading.Lock()


class LocalTransformersLLM:
    """Run a text-only chat model from local weights without an HTTP server."""

    def __init__(self, config: Dict[str, Any]):
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the local extra with `pip install -e '.[local]'.") from exc

        model_path = Path(str(config.get("model_path", ""))).expanduser().resolve()
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(f"Local LLM model directory is incomplete: {model_path}")
        dtype_name = str(config.get("dtype", "bfloat16"))
        if dtype_name not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"Unsupported local LLM dtype: {dtype_name}")
        device_map = str(config.get("device_map", "auto"))
        cache_key = (str(model_path), dtype_name, device_map)
        with _LOCAL_MODEL_CACHE_LOCK:
            if cache_key not in _LOCAL_MODEL_CACHE:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, local_files_only=True
                )
                model = AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    local_files_only=True,
                    dtype=getattr(torch, dtype_name),
                    device_map=device_map,
                ).eval()
                _LOCAL_MODEL_CACHE[cache_key] = (tokenizer, model, threading.Lock())
            self.tokenizer, self.model, self._generation_lock = _LOCAL_MODEL_CACHE[cache_key]

        self.model_name = str(config.get("model") or model_path.name)
        self.max_retries = int(config.get("max_retries", 2))
        self.max_tokens = int(config.get("max_tokens", 2048))
        self.component_max_tokens = {
            str(key): int(value)
            for key, value in dict(config.get("component_max_tokens", {})).items()
        }
        if self.max_tokens <= 0 or any(
            value <= 0 for value in self.component_max_tokens.values()
        ):
            raise ValueError("Local max_tokens values must be positive integers.")
        self.thinking_enabled = bool(config.get("thinking_enabled", False))
        self.component_thinking_enabled = dict(
            config.get("component_thinking_enabled", {})
        )
        self.temperature = float(config.get("temperature", 0.0))
        self.component_temperatures = dict(
            config.get("component_temperatures", {})
        )
        self.top_p = float(config.get("top_p", 0.95))
        self.usage: Dict[str, Any] = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "by_component": defaultdict(
                lambda: {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}
            ),
        }

    def complete_json(
        self,
        component: str,
        system_prompt: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        import torch

        base_messages = [
            {
                "role": "system",
                "content": system_prompt
                + "\nReturn exactly one JSON object without Markdown or explanation.",
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            },
        ]
        messages = list(base_messages)
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            thinking_enabled = bool(
                self.component_thinking_enabled.get(component, self.thinking_enabled)
            )
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=thinking_enabled,
                return_tensors="pt",
                return_dict=True,
            )
            prompt_tokens = int(encoded["input_ids"].shape[-1])
            encoded = {
                key: value.to(next(self.model.parameters()).device)
                for key, value in encoded.items()
            }
            temperature = float(
                self.component_temperatures.get(component, self.temperature)
            )
            generation: Dict[str, Any] = {
                "max_new_tokens": self.component_max_tokens.get(
                    component, self.max_tokens
                ),
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                generation["temperature"] = temperature
                generation["top_p"] = self.top_p
            with self._generation_lock, torch.inference_mode():
                output = self.model.generate(**encoded, **generation)
            completion = output[0][prompt_tokens:]
            completion_tokens = int(completion.shape[-1])
            content = self.tokenizer.decode(
                completion, skip_special_tokens=True
            ).strip()
            if "</think>" in content:
                content = content.rsplit("</think>", 1)[-1].strip()
            self.usage["requests"] += 1
            self.usage["prompt_tokens"] += prompt_tokens
            self.usage["completion_tokens"] += completion_tokens
            component_usage = self.usage["by_component"][component]
            component_usage["requests"] += 1
            component_usage["prompt_tokens"] += prompt_tokens
            component_usage["completion_tokens"] += completion_tokens
            try:
                return OpenAICompatibleLLM._parse_json(content)
            except (json.JSONDecodeError, TypeError) as exc:
                last_error = exc
                messages = list(base_messages)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous response was rejected by the JSON parser with: "
                            f"{exc}. Regenerate one concise JSON object from scratch. "
                            f"This is structured retry {attempt + 1}."
                        ),
                    }
                )
        raise RuntimeError(f"{component} did not return valid JSON: {last_error}")

    def usage_snapshot(self) -> Dict[str, Any]:
        return {
            "requests": self.usage["requests"],
            "prompt_tokens": self.usage["prompt_tokens"],
            "completion_tokens": self.usage["completion_tokens"],
            "by_component": {
                key: dict(value) for key, value in self.usage["by_component"].items()
            },
        }


def make_llm(config: Dict[str, Any]) -> StructuredLLMProtocol:
    provider = str(config.get("provider", "openai")).lower()
    if provider == "openai":
        return OpenAICompatibleLLM(config)
    if provider == "transformers":
        return LocalTransformersLLM(config)
    raise ValueError(f"Unknown LLM provider: {provider}")
