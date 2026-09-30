import json

import requests
from langchain.schema import AIMessage

from odyssey.utils import config


class ModelType:
    """Legacy model aliases kept for existing experiment code.

    The actual model sent to the API is configured by `openai_model`.
    """

    LLAMA2_70B = 'llama2_70b'
    LLAMA3_8B_V3 = 'llama3_8b_v3'
    LLAMA3_8B = 'llama3_8b'
    LLAMA3_70B_V1 = 'llama3_70b_v1'
    QWEN2_72B = 'qwen2-72b'
    QWEN2_7B = 'qwen2-7b'
    BAICHUAN2_7B = 'baichuan2-7b'


def _get_bool_config(key, default=False):
    value = config.get(key)
    if value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _get_int_config(key):
    value = config.get(key)
    if value == "":
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _get_request_timeout():
    timeout = config.get("llm_request_timeout")
    if not timeout:
        return None
    try:
        return float(timeout)
    except (TypeError, ValueError):
        return None


def _build_extra_body():
    """Build optional OpenAI-compatible vendor extensions from config.json."""
    extra_body = config.get("llm_extra_body")
    if not isinstance(extra_body, dict):
        extra_body = {}
    else:
        extra_body = dict(extra_body)

    thinking_enabled = _get_bool_config("llm_thinking_enabled", default=False)
    thinking_style = config.get("llm_thinking_style") or "deepseek"
    thinking_effort = config.get("llm_thinking_effort") or ""
    thinking_budget = _get_int_config("llm_thinking_budget_tokens")

    if thinking_style == "none":
        return extra_body

    if thinking_style == "reasoning_effort":
        if thinking_enabled and thinking_effort:
            extra_body["reasoning_effort"] = thinking_effort
        return extra_body

    if thinking_style == "qwen":
        extra_body["enable_thinking"] = thinking_enabled
        if thinking_enabled and thinking_budget:
            extra_body["thinking_budget"] = thinking_budget
        return extra_body

    thinking = {"type": "enabled" if thinking_enabled else "disabled"}
    if thinking_enabled and thinking_effort:
        thinking["effort"] = thinking_effort
    if thinking_enabled and thinking_budget:
        thinking["budget_tokens"] = thinking_budget
    extra_body["thinking"] = thinking
    return extra_body


def call_with_messages(msgs, model_name=ModelType.LLAMA3_8B_V3):
    base_url = config.get("openai_base_url")
    if not base_url:
        raise RuntimeError("`openai_base_url` must be configured.")

    base_url = base_url.rstrip("/")
    model = config.get("openai_model") or model_name
    api_key = config.get("openai_api_key")
    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": msgs[0].content},
            {"role": "user", "content": msgs[1].content},
        ],
    }
    payload.update(_build_extra_body())
    result = requests.post(
        url,
        json=payload,
        headers=headers,
        timeout=_get_request_timeout(),
    )
    result.raise_for_status()
    json_result = result.json()
    choices = json_result.get("choices") or []
    if not choices:
        preview = json.dumps(json_result, ensure_ascii=False)[:1000]
        raise RuntimeError(f"OpenAI-compatible API returned no choices: {preview}")

    choice = choices[0] or {}
    message = choice.get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    if content is None:
        content = choice.get("text")
    if content is None:
        preview = json.dumps(json_result, ensure_ascii=False)[:1000]
        raise RuntimeError(f"OpenAI-compatible API returned no text content: {preview}")
    return AIMessage(content=content)
