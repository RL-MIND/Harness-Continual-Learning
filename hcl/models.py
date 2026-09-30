from __future__ import annotations

import base64
import json
import os
import http.client
import mimetypes
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Semaphore
from time import perf_counter
from typing import Any

from .progress import progress, progress_heartbeat


_CALL_LOG_LOCK = Lock()


class BaseModel:
    def generate(self, prompt: str, *, state: dict[str, Any]) -> str:
        raise NotImplementedError


class RuleBasedSmokeModel:
    """Small deterministic model for checking CLI and pipeline wiring."""

    def generate(self, prompt: str, *, state: dict[str, Any]) -> str:
        if state.get("task_interface_phase") == "structure_task":
            return json.dumps(
                {
                    "evidence": {"items": [], "summary": "Visible evidence is supplied by the runtime."},
                    "recognizable_goal": {
                        "status": "under_specified",
                        "content": "The current evidence does not provide a complete executable goal.",
                        "supporting_evidence": [],
                    },
                    "recognizable_constraints": [],
                },
                ensure_ascii=False,
            )
        if state.get("router_phase") == "plan_workflow":
            return json.dumps(
                {
                    "use_memory": False,
                    "use_capabilities": False,
                    "memory_query": "",
                    "capability_request": [],
                    "rationale": "smoke model does not use external modules",
                },
                ensure_ascii=False,
            )
        if state.get("router_phase") == "select_memory":
            return json.dumps(
                {
                    "selected": [],
                    "discarded_summary": "No memory is required for this deterministic smoke check.",
                    "selection_confidence": "low",
                },
                ensure_ascii=False,
            )
        if state.get("optimizer_phase") == "generate_artifact_candidates":
            component = str(state.get("optimizer_component", ""))
            if component == "memory_selector":
                sections = [
                    "Select zero to {retrieval_top_k} memories by semantic relevance. Select nothing when none is useful.",
                    "Return one JSON object with selected, discarded_summary, and selection_confidence.",
                    "Each selected item contains memory_id, role, reason, and risk.",
                    "Current input:",
                    "{current_input}",
                    "Workflow:",
                    "{workflow_decision}",
                    "Memory candidates:",
                    "{memory_candidates}",
                ]
            elif component == "task_interface_structuring":
                sections = [
                    "Convert the raw visible interaction into exactly evidence, recognizable_goal, and recognizable_constraints.",
                    "Preserve visible evidence; use under_specified when no goal is supported.",
                    "Return one JSON object only.",
                    "Raw visible input:",
                    "{raw_visible_input}",
                ]
            elif component == "router_workflow":
                sections = [
                    "Choose a concise workflow from the minimal HCL interface.",
                    "Return one JSON object with workflow_name, steps, use_memory, use_capabilities, memory_query, capability_request, and rationale.",
                    "Model-visible input:",
                    "{model_visible_input}",
                ]
            else:
                sections = [
                    "Model-visible input:",
                    "{model_visible_input}",
                    "Workflow:",
                    "{workflow_decision}",
                    "Selected memory:",
                    "{selected_memory}",
                    "Selected skills:",
                    "{selected_skills}",
                    "Selected tools:",
                    "{selected_tools}",
                    "In raw memory, correct_feedback is authoritative; previous_attempt may be incorrect.",
                    "Reason internally and output only the final answer.",
                ]
            candidate_count = max(int(state.get("optimizer_candidate_count", 1) or 1), 1)
            return json.dumps(
                {
                    "candidates": [
                        {
                            "candidate_name": f"smoke_{component or 'prompt'}_{index + 1}",
                            "rationale": "Exercise the candidate parsing, screening, and validation path.",
                            "sections": sections,
                            "artifact_diff_summary": f"Compact smoke candidate {index + 1}.",
                        }
                        for index in range(candidate_count)
                    ]
                },
                ensure_ascii=False,
            )
        question = str(state.get("question") or state.get("task_context", {}).get("raw_question", ""))
        if "2+2" in question:
            return "4"
        return "unknown"

@dataclass
class LocalTransformersModelConfig:
    name: str
    path: str
    backend: str | None = None
    device_map: str = "auto"
    torch_dtype: str = "auto"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    quantization_fallback_to_full_precision: bool = False
    max_new_tokens: int = 128
    answer_max_new_tokens: int = 128
    temperature: float = 0.0
    enable_thinking: bool = True
    system_prompt: str = "You are a concise multimodal reasoning assistant. Answer with the required format."
    log_path: str | None = None
    log_prompt_chars: int = 4000
    attn_implementation: str | None = None
    local_files_only: bool = True


class LocalTransformersModel(BaseModel):
    """Generic local HuggingFace/transformers model wrapper.

    Config example:

    {
      "model": {
        "backend": "local_transformers",
        "name": "my-local-model",
        "path": "/path/to/checkpoint",
        "device_map": "auto",
        "torch_dtype": "auto",
        "max_new_tokens": 256,
        "temperature": 0.0,
        "enable_thinking": false
      }
    }

    The pipeline calls `generate(prompt, state=...)`. For multimodal tasks,
    `state["task_context"]["files"]` may include {"type": "image", "path": "..."}.
    """

    def __init__(
        self,
        *,
        name: str,
        path: str,
        backend: str | None = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        bnb_4bit_compute_dtype: str = "bfloat16",
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        quantization_fallback_to_full_precision: bool = False,
        max_new_tokens: int = 128,
        answer_max_new_tokens: int = 128,
        temperature: float = 0.0,
        enable_thinking: bool = True,
        system_prompt: str = "You are a concise multimodal reasoning assistant. Answer with the required format.",
        log_path: str | None = None,
        log_prompt_chars: int = 4000,
        attn_implementation: str | None = None,
        local_files_only: bool = True,
        **_: Any,
    ) -> None:
        self.config = LocalTransformersModelConfig(
            name=name,
            path=path,
            backend=backend,
            device_map=device_map,
            torch_dtype=torch_dtype,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            quantization_fallback_to_full_precision=quantization_fallback_to_full_precision,
            max_new_tokens=max_new_tokens,
            answer_max_new_tokens=answer_max_new_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
            system_prompt=system_prompt,
            log_path=log_path,
            log_prompt_chars=log_prompt_chars,
            attn_implementation=attn_implementation,
            local_files_only=local_files_only,
        )
        self._processor = None
        self._model = None

    def generate(self, prompt: str, *, state: dict[str, Any]) -> str:
        start_time = perf_counter()
        phase = _state_phase(state)
        task_id = state.get("task_id") or state.get("internal_sample_id") or "-"
        attempt = state.get("task_interface_attempt", state.get("router_attempt"))
        attempt_text = f" attempt={attempt}" if attempt is not None else ""
        progress(f"model request phase={phase} task_id={task_id}{attempt_text}")
        self._load()
        assert self._processor is not None
        assert self._model is not None
        image_path = self._image_path_from_state(state)
        image_exists = Path(image_path).exists() if image_path else False
        image = self._load_image_from_state(state)
        image_loaded = image is not None
        phase_max_new_tokens = (
            min(self.config.max_new_tokens, self.config.answer_max_new_tokens)
            if phase == "final_answer"
            else self.config.max_new_tokens
        )
        requested_max_new_tokens = state.get("max_new_tokens")
        if requested_max_new_tokens is not None:
            try:
                phase_max_new_tokens = min(phase_max_new_tokens, max(int(requested_max_new_tokens), 1))
            except (TypeError, ValueError):
                pass
        max_new_tokens = phase_max_new_tokens
        progress(
            f"model generate start phase={phase} task_id={task_id} "
            f"image={'yes' if image_loaded else 'no'} max_new_tokens={max_new_tokens}{attempt_text}"
        )
        user_content: str | list[dict[str, object]]
        if image is None:
            user_content = prompt
        else:
            user_content = [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_content},
        ]
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.config.enable_thinking,
        )
        if image is None:
            inputs = self._processor(text=[text], return_tensors="pt").to(self._model.device)
        else:
            inputs = self._processor(text=[text], images=[image], return_tensors="pt").to(self._model.device)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": self.config.temperature > 0,
        }
        if self.config.temperature > 0:
            generation_kwargs["temperature"] = self.config.temperature
        with progress_heartbeat(f"model generate phase={phase} task_id={task_id}"):
            outputs = self._model.generate(**inputs, **generation_kwargs)
        generated_ids = outputs[:, inputs.input_ids.shape[-1] :]
        output = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        duration = perf_counter() - start_time
        self._record_call_log(
            prompt,
            state,
            output,
            duration,
            image_path=image_path,
            image_exists=image_exists,
            image_loaded=image_loaded,
        )
        progress(
            f"model generate done phase={phase} task_id={task_id} "
            f"seconds={duration:.1f} output_chars={len(output)}{attempt_text}"
        )
        return output

    def _load(self) -> None:
        if self._model is not None:
            return
        load_start = perf_counter()
        quantization = "4bit" if self.config.load_in_4bit else "8bit" if self.config.load_in_8bit else "none"
        progress(
            f"model load start checkpoint={self.config.path} device_map={self.config.device_map} "
            f"quantization={quantization}"
        )
        checkpoint = Path(self.config.path)
        if self.config.local_files_only and not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(
                f"Local Qwen/transformers checkpoint is incomplete or missing config.json: {checkpoint}"
            )
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError("LocalTransformersModel requires torch, transformers, and Pillow for image inputs.") from exc
        if self.config.load_in_4bit and self.config.load_in_8bit:
            raise ValueError("Set only one of model.load_in_4bit or model.load_in_8bit.")
        dtype = self.config.torch_dtype
        torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
        load_kwargs = self._model_load_kwargs(torch=torch, torch_dtype=torch_dtype, use_quantization=True)
        self._processor = AutoProcessor.from_pretrained(
            self.config.path,
            trust_remote_code=True,
            local_files_only=self.config.local_files_only,
        )
        try:
            self._load_model_with_kwargs(load_kwargs)
        except Exception as exc:
            if not (
                (self.config.load_in_4bit or self.config.load_in_8bit)
                and self.config.quantization_fallback_to_full_precision
            ):
                raise
            progress(
                f"model quantized load failed; retrying full precision checkpoint={self.config.path} "
                f"error={type(exc).__name__}: {exc}"
            )
            self._model = None
            load_kwargs = self._model_load_kwargs(torch=torch, torch_dtype=torch_dtype, use_quantization=False)
            self._load_model_with_kwargs(load_kwargs)
        if self._model is None:
            raise RuntimeError("No compatible transformers AutoModel class found for this checkpoint.")
        self._model.eval()
        progress(f"model load done seconds={perf_counter() - load_start:.1f}")

    def _model_load_kwargs(self, *, torch: Any, torch_dtype: Any, use_quantization: bool) -> dict[str, Any]:
        load_kwargs: dict[str, Any] = {
            "device_map": self.config.device_map,
            "trust_remote_code": True,
            "local_files_only": self.config.local_files_only,
        }
        if self.config.attn_implementation:
            load_kwargs["attn_implementation"] = self.config.attn_implementation
        if use_quantization and (self.config.load_in_4bit or self.config.load_in_8bit):
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError(
                    "4bit/8bit local loading requires transformers with BitsAndBytesConfig and bitsandbytes installed."
                ) from exc
            compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            )
            return load_kwargs
        # Transformers 5.x renamed torch_dtype to dtype; current Qwen3.5
        # checkpoints use that code path. Older supported releases accept the
        # same dtype argument through from_pretrained.
        load_kwargs["dtype"] = torch_dtype
        return load_kwargs

    def _load_model_with_kwargs(self, load_kwargs: dict[str, Any]) -> None:
        last_config_error: ValueError | None = None
        for model_cls in self._resolve_model_classes():
            try:
                self._model = model_cls.from_pretrained(self.config.path, **load_kwargs)
                break
            except ValueError as exc:
                if "Unrecognized configuration class" not in str(exc):
                    raise
                last_config_error = exc
        if self._model is None:
            raise RuntimeError("No compatible transformers AutoModel class found for this checkpoint.") from last_config_error

    @staticmethod
    def _resolve_model_classes() -> list[Any]:
        import transformers

        classes: list[Any] = []
        for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
            model_cls = getattr(transformers, name, None)
            if model_cls is not None:
                classes.append(model_cls)
        if classes:
            return classes
        raise RuntimeError("No compatible transformers AutoModel class found.")

    def _load_image_from_state(self, state: dict[str, Any]) -> Any | None:
        image_path = self._image_path_from_state(state)
        if not image_path:
            return None
        path = Path(image_path)
        if not path.exists():
            return None
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("LocalTransformersModel image loading requires Pillow.") from exc
        return Image.open(path).convert("RGB")

    @staticmethod
    def _image_path_from_state(state: dict[str, Any]) -> str | None:
        task_context = state.get("task_context")
        if isinstance(task_context, dict):
            files = task_context.get("files")
            if isinstance(files, list):
                for item in files:
                    if isinstance(item, dict) and str(item.get("type", "")).lower() == "image" and item.get("path"):
                        return str(item["path"])
        image = state.get("image") or state.get("image_path")
        return str(image) if image else None

    def _record_call_log(
        self,
        prompt: str,
        state: dict[str, Any],
        output: str,
        duration_seconds: float,
        *,
        image_path: str | None,
        image_exists: bool,
        image_loaded: bool,
    ) -> None:
        log_path = self.config.log_path or state.get("local_model_debug_path") or state.get("qwen_debug_path")
        if not log_path:
            return
        path = Path(str(log_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        phase = _state_phase(state)
        prompt_preview = prompt[: max(int(self.config.log_prompt_chars), 0)]
        with _CALL_LOG_LOCK, path.open("a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"phase: {phase}\n")
            f.write(f"internal_sample_id: {state.get('task_id') or state.get('internal_sample_id')}\n")
            f.write(f"duration_seconds: {duration_seconds:.4f}\n")
            f.write(f"image_path: {image_path or ''}\n")
            f.write(f"image_exists: {image_exists}\n")
            f.write(f"image_loaded: {image_loaded}\n")
            f.write("state_keys: " + ", ".join(sorted(str(key) for key in state.keys())) + "\n")
            f.write("\n[PROMPT]\n")
            f.write(prompt_preview + "\n")
            if len(prompt) > len(prompt_preview):
                f.write("... [prompt truncated]\n")
            f.write("\n[OUTPUT]\n")
            f.write(output + "\n")


@dataclass
class ChatCompletionsAPIModelConfig:
    name: str = "model"
    backend: str | None = "chat_completions"
    base_url: str = "http://127.0.0.1:8000"
    api_key: str | None = None
    api_key_env: str = "MODEL_API_KEY"
    max_new_tokens: int = 512
    answer_max_new_tokens: int = 512
    temperature: float = 0.0
    repetition_penalty: float = 1.0
    thinking: dict[str, str] | None = None
    reasoning_effort: str | None = None
    system_prompt: str | None = "You are a concise reasoning assistant. Answer with the required format."
    timeout_seconds: float = 120.0
    max_retries: int = 2
    log_path: str | None = None
    log_prompt_chars: int = 4000
    cleanup_output: bool = False
    enable_image_input: bool = True
    chat_template_kwargs: dict[str, Any] | None = None
    thinking_phases: list[str] | None = None
    json_object_for_structured_phases: bool = False
    max_concurrent_image_requests: int = 1


class ChatCompletionsAPIModel(BaseModel):
    """OpenAI-compatible chat-completions wrapper.

    The API key is read from `api_key` or `api_key_env`. The endpoint is
    configurable because many providers and local servers expose the same
    request shape behind a different base URL.
    """

    def __init__(
        self,
        *,
        name: str = "model",
        backend: str | None = "chat_completions",
        base_url: str = "http://127.0.0.1:8000",
        api_key: str | None = None,
        api_key_env: str = "MODEL_API_KEY",
        max_new_tokens: int = 512,
        answer_max_new_tokens: int = 512,
        temperature: float = 0.0,
        repetition_penalty: float = 1.0,
        thinking: dict[str, str] | None = None,
        reasoning_effort: str | None = None,
        system_prompt: str | None = "You are a concise reasoning assistant. Answer with the required format.",
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        log_path: str | None = None,
        log_prompt_chars: int = 4000,
        cleanup_output: bool = False,
        enable_image_input: bool = True,
        chat_template_kwargs: dict[str, Any] | None = None,
        thinking_phases: list[str] | None = None,
        json_object_for_structured_phases: bool = False,
        max_concurrent_image_requests: int = 1,
        **_: Any,
    ) -> None:
        if api_key is None and isinstance(base_url, str) and base_url.startswith("sk-"):
            api_key = base_url
            base_url = "http://127.0.0.1:8000"
        if api_key is None and isinstance(api_key_env, str) and api_key_env.startswith("sk-"):
            api_key = api_key_env
            api_key_env = "MODEL_API_KEY"
        if not base_url:
            base_url = "http://127.0.0.1:8000"
        self.config = ChatCompletionsAPIModelConfig(
            name=name,
            backend=backend,
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            api_key_env=api_key_env,
            max_new_tokens=max_new_tokens,
            answer_max_new_tokens=answer_max_new_tokens,
            temperature=temperature,
            repetition_penalty=max(float(repetition_penalty), 0.0),
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            system_prompt=system_prompt,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            log_path=log_path,
            log_prompt_chars=log_prompt_chars,
            cleanup_output=cleanup_output,
            enable_image_input=enable_image_input,
            chat_template_kwargs=dict(chat_template_kwargs) if chat_template_kwargs is not None else None,
            thinking_phases=list(thinking_phases) if thinking_phases is not None else None,
            json_object_for_structured_phases=bool(json_object_for_structured_phases),
            max_concurrent_image_requests=max(int(max_concurrent_image_requests), 1),
        )
        self._image_request_gate = Semaphore(self.config.max_concurrent_image_requests)

    def generate(self, prompt: str, *, state: dict[str, Any]) -> str:
        start_time = perf_counter()
        phase = _state_phase(state)
        task_id = state.get("task_id") or state.get("internal_sample_id") or "-"
        attempt = state.get("task_interface_attempt", state.get("router_attempt"))
        attempt_text = f" attempt={attempt}" if attempt is not None else ""
        max_new_tokens = self._max_tokens_for_phase(phase, state)
        progress(
            f"chat api request start phase={phase} task_id={task_id} "
            f"model={self.config.name} max_new_tokens={max_new_tokens}{attempt_text}"
        )
        response_payload = self._chat_completion(prompt, state=state, max_new_tokens=max_new_tokens)
        output = self._extract_text(response_payload).strip()
        if self.config.cleanup_output:
            output = self._cleanup_output(output, phase=phase, prompt=prompt)
        duration = perf_counter() - start_time
        self._record_call_log(prompt, state, output, duration, response_payload)
        progress(
            f"chat api request done phase={phase} task_id={task_id} "
            f"seconds={duration:.1f} output_chars={len(output)}{attempt_text}"
        )
        return output

    def generate_with_tools(
        self,
        prompt: str,
        *,
        state: dict[str, Any],
        tools: list[dict[str, Any]],
        tool_executor: Any,
        max_tool_rounds: int = 2,
    ) -> str:
        start_time = perf_counter()
        phase = _state_phase(state)
        task_id = state.get("task_id") or state.get("internal_sample_id") or "-"
        max_new_tokens = self._max_tokens_for_phase(phase, state)
        progress(
            f"chat api tool request start phase={phase} task_id={task_id} "
            f"model={self.config.name} tools={len(tools)} max_new_tokens={max_new_tokens}"
        )
        messages: list[dict[str, Any]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": self._user_content(prompt, state)})
        response_payload: dict[str, Any] | None = None
        tool_rounds = max(int(max_tool_rounds), 0)
        for round_index in range(tool_rounds + 1):
            response_payload = self._chat_completion_messages(
                messages,
                max_new_tokens=max_new_tokens,
                tools=tools if tools else None,
                phase=phase,
            )
            message = self._extract_message(response_payload)
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list) or not tool_calls or round_index >= tool_rounds:
                output = self._message_text(message).strip()
                if self.config.cleanup_output:
                    output = self._cleanup_output(output, phase=phase, prompt=prompt)
                duration = perf_counter() - start_time
                self._record_call_log(prompt, state, output, duration, response_payload)
                progress(
                    f"chat api tool request done phase={phase} task_id={task_id} "
                    f"seconds={duration:.1f} output_chars={len(output)} tool_rounds={round_index}"
                )
                return output
            messages.append(_assistant_tool_call_message(message))
            for tool_call in tool_calls:
                tool_result = self._execute_tool_call(tool_call, tool_executor)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(tool_call.get("id", "")),
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )
        if response_payload is None:
            raise RuntimeError("Chat-completions tool request did not produce a response.")
        output = self._extract_text(response_payload).strip()
        if self.config.cleanup_output:
            output = self._cleanup_output(output, phase=phase, prompt=prompt)
        duration = perf_counter() - start_time
        self._record_call_log(prompt, state, output, duration, response_payload)
        return output

    def _max_tokens_for_phase(self, phase: str, state: dict[str, Any]) -> int:
        max_tokens = (
            min(self.config.max_new_tokens, self.config.answer_max_new_tokens)
            if phase == "final_answer"
            else self.config.max_new_tokens
        )
        requested_max_new_tokens = state.get("max_new_tokens")
        if requested_max_new_tokens is not None:
            try:
                max_tokens = min(max_tokens, max(int(requested_max_new_tokens), 1))
            except (TypeError, ValueError):
                pass
        return max(int(max_tokens), 1)

    def _chat_completion(
        self,
        prompt: str,
        *,
        state: dict[str, Any],
        max_new_tokens: int,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": self._user_content(prompt, state)})
        return self._chat_completion_messages(
            messages,
            max_new_tokens=max_new_tokens,
            phase=_state_phase(state),
        )

    def _chat_completion_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        phase: str | None = None,
    ) -> dict[str, Any]:
        api_key = self.config.api_key or os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Chat-completions API key is missing. Set {self.config.api_key_env} "
                "or provide model.api_key in the config."
            )
        url = f"{self.config.base_url}/v1/chat/completions"
        payload = {
            "model": self.config.name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": max_new_tokens,
        }
        if self.config.repetition_penalty != 1.0:
            payload["repetition_penalty"] = self.config.repetition_penalty
        if tools:
            payload["tools"] = tools
        if self.config.thinking is not None:
            payload["thinking"] = self.config.thinking
        if self.config.reasoning_effort:
            payload["reasoning_effort"] = self.config.reasoning_effort
        if self.config.chat_template_kwargs is not None:
            chat_template_kwargs = dict(self.config.chat_template_kwargs)
            if self.config.thinking_phases is not None:
                chat_template_kwargs["enable_thinking"] = bool(
                    phase and phase in self.config.thinking_phases
                )
            payload["chat_template_kwargs"] = chat_template_kwargs
        if self.config.json_object_for_structured_phases and phase and phase != "final_answer":
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        attempts = max(int(self.config.max_retries), 0) + 1
        has_image = _messages_contain_image(messages)
        for attempt in range(attempts):
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                if has_image:
                    with self._image_request_gate:
                        with self._open_request(request) as response:
                            raw = response.read().decode("utf-8")
                else:
                    with self._open_request(request) as response:
                        raw = response.read().decode("utf-8")
                decoded = json.loads(raw)
                if not isinstance(decoded, dict):
                    raise RuntimeError("Chat-completions API response was not a JSON object.")
                return decoded
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP {exc.code} from chat-completions API: {detail[:1000]}")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                TimeoutError,
                json.JSONDecodeError,
                RuntimeError,
            ) as exc:
                last_error = exc
            if attempt + 1 < attempts:
                progress(f"chat api retry attempt={attempt + 2}/{attempts} reason={type(last_error).__name__}")
        raise RuntimeError(f"Chat-completions API request failed after {attempts} attempt(s): {last_error}")

    def _open_request(self, request: urllib.request.Request) -> Any:
        hostname = urllib.parse.urlparse(request.full_url).hostname
        timeout = float(self.config.timeout_seconds)
        if hostname in {"127.0.0.1", "localhost", "::1"}:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            return opener.open(request, timeout=timeout)
        return urllib.request.urlopen(request, timeout=timeout)

    def _user_content(self, prompt: str, state: dict[str, Any]) -> str | list[dict[str, Any]]:
        image_path = LocalTransformersModel._image_path_from_state(state)
        if not self.config.enable_image_input or not image_path:
            return prompt
        path = Path(image_path)
        if not path.is_file():
            return prompt
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        ]

    @staticmethod
    def _extract_message(payload: dict[str, Any]) -> dict[str, Any]:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"Chat-completions API response missing choices: {payload}")
        first = choices[0]
        if not isinstance(first, dict):
            raise RuntimeError(f"Chat-completions API response choice is invalid: {first}")
        message = first.get("message")
        if not isinstance(message, dict):
            raise RuntimeError(f"Chat-completions API response choice has no message object: {first}")
        return message

    @staticmethod
    def _message_text(message: dict[str, Any]) -> str:
        content = message.get("content")
        if content is None:
            return ""
        return str(content)

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"Chat-completions API response missing choices: {payload}")
        first = choices[0]
        if not isinstance(first, dict):
            raise RuntimeError(f"Chat-completions API response choice is invalid: {first}")
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if content is not None:
                return str(content)
        text = first.get("text")
        if text is not None:
            return str(text)
        raise RuntimeError(f"Chat-completions API response choice has no text content: {first}")

    @classmethod
    def _cleanup_output(cls, output: str, *, phase: str, prompt: str) -> str:
        text = cls._strip_thinking_and_fences(output)
        if phase != "final_answer":
            structured = cls._extract_first_json_value(text)
            return structured if structured is not None else text
        lowered_prompt = prompt.lower()
        answer = cls._strip_answer_prefixes(text)
        structured = cls._extract_first_json_value(answer)
        if structured is not None:
            return structured
        if "return only the final numeric answer" in lowered_prompt or "final numeric answer" in lowered_prompt:
            numeric = cls._extract_numeric_answer(answer)
            if numeric:
                return numeric
        if "return exactly one of" in lowered_prompt:
            labels = cls._labels_from_prompt(prompt)
            if labels:
                found = cls._extract_label_answer(answer, labels)
                if found:
                    return found
        if "yes" in lowered_prompt and "no" in lowered_prompt and "return" in lowered_prompt:
            found = cls._extract_label_answer(answer, ["yes", "no"])
            if found:
                return found
        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        if len(lines) > 1:
            answer = lines[-1]
        return cls._strip_answer_prefixes(answer).strip()

    @staticmethod
    def _strip_thinking_and_fences(output: str) -> str:
        text = output.strip()
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[-1].strip()
        fence_match = re.fullmatch(r"```(?:json|JSON)?\s*(.*?)\s*```", text, flags=re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()
        return text

    @staticmethod
    def _strip_answer_prefixes(text: str) -> str:
        answer = text.strip()
        answer = re.sub(r"^\s*(?:\*\*)?\s*final\s+answer\s*(?:\*\*)?\s*[:：-]\s*", "", answer, flags=re.I)
        answer = re.sub(r"^\s*(?:\*\*)?\s*answer\s*(?:\*\*)?\s*[:：-]\s*", "", answer, flags=re.I)
        return answer.strip()

    @staticmethod
    def _extract_first_json_value(text: str) -> str | None:
        start_positions = [pos for pos in (text.find("{"), text.find("[")) if pos >= 0]
        if not start_positions:
            return None
        start = min(start_positions)
        opener = text[start]
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1].strip()
                    try:
                        json.loads(candidate)
                    except json.JSONDecodeError:
                        return None
                    return candidate
        return None

    @staticmethod
    def _extract_numeric_answer(text: str) -> str | None:
        final_match = re.search(r"(?:final\s+answer\s+is|answer\s+is)\s*([-+]?\d[\d,]*(?:\.\d+)?)", text, flags=re.I)
        if final_match:
            return final_match.group(1).replace(",", "")
        numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
        if numbers:
            return numbers[-1].replace(",", "")
        return None

    @staticmethod
    def _labels_from_prompt(prompt: str) -> list[str]:
        match = re.search(r"return exactly one of:\s*([^\n.]+)", prompt, flags=re.I)
        if not match:
            return []
        return [part.strip(" `\"'") for part in re.split(r",|\bor\b", match.group(1)) if part.strip()]

    @staticmethod
    def _extract_label_answer(text: str, labels: list[str]) -> str | None:
        lowered = text.lower()
        result: tuple[int, str] | None = None
        for label in labels:
            pattern = r"\b" + re.escape(label.lower()) + r"\b"
            for match in re.finditer(pattern, lowered):
                result = (match.start(), label)
        return result[1] if result else None

    def _execute_tool_call(self, tool_call: dict[str, Any], tool_executor: Any) -> dict[str, Any]:
        function = tool_call.get("function")
        if not isinstance(function, dict):
            return {"ok": False, "error": "tool_call_missing_function"}
        name = str(function.get("name", ""))
        arguments = function.get("arguments", "{}")
        try:
            result = tool_executor(name, arguments)
        except Exception as exc:  # Tool errors should be returned to the model, not crash the run.
            return {"ok": False, "tool": name, "error": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "tool": name, "result": result}

    def _record_call_log(
        self,
        prompt: str,
        state: dict[str, Any],
        output: str,
        duration_seconds: float,
        response_payload: dict[str, Any],
    ) -> None:
        log_path = self.config.log_path or state.get("chat_api_debug_path") or state.get("ds_debug_path")
        if not log_path:
            return
        path = Path(str(log_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        phase = _state_phase(state)
        prompt_preview = prompt[: max(int(self.config.log_prompt_chars), 0)]
        usage = response_payload.get("usage", {})
        message: dict[str, Any] = {}
        choices = response_payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            candidate = choices[0].get("message")
            if isinstance(candidate, dict):
                message = candidate
        raw_content = message.get("content")
        reasoning_content = message.get("reasoning_content")
        if reasoning_content is None:
            reasoning_content = message.get("reasoning")
        with _CALL_LOG_LOCK, path.open("a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"phase: {phase}\n")
            f.write(f"internal_sample_id: {state.get('task_id') or state.get('internal_sample_id')}\n")
            f.write(f"duration_seconds: {duration_seconds:.4f}\n")
            f.write(f"model: {self.config.name}\n")
            f.write(f"usage: {json.dumps(usage, ensure_ascii=False)}\n")
            f.write("state_keys: " + ", ".join(sorted(str(key) for key in state.keys())) + "\n")
            f.write("\n[SYSTEM]\n")
            f.write((self.config.system_prompt or "") + "\n")
            f.write("\n[PROMPT]\n")
            f.write(prompt_preview + "\n")
            if len(prompt) > len(prompt_preview):
                f.write("... [prompt truncated]\n")
            f.write("\n[REASONING_CONTENT]\n")
            f.write((str(reasoning_content) if reasoning_content is not None else "") + "\n")
            f.write("\n[RAW_OUTPUT]\n")
            f.write((str(raw_content) if raw_content is not None else "") + "\n")
            f.write("\n[CLEANED_OUTPUT]\n")
            f.write(output + "\n")


def _state_phase(state: dict[str, Any]) -> str:
    if state.get("task_interface_phase"):
        return str(state["task_interface_phase"])
    if state.get("router_phase"):
        return str(state["router_phase"])
    if state.get("optimizer_phase"):
        return str(state["optimizer_phase"])
    if state.get("memory_phase"):
        return str(state["memory_phase"])
    return "final_answer"


def _messages_contain_image(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"image_url", "image"}:
                return True
    return False


def _assistant_tool_call_message(message: dict[str, Any]) -> dict[str, Any]:
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content") or "",
    }
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        assistant_message["tool_calls"] = tool_calls
    return assistant_message


QwenModelConfig = LocalTransformersModelConfig
QwenModel = LocalTransformersModel
DeepSeekAPIModelConfig = ChatCompletionsAPIModelConfig
DeepSeekAPIModel = ChatCompletionsAPIModel
