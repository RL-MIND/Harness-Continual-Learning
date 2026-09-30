from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import local
from typing import Any, Protocol, TYPE_CHECKING

from .cache_store import JsonObjectCache as _BaseJsonObjectCache
from .cache_store import TextValueCache as _BaseTextValueCache
from .capability import BasicCapabilityRegistry, CapabilityExecutionContext, CapabilityExecutionError
from .json_utils import json_clone as _json_clone
from .json_utils import stable_json_hash as _stable_json_hash
from .progress import progress
from .prompt_templates import PromptTemplate
from .skills import SkillRegistry
from .task_interface import (
    TaskInterface,
    TaskInterfaceConfig,
    assert_no_label_leakage,
    sanitize_label_leakage_for_visible_prompt,
)

if TYPE_CHECKING:
    from .memory import ExperienceMemory


ROUTER_DIR = Path(__file__).resolve().parent / "router"
DEFAULT_WORKFLOW_TEMPLATE = ROUTER_DIR / "workflow_prompt_default.json"
DEFAULT_CONTEXT_TEMPLATE = ROUTER_DIR / "context_prompt_default.json"
DEFAULT_MEMORY_SELECTOR_TEMPLATE = ROUTER_DIR / "memory_selector_prompt_default.json"
DEFAULT_SKILL_SELECTOR_TEMPLATE = ROUTER_DIR / "skill_selector_prompt_default.json"
DEFAULT_TOOL_SELECTOR_TEMPLATE = ROUTER_DIR / "tool_selector_prompt_default.json"
DEFAULT_TOOL_ARGUMENT_TEMPLATE = ROUTER_DIR / "tool_argument_prompt_default.json"
DEFAULT_CAPABILITY_WORKFLOW_TEMPLATE = ROUTER_DIR / "workflow_prompt_capability_aware.json"
DEFAULT_CAPABILITY_CONTEXT_TEMPLATE = ROUTER_DIR / "context_prompt_capability_aware.json"
INVALID_MEMORY_SELECTOR_FALLBACK_SUMMARY = "Memory selector output invalid; no memory selected."


class Model(Protocol):
    def generate(self, prompt: str, *, state: dict[str, Any]) -> str:
        ...


@dataclass
class RouterConfig:
    route_name: str = "direct"
    strip_output: bool = True
    use_llm_workflow_selector: bool = True
    use_memory_selector: bool = True
    memory_always_on: bool = True
    use_skill_selector: bool = True
    use_tool_selector: bool = True
    tool_auto_select: bool = True
    use_chat_tool_calls: bool = True
    tool_max_rounds: int = 2
    tool_strict_schema: bool = False
    skill_dir: str | None = None
    tool_js_dir: str | None = None
    tool_js_timeout_seconds: float = 8.0
    workflow_template_path: str = str(DEFAULT_WORKFLOW_TEMPLATE)
    context_template_path: str = str(DEFAULT_CONTEXT_TEMPLATE)
    memory_selector_template_path: str = str(DEFAULT_MEMORY_SELECTOR_TEMPLATE)
    skill_selector_template_path: str = str(DEFAULT_SKILL_SELECTOR_TEMPLATE)
    tool_selector_template_path: str = str(DEFAULT_TOOL_SELECTOR_TEMPLATE)
    workflow_selector_cache_path: str | None = None
    workflow_selector_max_new_tokens: int = 768
    memory_selector_max_new_tokens: int = 256
    memory_selector_invalid_output_retries: int = 1
    memory_selector_retry_max_new_tokens: int = 256
    memory_selector_fallback_to_empty_on_invalid: bool = False
    memory_selector_cache_path: str | None = None
    skill_selector_cache_path: str | None = None
    skill_selector_max_new_tokens: int = 512
    skill_selector_invalid_output_retries: int = 1
    skill_selector_retry_max_new_tokens: int = 512
    skill_selector_fallback_to_empty_on_invalid: bool = False
    tool_selector_cache_path: str | None = None
    tool_selector_max_new_tokens: int = 512
    tool_selector_invalid_output_retries: int = 1
    tool_selector_retry_max_new_tokens: int = 512
    tool_selector_fallback_to_empty_on_invalid: bool = False
    tool_argument_template_path: str = str(DEFAULT_TOOL_ARGUMENT_TEMPLATE)
    tool_argument_cache_path: str | None = None
    tool_argument_max_new_tokens: int = 256
    tool_argument_invalid_output_retries: int = 1
    tool_argument_retry_max_new_tokens: int = 256
    final_generation_cache_path: str | None = None
    capability_enabled: bool = False
    capability_execution_mode: str = "disabled"
    capability_workflow_template_path: str = str(DEFAULT_CAPABILITY_WORKFLOW_TEMPLATE)
    capability_context_template_path: str = str(DEFAULT_CAPABILITY_CONTEXT_TEMPLATE)
    capability_service_url: str | None = None
    capability_timeout_seconds: float = 15.0
    capability_allowed_corpora: tuple[str, ...] = ()
    capability_default_corpus_id: str = "task_corpus"
    capability_default_top_k: int = 5
    capability_continue_on_error: bool = True
    capability_force_semantic_search: bool = False
    semantic_search_enabled: bool = True
    cross_modal_match_enabled: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class Router:
    def __init__(
        self,
        model: Model,
        config: RouterConfig | None = None,
        *,
        memory: ExperienceMemory | None = None,
        selection_model: Model | None = None,
        workflow_template_content: dict[str, Any] | None = None,
        context_template_content: dict[str, Any] | None = None,
        memory_selector_template_content: dict[str, Any] | None = None,
        skill_selector_template_content: dict[str, Any] | None = None,
        tool_selector_template_content: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.selection_model = selection_model or model
        self.config = config or RouterConfig()
        if self.config.capability_execution_mode not in {"disabled", "router_managed", "llm_tool_call"}:
            raise ValueError(
                "capability_execution_mode must be disabled, router_managed, or llm_tool_call"
            )
        self.memory = memory
        self.skills = SkillRegistry(skill_dir=self.config.skill_dir)
        self.tools = BasicCapabilityRegistry(
            js_tool_dir=self.config.tool_js_dir,
            strict_tools=self.config.tool_strict_schema,
            js_tool_timeout_seconds=self.config.tool_js_timeout_seconds,
            external_capabilities_enabled=self.config.capability_enabled,
            external_service_url=self.config.capability_service_url,
            external_timeout_seconds=self.config.capability_timeout_seconds,
            semantic_search_enabled=self.config.semantic_search_enabled,
            cross_modal_match_enabled=self.config.cross_modal_match_enabled,
        )
        if workflow_template_content is None:
            self.workflow_template = PromptTemplate.from_path(self.config.workflow_template_path)
        else:
            self.workflow_template = PromptTemplate.from_dict(
                workflow_template_content,
                template_id_fallback="candidate_router_workflow",
            )
        if context_template_content is None:
            self.context_template = PromptTemplate.from_path(self.config.context_template_path)
        else:
            self.context_template = PromptTemplate.from_dict(
                context_template_content,
                template_id_fallback="candidate_router_context",
            )
        self.capability_workflow_template = None
        self.capability_context_template = None
        if self.config.capability_enabled:
            self.capability_workflow_template = (
                PromptTemplate.from_dict(
                    workflow_template_content,
                    template_id_fallback="candidate_capability_router_workflow",
                )
                if workflow_template_content is not None
                else PromptTemplate.from_path(self.config.capability_workflow_template_path)
            )
            self.capability_context_template = (
                PromptTemplate.from_dict(
                    context_template_content,
                    template_id_fallback="candidate_capability_router_context",
                )
                if context_template_content is not None
                else PromptTemplate.from_path(self.config.capability_context_template_path)
            )
        if memory_selector_template_content is None:
            self.memory_selector_template = PromptTemplate.from_path(self.config.memory_selector_template_path)
        else:
            self.memory_selector_template = PromptTemplate.from_dict(
                memory_selector_template_content,
                template_id_fallback="candidate_memory_selector",
            )
        if skill_selector_template_content is None:
            self.skill_selector_template = PromptTemplate.from_path(self.config.skill_selector_template_path)
        else:
            self.skill_selector_template = PromptTemplate.from_dict(
                skill_selector_template_content,
                template_id_fallback="candidate_skill_selector",
            )
        if tool_selector_template_content is None:
            self.tool_selector_template = PromptTemplate.from_path(self.config.tool_selector_template_path)
        else:
            self.tool_selector_template = PromptTemplate.from_dict(
                tool_selector_template_content,
                template_id_fallback="candidate_tool_selector",
            )
        self.tool_argument_template = PromptTemplate.from_path(self.config.tool_argument_template_path)
        self._workflow_selector_cache = _JsonObjectCache(
            self.config.workflow_selector_cache_path,
            value_field="workflow_decision",
            schema_error=_workflow_decision_schema_error,
        )
        self._memory_selector_cache = _MemorySelectorCache(self.config.memory_selector_cache_path)
        self._skill_selector_cache = _JsonObjectCache(
            self.config.skill_selector_cache_path,
            value_field="skill_selection",
            schema_error=lambda value: _named_selection_schema_error(value, name_field="skill_name"),
        )
        self._tool_selector_cache = _JsonObjectCache(
            self.config.tool_selector_cache_path,
            value_field="tool_selection",
            schema_error=lambda value: _named_selection_schema_error(value, name_field="tool_name"),
        )
        self._tool_argument_cache = _JsonObjectCache(
            self.config.tool_argument_cache_path,
            value_field="tool_argument_decision",
            schema_error=_tool_argument_decision_schema_error,
        )
        self._final_generation_cache = _FinalGenerationCache(self.config.final_generation_cache_path)
        self._workflow_trace_local = local()
        self._set_workflow_selection_trace({
            "cache": "disabled",
            "cache_key": "",
            "raw_output": "",
            "reason": "not_run",
        })

    def _set_workflow_selection_trace(self, trace: dict[str, object]) -> None:
        self._workflow_trace_local.value = dict(trace)

    def _workflow_selection_trace(self) -> dict[str, object]:
        value = getattr(self._workflow_trace_local, "value", None)
        if isinstance(value, dict):
            return dict(value)
        return {"cache": "disabled", "cache_key": "", "raw_output": "", "reason": "not_run"}

    def run(
        self,
        taskinterfacechunk: dict[str, Any],
        task_interface_config: TaskInterfaceConfig,
        *,
        allow_final_generation_cache: bool = True,
    ) -> dict[str, Any]:
        selected_skills = self.select_skills(taskinterfacechunk)
        workflow_decision = self.plan_workflow(taskinterfacechunk, selected_skills=selected_skills)
        workflow_selection_trace = self._workflow_selection_trace()
        selected_memory, memory_selection = self.select_memory(taskinterfacechunk, workflow_decision)
        selected_tools = self.select_tools(taskinterfacechunk, workflow_decision, selected_skills=selected_skills)
        capability_execution_context = self._capability_execution_context(taskinterfacechunk)
        capability_results = self.execute_selected_capabilities(
            taskinterfacechunk,
            workflow_decision,
            selected_tools,
            execution_context=capability_execution_context,
        )
        context = self.build_context(
            taskinterfacechunk,
            workflow_decision=workflow_decision,
            selected_memory=selected_memory,
            selected_skills=selected_skills,
            selected_tools=selected_tools,
            capability_results=capability_results,
        )
        assert_no_label_leakage(context, phase="final_answer")
        internal_context = taskinterfacechunk.get("task_context", {})
        final_state = {
            "question": TaskInterface.visible_question(taskinterfacechunk),
            "task_context": {
                "files": internal_context.get("files", []) if isinstance(internal_context, dict) else [],
            },
            "internal_sample_id": taskinterfacechunk.get("task_id"),
        }
        final_generation_trace: dict[str, object] = {
            "cache": "disabled",
            "cache_key": "",
            "reason": "cache_path_not_configured",
        }
        tool_names = [
            str(item.get("tool_name"))
            for item in selected_tools
            if isinstance(item, dict) and item.get("tool_name")
        ]
        uses_tool_generation = bool(
            self.config.use_chat_tool_calls
            and (
                not self.config.capability_enabled
                or self.config.capability_execution_mode == "llm_tool_call"
            )
            and tool_names
            and hasattr(self.model, "generate_with_tools")
        )
        final_cache_key = ""
        raw_output: str | None = None
        if not self.config.final_generation_cache_path:
            final_generation_trace = {
                "cache": "disabled",
                "cache_key": "",
                "reason": "cache_path_not_configured",
            }
        elif not allow_final_generation_cache:
            final_generation_trace = {
                "cache": "bypass",
                "cache_key": "",
                "reason": "update_memory_true",
            }
        elif uses_tool_generation:
            final_generation_trace = {
                "cache": "bypass",
                "cache_key": "",
                "reason": "tool_generation_path",
            }
        else:
            final_cache_key = _final_generation_cache_key(
                context=context,
                final_state=final_state,
                model=self.model,
                strip_output=self.config.strip_output,
            )
            cached_raw_output = self._final_generation_cache.get(final_cache_key)
            if cached_raw_output is not None:
                raw_output = cached_raw_output
                final_generation_trace = {
                    "cache": "hit",
                    "cache_key": final_cache_key[:12],
                    "reason": "context_model_and_attachments_match",
                }
                progress(
                    f"final generation cache hit task_id={taskinterfacechunk.get('task_id', '')} "
                    f"key={final_cache_key[:12]}"
                )
            else:
                final_generation_trace = {
                    "cache": "miss",
                    "cache_key": final_cache_key[:12],
                    "reason": "cache_miss",
                }
        if raw_output is not None:
            pass
        elif uses_tool_generation:
            raw_output = self.model.generate_with_tools(
                context,
                state=final_state,
                tools=self.tools.chat_tools(tool_names),
                tool_executor=lambda name, arguments: self.tools.execute(
                    name,
                    arguments,
                    execution_context=capability_execution_context,
                ),
                max_tool_rounds=max(int(self.config.tool_max_rounds), 0),
            )
        else:
            raw_output = self.model.generate(context, state=final_state)
            if final_cache_key and raw_output:
                self._final_generation_cache.put(final_cache_key, raw_output)
                final_generation_trace = {
                    "cache": "store",
                    "cache_key": final_cache_key[:12],
                    "reason": "stored_nonempty_output",
                }
                progress(
                    f"final generation cache store task_id={taskinterfacechunk.get('task_id', '')} "
                    f"key={final_cache_key[:12]}"
                )
        answer = raw_output.strip() if self.config.strip_output else raw_output
        return {
            "task_id": taskinterfacechunk.get("task_id"),
            "answer": answer,
            "trace": {
                "task_id": taskinterfacechunk.get("task_id"),
                "route_name": self.config.route_name,
                "router_config": self.config.to_dict(),
                "task_interface_config": task_interface_config.to_dict(),
                "task_interface_mode": taskinterfacechunk.get("metadata", {}).get("task_interface_mode"),
                "workflow_decision": workflow_decision,
                "workflow_selection": workflow_selection_trace,
                "prompt": context,
                "selected_memory": selected_memory,
                "memory_selection": memory_selection,
                "selected_skills": selected_skills,
                "selected_tools": selected_tools,
                "capability_results": capability_results,
                "final_generation": final_generation_trace,
                "raw_output": raw_output,
                "final_answer": answer,
            },
        }

    def plan_workflow(
        self,
        taskinterfacechunk: dict[str, Any],
        *,
        selected_skills: list[dict[str, object]],
    ) -> dict[str, object]:
        workflow_name, steps = _default_workflow(taskinterfacechunk)
        default = {
            "workflow_name": workflow_name,
            "steps": steps,
            "use_memory": self.config.memory_always_on,
            "use_tools": False,
            "memory_query": "",
            "tool_needs": [],
            "rationale": "deterministic_memory_policy" if self.config.memory_always_on else "default_no_external_context",
        }
        if not self.config.use_llm_workflow_selector:
            self._set_workflow_selection_trace({
                "cache": "bypass",
                "cache_key": "",
                "raw_output": "",
                "reason": "llm_workflow_selector_disabled",
            })
            return default
        model_visible = TaskInterface.to_model_visible(taskinterfacechunk)
        cache_key = self._workflow_selector_cache_key(
            model_visible=model_visible,
            selected_skills=selected_skills,
        )
        cached = self._workflow_selector_cache.get(cache_key)
        if cached is not None:
            progress(
                f"workflow selector cache hit task_id={taskinterfacechunk.get('task_id', '')} "
                f"key={cache_key[:12]}"
            )
            self._set_workflow_selection_trace({
                "cache": "hit",
                "cache_key": cache_key[:12],
                "raw_output": "",
                "reason": "cache_hit",
            })
            return _normalize_workflow_decision(cached, default)
        prompt_values = {
            "model_visible_input": json.dumps(model_visible, ensure_ascii=False, indent=2),
            "selected_skills": json.dumps(selected_skills, ensure_ascii=False, indent=2),
        }
        workflow_template = self.workflow_template
        if self.config.capability_enabled and self.capability_workflow_template is not None:
            workflow_template = self.capability_workflow_template
            prompt_values["available_capabilities"] = json.dumps(
                self.tools.capability_summaries(), ensure_ascii=False, indent=2
            )
        prompt = workflow_template.render(prompt_values)
        assert_no_label_leakage(prompt, phase="workflow_decision")
        raw_output = self.selection_model.generate(
            prompt,
            state={
                "router_phase": "plan_workflow",
                "internal_sample_id": taskinterfacechunk.get("task_id"),
                "max_new_tokens": max(int(self.config.workflow_selector_max_new_tokens), 1),
            },
        )
        decoded = _extract_json(raw_output)
        decision = _normalize_workflow_decision(decoded, default)
        self._workflow_selector_cache.put(cache_key, decision)
        self._set_workflow_selection_trace({
            "cache": "store" if self.config.workflow_selector_cache_path else "disabled",
            "cache_key": cache_key[:12],
            "raw_output": raw_output,
            "reason": "stored_normalized_decision" if self.config.workflow_selector_cache_path else "cache_path_not_configured",
        })
        if self.config.workflow_selector_cache_path:
            progress(
                f"workflow selector cache store task_id={taskinterfacechunk.get('task_id', '')} "
                f"key={cache_key[:12]}"
            )
        return decision

    def _workflow_selector_cache_key(
        self,
        *,
        model_visible: dict[str, Any],
        selected_skills: list[dict[str, object]],
    ) -> str:
        workflow_template = (
            self.capability_workflow_template
            if self.config.capability_enabled and self.capability_workflow_template is not None
            else self.workflow_template
        )
        payload = {
            "schema": "hcl_workflow_selector_cache_v1",
            "model": _model_cache_fingerprint(self.selection_model),
            "model_visible_input": model_visible,
            "selected_skills": selected_skills,
            "workflow_selector_artifact": _template_fingerprint(workflow_template),
            "memory_always_on": self.config.memory_always_on,
        }
        if self.config.capability_enabled:
            payload["available_capabilities"] = self.tools.capability_summaries()
        return _stable_json_hash(payload)

    def select_memory(
        self,
        taskinterfacechunk: dict[str, Any],
        workflow_decision: dict[str, object],
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        should_retrieve = self.config.memory_always_on or bool(workflow_decision.get("use_memory"))
        if not self.config.use_memory_selector or not should_retrieve or self.memory is None:
            return [], {"selected": [], "reason": "memory_selector_disabled"}
        model_visible = TaskInterface.to_model_visible(taskinterfacechunk)
        if hasattr(self.memory, "select_for_context"):
            return self.memory.select_for_context(
                current_input=model_visible,
                workflow_decision=workflow_decision,
                task_name=str(taskinterfacechunk.get("task_name", "")),
                task_description=str(taskinterfacechunk.get("question", "")),
            )
        candidates = self.memory.retrieval_candidates(
            current_input=model_visible,
            workflow_decision=workflow_decision,
        )
        if not candidates:
            return [], {"selected": [], "reason": "empty_global_memory_pool"}
        retrieval_stats = (
            self.memory.retrieval_stats()
            if hasattr(self.memory, "retrieval_stats")
            else {"pool_count": len(candidates), "candidate_count": len(candidates)}
        )
        prompt_values = {
            "current_input": json.dumps(
                sanitize_label_leakage_for_visible_prompt(model_visible),
                ensure_ascii=False,
                indent=2,
            ),
            "workflow_decision": json.dumps(
                sanitize_label_leakage_for_visible_prompt(workflow_decision),
                ensure_ascii=False,
                indent=2,
            ),
            "memory_candidates": json.dumps(
                sanitize_label_leakage_for_visible_prompt(candidates),
                ensure_ascii=False,
                indent=2,
            ),
            "retrieval_top_k": max(int(self.memory.config.retrieval_top_k), 0),
        }
        cache_key = self._memory_selector_cache_key(
            model_visible=model_visible,
            workflow_decision=workflow_decision,
            candidates=candidates,
            retrieval_top_k=prompt_values["retrieval_top_k"],
        )
        cached = self._memory_selector_cache.get(cache_key)
        if cached is not None:
            selected, trace = self._materialize_selected_memory(cached, candidates)
            fallback_used = (
                cached.get("discarded_summary") == INVALID_MEMORY_SELECTOR_FALLBACK_SUMMARY
            )
            trace.update(
                {
                    "prefilter_pool_count": retrieval_stats.get("pool_count", len(candidates)),
                    "prefilter_candidate_count": retrieval_stats.get("candidate_count", len(candidates)),
                    "cache": "hit",
                    "cache_key": cache_key[:12],
                    "attempt_count": 0,
                    "raw_output": "",
                    "raw_outputs": [],
                    "validation_diagnostics": [],
                    "fallback_to_empty_on_invalid": fallback_used,
                    "fallback_reason": "invalid_llm_output" if fallback_used else "",
                }
            )
            return selected, trace
        base_prompt = self.memory_selector_template.render(prompt_values)
        assert_no_label_leakage(base_prompt, phase="memory_selector")
        attempts = max(int(self.config.memory_selector_invalid_output_retries), 0) + 1
        raw_outputs: list[str] = []
        diagnostics: list[str] = []
        decoded: dict[str, Any] | None = None
        for attempt in range(attempts):
            is_retry = attempt > 0
            prompt = _memory_selector_retry_prompt(base_prompt) if is_retry else base_prompt
            token_limit = (
                self.config.memory_selector_retry_max_new_tokens
                if is_retry
                else self.config.memory_selector_max_new_tokens
            )
            raw_output = self.selection_model.generate(
                prompt,
                state={
                    "router_phase": "select_memory",
                    "router_attempt": attempt + 1,
                    "internal_sample_id": taskinterfacechunk.get("task_id"),
                    "max_new_tokens": max(int(token_limit), 1),
                },
            )
            raw_outputs.append(raw_output)
            parsed, parse_error = _extract_json_with_diagnostics(raw_output)
            if parse_error is not None:
                diagnostics.append(f"attempt_{attempt + 1}:{parse_error}")
                continue
            schema_error = _memory_selection_schema_error(parsed)
            if schema_error is not None:
                diagnostics.append(f"attempt_{attempt + 1}:schema_error:{schema_error}")
                continue
            decoded = parsed
            break
        fallback_used = False
        if decoded is None:
            if not self.config.memory_selector_fallback_to_empty_on_invalid:
                raise ValueError(
                    "Memory Selector LLM output remained invalid after "
                    f"{attempts} attempt(s): {'; '.join(diagnostics)}"
                )
            decoded = {
                "selected": [],
                "discarded_summary": INVALID_MEMORY_SELECTOR_FALLBACK_SUMMARY,
                "selection_confidence": "low",
            }
            fallback_used = True
            progress(
                "memory selector invalid; fallback to empty selection "
                f"task_id={taskinterfacechunk.get('task_id', '')} "
                f"attempts={attempts} diagnostics={'; '.join(diagnostics)}"
            )
        self._memory_selector_cache.put(cache_key, decoded)
        selected, trace = self._materialize_selected_memory(decoded, candidates)
        trace.update(
            {
                "prefilter_pool_count": retrieval_stats.get("pool_count", len(candidates)),
                "prefilter_candidate_count": retrieval_stats.get("candidate_count", len(candidates)),
                "cache": "store" if self.config.memory_selector_cache_path else "disabled",
                "cache_key": cache_key[:12],
                "attempt_count": len(raw_outputs),
                "raw_output": raw_outputs[-1],
                "raw_outputs": raw_outputs,
                "validation_diagnostics": diagnostics,
                "fallback_to_empty_on_invalid": fallback_used,
                "fallback_reason": "invalid_llm_output" if fallback_used else "",
            }
        )
        return selected, trace

    def _memory_selector_cache_key(
        self,
        *,
        model_visible: dict[str, Any],
        workflow_decision: dict[str, object],
        candidates: list[dict[str, Any]],
        retrieval_top_k: int,
    ) -> str:
        payload = {
            "schema": "hcl_memory_selector_cache_v1",
            "model": _model_cache_fingerprint(self.selection_model),
            "model_visible_input": model_visible,
            "workflow_decision": workflow_decision,
            "memory_candidates": candidates,
            "retrieval_top_k": retrieval_top_k,
            "memory_selector_artifact": {
                "template_id": self.memory_selector_template.template_id,
                "description": self.memory_selector_template.description,
                "sections": self.memory_selector_template.sections,
            },
        }
        return _stable_json_hash(payload)

    def _materialize_selected_memory(
        self,
        decoded: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        selected_specs = decoded["selected"]
        candidate_by_id = {str(item.get("memory_id", "")): item for item in candidates}
        selected: list[dict[str, object]] = []
        accepted_specs: list[dict[str, object]] = []
        seen_ids: set[str] = set()
        limit = max(int(self.memory.config.retrieval_top_k), 0)
        for spec in selected_specs:
            if len(selected) >= limit:
                break
            if not isinstance(spec, dict):
                continue
            memory_id = str(spec.get("memory_id", ""))
            if not memory_id or memory_id in seen_ids or memory_id not in candidate_by_id:
                continue
            view = dict(candidate_by_id[memory_id])
            view["selected_role"] = str(spec.get("role", "support"))
            view["selection_reason"] = str(spec.get("reason", ""))
            view["selection_risk"] = str(spec.get("risk", "medium"))
            selected.append(view)
            accepted_specs.append(dict(spec))
            seen_ids.add(memory_id)
        trace = {
            "candidate_count": len(candidates),
            "selected": accepted_specs,
            "discarded_summary": str(decoded.get("discarded_summary", "")),
            "selection_confidence": str(decoded.get("selection_confidence", "low")),
        }
        return selected, trace

    def select_skills(self, taskinterfacechunk: dict[str, Any]) -> list[dict[str, object]]:
        if not self.config.use_skill_selector:
            return []
        model_visible = TaskInterface.to_model_visible(taskinterfacechunk)
        candidates = (
            self.memory.available_skill_views()
            if self.memory is not None and hasattr(self.memory, "available_skill_views")
            else []
        )
        if not candidates and self.config.skill_dir:
            candidates = self.skills.available_skill_views()
        if not candidates:
            return []
        cache_key = self._skill_selector_cache_key(model_visible=model_visible, candidates=candidates)
        cached = self._skill_selector_cache.get(cache_key)
        if cached is not None:
            progress(
                f"skill selector cache hit task_id={taskinterfacechunk.get('task_id', '')} "
                f"key={cache_key[:12]}"
            )
            return _materialize_named_selection(cached, candidates, name_field="skill_name")
        prompt_values = {
            "model_visible_input": json.dumps(
                sanitize_label_leakage_for_visible_prompt(model_visible),
                ensure_ascii=False,
                indent=2,
            ),
            "skill_candidates": json.dumps(
                sanitize_label_leakage_for_visible_prompt(candidates),
                ensure_ascii=False,
                indent=2,
            ),
        }
        base_prompt = self.skill_selector_template.render(prompt_values)
        assert_no_label_leakage(base_prompt, phase="skill_selector")
        attempts = max(int(self.config.skill_selector_invalid_output_retries), 0) + 1
        diagnostics: list[str] = []
        decoded: dict[str, Any] | None = None
        for attempt in range(attempts):
            is_retry = attempt > 0
            prompt = _selector_retry_prompt(base_prompt) if is_retry else base_prompt
            token_limit = (
                self.config.skill_selector_retry_max_new_tokens
                if is_retry
                else self.config.skill_selector_max_new_tokens
            )
            raw_output = self.selection_model.generate(
                prompt,
                state={
                    "router_phase": "select_skill",
                    "router_attempt": attempt + 1,
                    "internal_sample_id": taskinterfacechunk.get("task_id"),
                    "max_new_tokens": max(int(token_limit), 1),
                },
            )
            parsed, parse_error = _extract_json_with_diagnostics(raw_output)
            if parse_error is not None:
                diagnostics.append(f"attempt_{attempt + 1}:{parse_error}")
                continue
            schema_error = _named_selection_schema_error(parsed, name_field="skill_name")
            if schema_error is not None:
                diagnostics.append(f"attempt_{attempt + 1}:schema_error:{schema_error}")
                continue
            decoded = parsed
            break
        if decoded is None:
            if self.config.skill_selector_fallback_to_empty_on_invalid:
                decoded = {
                    "selected": [],
                    "discarded_summary": "skill selector invalid; fallback to no skills",
                    "selection_confidence": "low",
                }
                progress(
                    "skill selector invalid; fallback to empty selection "
                    f"task_id={taskinterfacechunk.get('task_id', '')} "
                    f"attempts={attempts} diagnostics={'; '.join(diagnostics)}"
                )
                self._skill_selector_cache.put(cache_key, decoded)
                return []
            raise ValueError(
                "Skill Selector LLM output remained invalid after "
                f"{attempts} attempt(s): {'; '.join(diagnostics)}"
            )
        self._skill_selector_cache.put(cache_key, decoded)
        if self.config.skill_selector_cache_path:
            progress(
                f"skill selector cache store task_id={taskinterfacechunk.get('task_id', '')} "
                f"key={cache_key[:12]}"
            )
        return _materialize_named_selection(decoded, candidates, name_field="skill_name")

    def _skill_selector_cache_key(
        self,
        *,
        model_visible: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> str:
        payload = {
            "schema": "hcl_skill_selector_cache_v1",
            "model": _model_cache_fingerprint(self.selection_model),
            "model_visible_input": model_visible,
            "skill_candidates": candidates,
            "skill_selector_artifact": _template_fingerprint(self.skill_selector_template),
        }
        return _stable_json_hash(payload)

    def select_tools(
        self,
        taskinterfacechunk: dict[str, Any],
        workflow_decision: dict[str, object],
        *,
        selected_skills: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        if not self.config.use_tool_selector:
            return []
        model_visible = TaskInterface.to_model_visible(taskinterfacechunk)
        requested = bool(workflow_decision.get("use_tools"))
        heuristic_names = self.tools.select_tool_names(model_visible, workflow_decision)
        forced_names = (
            ["semantic_search"]
            if self.config.capability_enabled
            and self.config.capability_execution_mode == "router_managed"
            and self.config.capability_force_semantic_search
            else []
        )
        if not requested and not (self.config.tool_auto_select and heuristic_names) and not forced_names:
            return []
        candidates = (
            self.tools.eligible_tool_views(
                model_visible,
                self._capability_execution_context(taskinterfacechunk),
            )
            if self.config.capability_enabled
            else self.tools.available_tool_views()
        )
        if not candidates:
            return []
        compact_candidates = _compact_tool_candidates(candidates)
        cache_key = self._tool_selector_cache_key(
            model_visible=model_visible,
            workflow_decision=workflow_decision,
            selected_skills=selected_skills,
            candidates=compact_candidates,
            heuristic_names=heuristic_names,
        )
        cached = self._tool_selector_cache.get(cache_key)
        if cached is not None:
            progress(
                f"tool selector cache hit task_id={taskinterfacechunk.get('task_id', '')} "
                f"key={cache_key[:12]}"
            )
            selected = _materialize_named_selection(cached, candidates, name_field="tool_name")
            return _append_named_tools(selected, candidates, forced_names)
        prompt_values = {
            "model_visible_input": json.dumps(
                sanitize_label_leakage_for_visible_prompt(model_visible),
                ensure_ascii=False,
                indent=2,
            ),
            "workflow_decision": json.dumps(
                sanitize_label_leakage_for_visible_prompt(workflow_decision),
                ensure_ascii=False,
                indent=2,
            ),
            "selected_skills": json.dumps(
                sanitize_label_leakage_for_visible_prompt(selected_skills),
                ensure_ascii=False,
                indent=2,
            ),
            "tool_candidates": json.dumps(
                sanitize_label_leakage_for_visible_prompt(compact_candidates),
                ensure_ascii=False,
                indent=2,
            ),
            "heuristic_tools": json.dumps(heuristic_names, ensure_ascii=False, indent=2),
        }
        base_prompt = self.tool_selector_template.render(prompt_values)
        assert_no_label_leakage(base_prompt, phase="tool_selector")
        attempts = max(int(self.config.tool_selector_invalid_output_retries), 0) + 1
        diagnostics: list[str] = []
        decoded: dict[str, Any] | None = None
        for attempt in range(attempts):
            is_retry = attempt > 0
            prompt = _selector_retry_prompt(base_prompt) if is_retry else base_prompt
            token_limit = (
                self.config.tool_selector_retry_max_new_tokens
                if is_retry
                else self.config.tool_selector_max_new_tokens
            )
            raw_output = self.selection_model.generate(
                prompt,
                state={
                    "router_phase": "select_tool",
                    "router_attempt": attempt + 1,
                    "internal_sample_id": taskinterfacechunk.get("task_id"),
                    "max_new_tokens": max(int(token_limit), 1),
                },
            )
            parsed, parse_error = _extract_json_with_diagnostics(raw_output)
            if parse_error is not None:
                diagnostics.append(f"attempt_{attempt + 1}:{parse_error}")
                continue
            schema_error = _named_selection_schema_error(parsed, name_field="tool_name")
            if schema_error is not None:
                diagnostics.append(f"attempt_{attempt + 1}:schema_error:{schema_error}")
                continue
            decoded = parsed
            break
        if decoded is None:
            if self.config.tool_selector_fallback_to_empty_on_invalid:
                decoded = {
                    "selected": [],
                    "discarded_summary": "tool selector invalid; fallback to no tools",
                    "selection_confidence": "low",
                }
                progress(
                    "tool selector invalid; fallback to empty selection "
                    f"task_id={taskinterfacechunk.get('task_id', '')} "
                    f"attempts={attempts} diagnostics={'; '.join(diagnostics)}"
                )
                self._tool_selector_cache.put(cache_key, decoded)
                return []
            raise ValueError(
                "Tool Selector LLM output remained invalid after "
                f"{attempts} attempt(s): {'; '.join(diagnostics)}"
            )
        self._tool_selector_cache.put(cache_key, decoded)
        if self.config.tool_selector_cache_path:
            progress(
                f"tool selector cache store task_id={taskinterfacechunk.get('task_id', '')} "
                f"key={cache_key[:12]}"
            )
        selected = _materialize_named_selection(decoded, candidates, name_field="tool_name")
        return _append_named_tools(selected, candidates, forced_names)

    def _tool_selector_cache_key(
        self,
        *,
        model_visible: dict[str, Any],
        workflow_decision: dict[str, object],
        selected_skills: list[dict[str, object]],
        candidates: list[dict[str, object]],
        heuristic_names: list[str],
    ) -> str:
        payload = {
            "schema": "hcl_tool_selector_cache_v1",
            "model": _model_cache_fingerprint(self.selection_model),
            "model_visible_input": model_visible,
            "workflow_decision": workflow_decision,
            "selected_skills": selected_skills,
            "tool_candidates": candidates,
            "heuristic_tools": heuristic_names,
            "tool_selector_artifact": _template_fingerprint(self.tool_selector_template),
        }
        return _stable_json_hash(payload)

    def build_context(
        self,
        taskinterfacechunk: dict[str, Any],
        *,
        workflow_decision: dict[str, object],
        selected_memory: list[dict[str, object]],
        selected_skills: list[dict[str, object]],
        selected_tools: list[dict[str, object]],
        capability_results: list[dict[str, object]] | None = None,
    ) -> str:
        model_visible = sanitize_label_leakage_for_visible_prompt(
            TaskInterface.to_model_visible(taskinterfacechunk)
        )
        visible_workflow = sanitize_label_leakage_for_visible_prompt(workflow_decision)
        visible_memory = sanitize_label_leakage_for_visible_prompt(selected_memory)
        visible_skills = sanitize_label_leakage_for_visible_prompt(selected_skills)
        visible_tools = sanitize_label_leakage_for_visible_prompt(selected_tools)
        prompt_values = {
            "model_visible_input": json.dumps(model_visible, ensure_ascii=False, indent=2),
            "workflow_decision": json.dumps(visible_workflow, ensure_ascii=False, indent=2),
            "selected_memory": json.dumps(visible_memory, ensure_ascii=False, indent=2),
            "selected_skills": json.dumps(visible_skills, ensure_ascii=False, indent=2),
            "selected_tools": json.dumps(visible_tools, ensure_ascii=False, indent=2),
        }
        context_template = self.context_template
        if self.config.capability_enabled and self.capability_context_template is not None:
            context_template = self.capability_context_template
            prompt_values["capability_results"] = json.dumps(
                sanitize_label_leakage_for_visible_prompt(capability_results or []),
                ensure_ascii=False,
                indent=2,
            )
        return context_template.render(prompt_values)

    def _capability_execution_context(
        self,
        taskinterfacechunk: dict[str, Any],
    ) -> CapabilityExecutionContext:
        task_context = taskinterfacechunk.get("task_context", {})
        raw_files = task_context.get("files", []) if isinstance(task_context, dict) else []
        capability_inputs = task_context.get("capability_inputs", {}) if isinstance(task_context, dict) else {}
        raw_candidates = (
            capability_inputs.get("text_candidates", [])
            if isinstance(capability_inputs, dict)
            else []
        )
        attachments: list[dict[str, Any]] = []
        for index, item in enumerate(raw_files if isinstance(raw_files, list) else []):
            if not isinstance(item, dict):
                continue
            attachments.append({**item, "attachment_id": str(item.get("attachment_id") or f"attachment_{index}")})
        allowed_corpora = tuple(str(item) for item in self.config.capability_allowed_corpora)
        if not allowed_corpora and self.config.capability_default_corpus_id:
            allowed_corpora = (self.config.capability_default_corpus_id,)
        return CapabilityExecutionContext(
            task_id=str(taskinterfacechunk.get("task_id", "")),
            attachments=tuple(attachments),
            text_candidates=tuple(
                {"id": str(item.get("id", "")), "text": str(item.get("text", ""))}
                for item in raw_candidates
                if isinstance(item, dict) and str(item.get("id", "")).strip() and str(item.get("text", "")).strip()
            ),
            allowed_corpora=allowed_corpora,
        )

    def execute_selected_capabilities(
        self,
        taskinterfacechunk: dict[str, Any],
        workflow_decision: dict[str, object],
        selected_tools: list[dict[str, object]],
        *,
        execution_context: CapabilityExecutionContext,
    ) -> list[dict[str, object]]:
        if not self.config.capability_enabled:
            return []
        if self.config.capability_execution_mode != "router_managed":
            return []
        results: list[dict[str, object]] = []
        for selected in selected_tools:
            tool_name = str(selected.get("tool_name", "")) if isinstance(selected, dict) else ""
            arguments = self._router_managed_arguments(
                tool_name,
                taskinterfacechunk,
                workflow_decision,
                execution_context,
            )
            if arguments is None:
                results.append({
                    "tool_name": tool_name,
                    "status": "skipped",
                    "error": "No safe router-managed argument builder is available.",
                })
                continue
            try:
                output = self.tools.execute(
                    tool_name,
                    arguments,
                    execution_context=execution_context,
                )
                if tool_name == "arithmetic_calculator":
                    cache_key = self._tool_argument_cache_key(
                        tool_name=tool_name,
                        taskinterfacechunk=taskinterfacechunk,
                        workflow_decision=workflow_decision,
                    )
                    self._tool_argument_cache.put(
                        cache_key,
                        {"tool_name": tool_name, "arguments": arguments},
                    )
                results.append({
                    "tool_name": tool_name,
                    "status": "success",
                    "arguments": arguments,
                    "output": output,
                })
            except (CapabilityExecutionError, RuntimeError, ValueError) as exc:
                if not self.config.capability_continue_on_error:
                    raise
                results.append({
                    "tool_name": tool_name,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                })
        return results

    def _router_managed_arguments(
        self,
        tool_name: str,
        taskinterfacechunk: dict[str, Any],
        workflow_decision: dict[str, object],
        execution_context: CapabilityExecutionContext,
    ) -> dict[str, Any] | None:
        question = TaskInterface.visible_question(taskinterfacechunk).strip()
        if tool_name == "semantic_search":
            query = _normalize_optional_text(workflow_decision.get("memory_query")) or question
            if not query:
                return None
            return {
                "query": query,
                "corpus_id": self.config.capability_default_corpus_id,
                "top_k": max(1, min(int(self.config.capability_default_top_k), 20)),
            }
        if tool_name == "cross_modal_match":
            image = next(
                (item for item in execution_context.attachments if str(item.get("type", "")) == "image"),
                None,
            )
            if image is None or not execution_context.text_candidates:
                return None
            return {
                "attachment_id": str(image.get("attachment_id", "")),
                "candidates": [dict(item) for item in execution_context.text_candidates],
                "top_k": min(
                    max(1, int(self.config.capability_default_top_k)),
                    len(execution_context.text_candidates),
                    20,
                ),
            }
        if tool_name == "arithmetic_calculator":
            return self._build_arithmetic_arguments(taskinterfacechunk, workflow_decision)
        return None

    def _build_arithmetic_arguments(
        self,
        taskinterfacechunk: dict[str, Any],
        workflow_decision: dict[str, object],
    ) -> dict[str, Any] | None:
        tool_name = "arithmetic_calculator"
        cache_key = self._tool_argument_cache_key(
            tool_name=tool_name,
            taskinterfacechunk=taskinterfacechunk,
            workflow_decision=workflow_decision,
        )
        cached = self._tool_argument_cache.get(cache_key)
        if cached is not None:
            progress(
                f"tool argument cache hit task_id={taskinterfacechunk.get('task_id', '')} "
                f"tool={tool_name} key={cache_key[:12]}"
            )
            arguments = cached.get("arguments")
            return dict(arguments) if isinstance(arguments, dict) else None

        chat_tools = self.tools.chat_tools([tool_name])
        if not chat_tools:
            return None
        function_schema = chat_tools[0].get("function", {})
        model_visible = sanitize_label_leakage_for_visible_prompt(
            TaskInterface.to_model_visible(taskinterfacechunk)
        )
        visible_workflow = sanitize_label_leakage_for_visible_prompt(workflow_decision)
        base_prompt = self.tool_argument_template.render({
            "model_visible_input": json.dumps(model_visible, ensure_ascii=False, indent=2),
            "workflow_decision": json.dumps(visible_workflow, ensure_ascii=False, indent=2),
            "tool_schema": json.dumps(function_schema, ensure_ascii=False, indent=2),
        })
        assert_no_label_leakage(base_prompt, phase="tool_argument_builder")
        attempts = max(int(self.config.tool_argument_invalid_output_retries), 0) + 1
        diagnostics: list[str] = []
        for attempt in range(attempts):
            is_retry = attempt > 0
            prompt = base_prompt
            if is_retry:
                prompt = "\n".join((
                    base_prompt,
                    "[RETRY_AFTER_INVALID_OUTPUT]",
                    "Return one compact JSON object matching the required shape exactly.",
                    "Use only numeric literals, parentheses, and + - * / // % ** in expression.",
                ))
            token_limit = (
                self.config.tool_argument_retry_max_new_tokens
                if is_retry
                else self.config.tool_argument_max_new_tokens
            )
            raw_output = self.model.generate(
                prompt,
                state={
                    "router_phase": "build_tool_arguments",
                    "router_attempt": attempt + 1,
                    "internal_sample_id": taskinterfacechunk.get("task_id"),
                    "max_new_tokens": max(int(token_limit), 1),
                },
            )
            decoded, parse_error = _extract_json_with_diagnostics(raw_output)
            if parse_error is not None:
                diagnostics.append(f"attempt_{attempt + 1}:{parse_error}")
                continue
            schema_error = _tool_argument_decision_schema_error(decoded)
            if schema_error is not None:
                diagnostics.append(f"attempt_{attempt + 1}:schema_error:{schema_error}")
                continue
            if str(decoded.get("tool_name", "")) != tool_name:
                diagnostics.append(f"attempt_{attempt + 1}:wrong_tool_name")
                continue
            arguments = decoded.get("arguments")
            if not isinstance(arguments, dict):
                diagnostics.append(f"attempt_{attempt + 1}:arguments_not_object")
                continue
            try:
                # This operation is pure. It validates both the JSON schema and
                # the calculator's small arithmetic-AST allow-list before caching.
                self.tools.execute(tool_name, arguments)
            except (CapabilityExecutionError, RuntimeError, ValueError, SyntaxError) as exc:
                diagnostics.append(
                    f"attempt_{attempt + 1}:unsafe_or_invalid:{type(exc).__name__}:{exc}"
                )
                continue
            if self.config.tool_argument_cache_path:
                progress(
                    f"tool argument cache store task_id={taskinterfacechunk.get('task_id', '')} "
                    f"tool={tool_name} key={cache_key[:12]}"
                )
            return dict(arguments)
        progress(
            f"tool argument builder invalid task_id={taskinterfacechunk.get('task_id', '')} "
            f"tool={tool_name} attempts={attempts} diagnostics={'; '.join(diagnostics)}"
        )
        return None

    def _tool_argument_cache_key(
        self,
        *,
        tool_name: str,
        taskinterfacechunk: dict[str, Any],
        workflow_decision: dict[str, object],
    ) -> str:
        chat_tools = self.tools.chat_tools([tool_name])
        payload = {
            "schema": "hcl_tool_argument_cache_v1",
            "model": _model_cache_fingerprint(self.model),
            "model_visible_input": TaskInterface.to_model_visible(taskinterfacechunk),
            "workflow_decision": workflow_decision,
            "tool_name": tool_name,
            "tool_schema": chat_tools[0] if chat_tools else {},
            "argument_template": _template_fingerprint(self.tool_argument_template),
        }
        return _stable_json_hash(payload)

def _extract_json(text: str) -> dict[str, Any]:
    data, _ = _extract_json_with_diagnostics(text)
    return data


def _tool_argument_decision_schema_error(value: dict[str, Any]) -> object | None:
    if not isinstance(value, dict):
        return "decision must be an object"
    tool_name = value.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name.strip():
        return "tool_name must be a non-empty string"
    if not isinstance(value.get("arguments"), dict):
        return "arguments must be an object"
    return None


def _append_named_tools(
    selected: list[dict[str, object]],
    candidates: list[dict[str, Any]],
    forced_names: list[str],
) -> list[dict[str, object]]:
    result = list(selected)
    present = {str(item.get("tool_name", "")) for item in result if isinstance(item, dict)}
    by_name = {
        str(item.get("tool_name", "")): item
        for item in candidates
        if isinstance(item, dict) and item.get("tool_name")
    }
    for name in forced_names:
        if name in present or name not in by_name:
            continue
        result.append(dict(by_name[name]))
        present.add(name)
    return result


def _normalize_workflow_decision(decoded: dict[str, Any], default: dict[str, object]) -> dict[str, object]:
    if not decoded:
        return dict(default)
    return {
        "workflow_name": str(decoded.get("workflow_name") or default.get("workflow_name", "")),
        "steps": _string_list(decoded.get("steps")) or list(default.get("steps", [])),
        "use_memory": bool(default.get("use_memory", False)) or bool(decoded.get("use_memory", False)),
        "use_tools": bool(decoded.get("use_tools", False)),
        "memory_query": _normalize_optional_text(decoded.get("memory_query")),
        "tool_needs": _string_list(decoded.get("tool_needs")),
        "rationale": str(decoded.get("rationale", "")),
    }


def _normalize_optional_text(value: object) -> str:
    """Normalize LLM null-like placeholders before they become tool arguments."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in {"none", "null", "nil", "n/a", "na", "not applicable"}:
        return ""
    return text


def _extract_json_with_diagnostics(text: str) -> tuple[dict[str, Any], str | None]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as direct_error:
        start = text.find("{")
        if start < 0:
            return {}, "json_decode_error:no_object_start"
        try:
            data, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError as extracted_error:
            kind = "likely_truncated_json" if _looks_truncated(text) else "json_decode_error"
            return {}, (
                f"{kind}:{extracted_error.msg} at line "
                f"{extracted_error.lineno} column {extracted_error.colno}; "
                f"direct_error={direct_error.msg}"
            )
    if not isinstance(data, dict):
        return {}, f"top_level_not_object:{type(data).__name__}"
    return data, None


def _looks_truncated(text: str) -> bool:
    stripped = str(text).rstrip()
    return not stripped or not stripped.endswith("}") or stripped.count("{") > stripped.count("}")


def _memory_selection_schema_error(decoded: dict[str, Any]) -> str | None:
    expected = {"selected", "discarded_summary", "selection_confidence"}
    if set(decoded) != expected:
        return f"top_level_fields_must_be_exactly:{sorted(expected)}"
    if not isinstance(decoded.get("selected"), list):
        return "selected_must_be_list"
    if not isinstance(decoded.get("discarded_summary"), str):
        return "discarded_summary_must_be_string"
    if decoded.get("selection_confidence") not in {"low", "medium", "high"}:
        return "selection_confidence_must_be_low_medium_or_high"
    required_item_fields = {"memory_id", "role", "reason", "risk"}
    allowed_roles = {"reasoning_support", "failure_avoidance", "output_format_control"}
    for index, item in enumerate(decoded["selected"]):
        if not isinstance(item, dict):
            return f"selected_{index}_must_be_object"
        if set(item) != required_item_fields:
            return f"selected_{index}_fields_must_be_exactly:{sorted(required_item_fields)}"
        if not str(item.get("memory_id", "")).strip():
            return f"selected_{index}_memory_id_must_be_nonempty"
        if item.get("role") not in allowed_roles:
            return f"selected_{index}_role_invalid"
        if not isinstance(item.get("reason"), str):
            return f"selected_{index}_reason_must_be_string"
        if item.get("risk") not in {"low", "medium", "high"}:
            return f"selected_{index}_risk_invalid"
    return None


def _workflow_decision_schema_error(decoded: dict[str, Any]) -> str | None:
    required = {"workflow_name", "steps", "use_memory", "use_tools", "memory_query", "tool_needs", "rationale"}
    if not required.issubset(set(decoded)):
        return f"missing_workflow_fields:{sorted(required - set(decoded))}"
    if not isinstance(decoded.get("workflow_name"), str):
        return "workflow_name_must_be_string"
    if not isinstance(decoded.get("steps"), list):
        return "steps_must_be_list"
    if not isinstance(decoded.get("use_memory"), bool):
        return "use_memory_must_be_bool"
    if not isinstance(decoded.get("use_tools"), bool):
        return "use_tools_must_be_bool"
    if not isinstance(decoded.get("memory_query"), str):
        return "memory_query_must_be_string"
    if not isinstance(decoded.get("tool_needs"), list):
        return "tool_needs_must_be_list"
    if not isinstance(decoded.get("rationale"), str):
        return "rationale_must_be_string"
    return None


def _memory_selector_retry_prompt(base_prompt: str) -> str:
    return "\n".join(
        (
            base_prompt,
            "[RETRY_AFTER_INVALID_OUTPUT]",
            "The previous response was invalid or truncated. Return a compact complete JSON object now.",
            "Use at most 18 words for each reason and at most 12 words for discarded_summary.",
            "Do not explain outside JSON. Close every string, array, and object.",
        )
    )


def _named_selection_schema_error(decoded: dict[str, Any], *, name_field: str) -> str | None:
    expected = {"selected", "discarded_summary", "selection_confidence"}
    if set(decoded) != expected:
        return f"top_level_fields_must_be_exactly:{sorted(expected)}"
    if not isinstance(decoded.get("selected"), list):
        return "selected_must_be_list"
    if not isinstance(decoded.get("discarded_summary"), str):
        return "discarded_summary_must_be_string"
    if decoded.get("selection_confidence") not in {"low", "medium", "high"}:
        return "selection_confidence_must_be_low_medium_or_high"
    required_item_fields = {name_field, "reason", "risk"}
    for index, item in enumerate(decoded["selected"]):
        if not isinstance(item, dict):
            return f"selected_{index}_must_be_object"
        if set(item) != required_item_fields:
            return f"selected_{index}_fields_must_be_exactly:{sorted(required_item_fields)}"
        if not str(item.get(name_field, "")).strip():
            return f"selected_{index}_{name_field}_must_be_nonempty"
        if not isinstance(item.get("reason"), str):
            return f"selected_{index}_reason_must_be_string"
        if item.get("risk") not in {"low", "medium", "high"}:
            return f"selected_{index}_risk_invalid"
    return None


def _selector_retry_prompt(base_prompt: str) -> str:
    return "\n".join(
        (
            base_prompt,
            "[RETRY_AFTER_INVALID_OUTPUT]",
            "The previous response was invalid or truncated. Return a compact complete JSON object now.",
            "Use at most 18 words for each reason and at most 12 words for discarded_summary.",
            "Do not explain outside JSON. Close every string, array, and object.",
        )
    )


def _compact_tool_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, object]]:
    compact: list[dict[str, object]] = []
    for item in candidates:
        compact.append(
            {
                "tool_name": item.get("tool_name", ""),
                "description": item.get("description", ""),
                "source": item.get("source", ""),
                "tags": item.get("tags", []),
            }
        )
    return compact


def _materialize_named_selection(
    decoded: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    name_field: str,
) -> list[dict[str, object]]:
    candidate_by_name = {str(item.get(name_field, "")): item for item in candidates}
    selected: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for spec in decoded.get("selected", []):
        if not isinstance(spec, dict):
            continue
        name = str(spec.get(name_field, ""))
        if not name or name in seen_names or name not in candidate_by_name:
            continue
        view = dict(candidate_by_name[name])
        view["selection_reason"] = str(spec.get("reason", ""))
        view["selection_risk"] = str(spec.get("risk", "medium"))
        selected.append(view)
        seen_names.add(name)
    return selected


class _MemorySelectorCache(_BaseJsonObjectCache):
    """Append-only cache of successful, schema-valid Memory Selector outputs."""

    def __init__(self, path: str | None) -> None:
        super().__init__(
            path,
            value_field="memory_selection",
            schema_error=_memory_selection_schema_error,
        )


class _JsonObjectCache(_BaseJsonObjectCache):
    """Append-only JSONL cache for schema-valid selector decisions."""

    def __init__(
        self,
        path: str | None,
        *,
        value_field: str,
        schema_error: Any,
    ) -> None:
        super().__init__(path, value_field=value_field, schema_error=schema_error)


class _FinalGenerationCache(_BaseTextValueCache):
    """Append-only JSONL cache for successful final model generations."""

    def __init__(self, path: str | None) -> None:
        super().__init__(path, value_field="raw_output")


def _template_fingerprint(template: PromptTemplate) -> dict[str, object]:
    return {
        "template_id": template.template_id,
        "description": template.description,
        "sections": template.sections,
    }


def _model_cache_fingerprint(model: Any) -> dict[str, object]:
    config = getattr(model, "config", None)
    return {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "name": str(getattr(config, "name", "")),
        "backend": str(getattr(config, "backend", "")),
        "path": str(getattr(config, "path", "")),
        "base_url": str(getattr(config, "base_url", "")),
        "max_new_tokens": str(getattr(config, "max_new_tokens", "")),
        "answer_max_new_tokens": str(getattr(config, "answer_max_new_tokens", "")),
        "temperature": str(getattr(config, "temperature", "")),
        "thinking": getattr(config, "thinking", None),
        "reasoning_effort": str(getattr(config, "reasoning_effort", "")),
        "enable_thinking": str(getattr(config, "enable_thinking", "")),
        "cleanup_output": str(getattr(config, "cleanup_output", "")),
        "system_prompt_hash": _stable_json_hash(str(getattr(config, "system_prompt", ""))),
    }


def _final_generation_cache_key(
    *,
    context: str,
    final_state: dict[str, Any],
    model: Any,
    strip_output: bool,
) -> str:
    payload = {
        "schema": "hcl_final_generation_cache_v1",
        "context": context,
        "state": _final_generation_state_fingerprint(final_state),
        "model": _final_generation_model_fingerprint(model),
        "strip_output": bool(strip_output),
    }
    return _stable_json_hash(payload)


def _final_generation_model_fingerprint(model: Any) -> dict[str, object]:
    config = getattr(model, "config", None)
    return {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "name": str(getattr(config, "name", "")),
        "backend": str(getattr(config, "backend", "")),
        "path": str(getattr(config, "path", "")),
        "base_url": str(getattr(config, "base_url", "")),
        "max_new_tokens": str(getattr(config, "max_new_tokens", "")),
        "answer_max_new_tokens": str(getattr(config, "answer_max_new_tokens", "")),
        "temperature": str(getattr(config, "temperature", "")),
        "thinking": getattr(config, "thinking", None),
        "reasoning_effort": str(getattr(config, "reasoning_effort", "")),
        "enable_thinking": str(getattr(config, "enable_thinking", "")),
        "cleanup_output": str(getattr(config, "cleanup_output", "")),
        "system_prompt_hash": _stable_json_hash(str(getattr(config, "system_prompt", ""))),
    }


def _final_generation_state_fingerprint(final_state: dict[str, Any]) -> dict[str, object]:
    task_context = final_state.get("task_context")
    files = task_context.get("files", []) if isinstance(task_context, dict) else []
    return {
        "question": str(final_state.get("question", "")),
        "files": _attachment_fingerprints(files if isinstance(files, list) else []),
    }


def _attachment_fingerprints(files: list[Any]) -> list[dict[str, object]]:
    fingerprints: list[dict[str, object]] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        path_value = str(item.get("path", "")) if item.get("path") else ""
        path = Path(path_value) if path_value else None
        content_hash = ""
        status = "missing"
        size: int | None = None
        if path is not None:
            try:
                data = path.read_bytes()
                content_hash = hashlib.sha256(data).hexdigest()
                status = "content_hash"
                size = len(data)
            except OSError:
                content_hash = _stable_json_hash(path_value)
                status = "unavailable_path_hash"
        fingerprints.append(
            {
                "type": str(item.get("type", "")),
                "content_hash": content_hash,
                "status": status,
                "size": size,
            }
        )
    return fingerprints


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _default_workflow(taskinterfacechunk: dict[str, Any]) -> tuple[str, list[str]]:
    visible = TaskInterface.to_model_visible(taskinterfacechunk)
    evidence = visible.get("evidence", {})
    evidence_items = evidence.get("items", []) if isinstance(evidence, dict) else []
    has_image = any(
        isinstance(item, dict) and item.get("modality") == "image"
        for item in evidence_items
    )
    constraints = visible.get("recognizable_constraints", [])
    constraint_text = " ".join(
        str(item.get("content", ""))
        for item in constraints
        if isinstance(item, dict)
    ).lower()
    if has_image:
        return "visual_evidence_reasoning", [
            "inspect_task_and_image",
            "consult_selected_memory",
            "reason_from_visual_evidence",
            "return_required_format",
        ]
    if "numeric value" in constraint_text or "number" in constraint_text:
        return "text_math_reasoning", [
            "parse_problem",
            "consult_selected_memory",
            "solve_carefully",
            "return_required_format",
        ]
    return "direct_reasoning", [
        "read_task",
        "consult_selected_memory",
        "reason",
        "return_required_format",
    ]
