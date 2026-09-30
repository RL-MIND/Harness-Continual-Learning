from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any, Protocol

from ..json_utils import atomic_write_json
from ..progress import progress
from ..task_interface import infer_output_form


class Model(Protocol):
    def generate(self, prompt: str, *, state: dict[str, Any]) -> str:
        ...


class AbstractMemory:
    """Two merged abstractions per task: policy and failure avoidance."""

    def __init__(
        self,
        path: str | Path,
        *,
        model: Model | None = None,
        mode: str = "rule",
        max_new_tokens: int = 512,
        prompt_max_chars: int = 12000,
    ) -> None:
        self.path = Path(path)
        self.model = model
        self.mode = str(mode or "rule").lower()
        self.max_new_tokens = max(int(max_new_tokens), 1)
        self.prompt_max_chars = max(int(prompt_max_chars), 1000)
        self.records = self._load()

    def clear(self) -> None:
        self.records = []
        self._persist()

    def consolidate(
        self,
        task_name: str,
        examples: list[dict[str, Any]],
        *,
        raw_records: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if not examples:
            return []
        previous = [dict(item) for item in self.records if _source_task(item) == task_name]
        abstractions = self._llm_abstractions(task_name, examples, raw_records or [], previous)
        fallback = _infer_abstractions(examples, raw_records or [])
        merged: list[dict[str, Any]] = []
        for role in ("main_policy", "failure_avoidance"):
            abstraction = abstractions.get(role) if abstractions else None
            if abstraction is None:
                abstraction = _previous_role(previous, role) or fallback[role]
            merged.append(
                {
                    "memory_id": _memory_id(task_name, role),
                    "memory_type": "abstract",
                    "abstract_role": role,
                    "internal_source": {"task_name": task_name},
                    **abstraction,
                }
            )
        self.records = [item for item in self.records if _source_task(item) != task_name]
        self.records.extend(merged)
        self._persist()
        return [dict(item) for item in merged]

    def _llm_abstractions(
        self,
        task_name: str,
        examples: list[dict[str, Any]],
        raw_records: list[dict[str, Any]],
        previous: list[dict[str, Any]],
    ) -> dict[str, dict[str, str]] | None:
        if self.mode not in {"llm", "ds", "deepseek"} or self.model is None:
            return None
        prompt = _render_abstraction_prompt(
            task_name,
            examples,
            raw_records,
            previous,
            max_chars=self.prompt_max_chars,
        )
        raw_output = self.model.generate(
            prompt,
            state={
                "memory_phase": "abstract_memory",
                "internal_sample_id": task_name,
                "max_new_tokens": self.max_new_tokens,
            },
        )
        decoded = _extract_json(raw_output)
        if not decoded:
            return None
        result: dict[str, dict[str, str]] = {}
        for role in ("main_policy", "failure_avoidance"):
            value = decoded.get(role)
            normalized = _normalize_abstraction(value) if isinstance(value, dict) else None
            if normalized is not None:
                result[role] = normalized
        return result or None

    def retrieval_candidates(self) -> list[dict[str, Any]]:
        return [
            {
                "memory_id": item.get("memory_id") or _memory_id(_source_task(item)),
                "memory_kind": "abstract",
                "abstract_role": item.get("abstract_role", "main_policy"),
                "guidance": item.get("guidance"),
                "applicable_when": item.get("applicable_when"),
                "output_form": _stored_output_form(item),
                "reusable_pattern": item.get("reusable_pattern", ""),
                "failure_mode": item.get("failure_mode", ""),
            }
            for item in self.records
        ]

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return [dict(item) for item in data if isinstance(item, dict)] if isinstance(data, list) else []

    def _persist(self) -> None:
        atomic_write_json(self.path, self.records)


def _infer_abstractions(
    examples: list[dict[str, Any]],
    raw_records: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    policies = [infer_output_form(str(item.get("question", ""))) for item in examples]
    formats = [str(policy["kind"]) for policy in policies]
    output_form = max(dict.fromkeys(formats), key=formats.count)
    has_image = any(item.get("image") or item.get("image_path") for item in examples)
    if output_form == "single_option_label":
        guidance = "Use the available evidence to choose one option and output only its letter."
        applicable = "The current input presents labelled answer options and asks for one choice."
    elif output_form == "numeric":
        guidance = "Solve the arithmetic internally and output only the final numeric value."
        applicable = "The current input asks for a computed count, number, or value."
    elif output_form == "short_text":
        guidance = "Ground the answer in the available evidence and output only the requested short text or value."
        applicable = "The current input explicitly requests a short word, phrase, or value."
    else:
        guidance = "Follow the visible output constraint and return only the final answer."
        applicable = "The current input requests a concise direct answer."
    if has_image:
        applicable += " Visual evidence is available."
    failures = [item for item in raw_records if not bool(item.get("response_is_correct", False))]
    format_failures = [item for item in failures if not bool(item.get("format_compliant", True))]
    if format_failures:
        failure_guidance = "Check the requested output form before returning the answer; remove unsupported explanation or wrappers."
        failure_mode = "A potentially correct result is invalidated by an output-format violation."
    elif failures:
        failure_guidance = "Before answering, verify every inference against the supplied evidence and re-check the final result."
        failure_mode = "An unsupported inference or unchecked intermediate step produces a wrong final answer."
    else:
        failure_guidance = "Preserve the requested output form and perform a brief consistency check before answering."
        failure_mode = "Correct reasoning can still fail through an unchecked final answer or formatting mismatch."
    return {
        "main_policy": {
            "guidance": guidance,
            "applicable_when": applicable,
            "output_form": output_form,
        },
        "failure_avoidance": {
            "guidance": failure_guidance,
            "applicable_when": "Use during final verification for inputs with similar reasoning or output constraints.",
            "output_form": output_form,
            "failure_mode": failure_mode,
        },
    }


def _render_abstraction_prompt(
    task_name: str,
    examples: list[dict[str, Any]],
    raw_records: list[dict[str, Any]],
    previous: list[dict[str, Any]],
    *,
    max_chars: int = 12000,
) -> str:
    visible_examples = [
        {
            "question": str(item.get("question", ""))[:3000],
            "answer_format_hint": infer_output_form(str(item.get("question", "")))["kind"],
        }
        for item in examples[-8:]
    ]
    interactions = [
        {
            "question": str(item.get("question", ""))[:2000],
            "previous_attempt": str(item.get("response", ""))[:1000],
            "correct_feedback": item.get("feedback"),
            "previous_attempt_correct": item.get("response_is_correct"),
        }
        for item in raw_records[-8:]
    ]
    payload = {
        "task_name_internal": task_name,
        "previous_abstract_memory": [
            {
                "role": str(item.get("abstract_role") or "main_policy"),
                "guidance": item.get("guidance"),
                "applicable_when": item.get("applicable_when"),
                "output_form": _stored_output_form(item),
                "reusable_pattern": item.get("reusable_pattern", ""),
                "failure_mode": item.get("failure_mode", ""),
            }
            for item in previous
        ],
        "examples": visible_examples,
        "recent_interactions": interactions,
    }
    instructions = [
        "You are consolidating abstract memory for a continual-learning harness.",
        "Maintain exactly two reusable, de-identified abstractions for this internal task.",
        "Merge previous_abstract_memory with the new evidence; do not simply overwrite useful prior knowledge.",
        "Do not include task names, dataset names, sample IDs, labels, gold-answer leakage, or exact answer text.",
        "Use feedback only to infer transferable behavior. Do not copy correct_feedback as an answer source.",
        "Prefer guidance about output format, failure avoidance, and reusable reasoning patterns.",
        "Return one valid JSON object only with keys main_policy and failure_avoidance.",
        "Each value is an object with guidance, applicable_when, output_form, reusable_pattern, and failure_mode.",
        "main_policy states the primary reusable reasoning/output policy.",
        "failure_avoidance states a concrete mistake pattern and how to prevent it.",
        "output_form must be one of single_option_label, numeric, short_text, list, structured_plan, solution_tagged, unspecified.",
        "Input evidence:",
    ]
    prefix = "\n".join(instructions) + "\n"
    original_prompt = prefix + json.dumps(payload, ensure_ascii=False, indent=2)
    bounded_prompt = _bounded_json_prompt(prefix, payload, max_chars=max_chars)
    if len(bounded_prompt) < len(original_prompt):
        progress(
            "abstract memory prompt bounded "
            f"task={task_name} original_chars={len(original_prompt)} "
            f"bounded_chars={len(bounded_prompt)} max_chars={max_chars}"
        )
    return bounded_prompt


def _bounded_json_prompt(prefix: str, payload: dict[str, Any], *, max_chars: int) -> str:
    """Bound evidence text while preserving a syntactically valid JSON payload."""

    limit = max(int(max_chars), len(prefix) + 256)
    cap = 3000
    while True:
        bounded_payload = _truncate_payload_strings(payload, max_chars=cap)
        prompt = prefix + json.dumps(bounded_payload, ensure_ascii=False, indent=2)
        if len(prompt) <= limit:
            return prompt
        if cap <= 64:
            for key in ("recent_interactions", "examples", "previous_abstract_memory"):
                records = bounded_payload.get(key)
                while isinstance(records, list) and records and len(prompt) > limit:
                    records.pop(0)
                    prompt = prefix + json.dumps(bounded_payload, ensure_ascii=False, indent=2)
            if len(prompt) <= limit:
                return prompt
            minimal_payload = {
                "task_name_internal": bounded_payload.get("task_name_internal", ""),
                "previous_abstract_memory": [],
                "examples": [],
                "recent_interactions": [],
            }
            return prefix + json.dumps(minimal_payload, ensure_ascii=False, indent=2)
        cap = max(int(cap * 0.75), 64)


def _truncate_payload_strings(value: Any, *, max_chars: int) -> Any:
    if isinstance(value, dict):
        return {
            key: _truncate_payload_strings(item, max_chars=max_chars)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_truncate_payload_strings(item, max_chars=max_chars) for item in value]
    if isinstance(value, str) and len(value) > max_chars:
        suffix = " ...[truncated]"
        return value[: max(max_chars - len(suffix), 1)] + suffix
    return value


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = str(text).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    attempts = [cleaned]
    start = cleaned.find("{")
    if start > 0:
        attempts.append(cleaned[start:])
    decoder = json.JSONDecoder()
    for candidate in attempts:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                parsed, _ = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _source_task(item: dict[str, Any]) -> str:
    internal = item.get("internal_source")
    if isinstance(internal, dict):
        return str(internal.get("task_name", ""))
    return str(item.get("task_name", ""))


def _stored_output_form(item: dict[str, Any]) -> str:
    if item.get("output_form"):
        return str(item["output_form"])
    return {
        "single_option_letter": "single_option_label",
        "numeric_answer": "numeric",
        "short_text_answer": "short_text",
        "short_answer": "unspecified",
    }.get(str(item.get("answer_format", "")), "unspecified")


def _normalize_abstraction(value: dict[str, Any]) -> dict[str, str] | None:
    guidance = str(value.get("guidance") or "").strip()
    applicable_when = str(value.get("applicable_when") or "").strip()
    if not guidance or not applicable_when:
        return None
    result = {
        "guidance": guidance[:1200],
        "applicable_when": applicable_when[:800],
        "output_form": (str(value.get("output_form") or "unspecified").strip() or "unspecified")[:80],
    }
    reusable_pattern = str(value.get("reusable_pattern") or "").strip()
    failure_mode = str(value.get("failure_mode") or "").strip()
    if reusable_pattern:
        result["reusable_pattern"] = reusable_pattern[:800]
    if failure_mode:
        result["failure_mode"] = failure_mode[:800]
    return result


def _previous_role(records: list[dict[str, Any]], role: str) -> dict[str, str] | None:
    for item in records:
        stored_role = str(item.get("abstract_role") or "main_policy")
        if stored_role == role:
            return _normalize_abstraction(item)
    return None


def _memory_id(task_name: str, role: str = "main_policy") -> str:
    digest = hashlib.sha256(f"{task_name}:{role}".encode("utf-8")).hexdigest()[:12]
    return f"abstract_{digest}"
