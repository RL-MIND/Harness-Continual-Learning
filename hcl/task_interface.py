from __future__ import annotations

import json
import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .cache_store import JsonObjectCache
from .progress import progress
from .prompt_templates import PromptTemplate


TASK_INTERFACE_DIR = Path(__file__).resolve().parent / "task_interface"
DEFAULT_STRUCTURING_TEMPLATE = TASK_INTERFACE_DIR / "structuring_prompt_default.json"

FORBIDDEN_VISIBLE_KEYS = (
    "task_name",
    "task_id",
    "task_type",
    "task_index",
    "source_task_id",
    "dataset",
    "dataset_index",
    "split",
)
FORBIDDEN_ASSERT_KEYS = tuple(
    key for key in FORBIDDEN_VISIBLE_KEYS if key != "dataset"
)
FORBIDDEN_TASK_LABELS = ("gsm8k", "mathverse", "arxivqa", "mathvista", "chartqa")
EVIDENCE_MODALITIES = {
    "text",
    "image",
    "document",
    "tool_output",
    "environment_observation",
    "user_feedback",
    "action_trace",
    "other",
}
EVIDENCE_SOURCES = {"user", "environment", "tool", "file", "prior_interaction", "unknown"}
GOAL_STATUSES = {"explicit", "inferred", "under_specified"}
CONSTRAINT_TYPES = {
    "output_format",
    "tool_condition",
    "environment_condition",
    "action_condition",
    "success_criterion",
    "resource_limit",
    "safety_condition",
    "user_preference",
    "other",
}
_ATTACHMENT_HASH_CACHE: dict[tuple[str, int, int], str] = {}


@dataclass
class TaskInterfaceConfig:
    mode: str = "auto"
    structuring_template_path: str = str(DEFAULT_STRUCTURING_TEMPLATE)
    cache_path: str | None = None
    max_new_tokens: int = 512
    invalid_output_retries: int = 1
    retry_max_new_tokens: int = 512
    fallback_to_rules_on_invalid: bool = False
    long_context_rule_threshold_chars: int = 0
    visible_context_max_chars: int = 0
    visible_context_max_items: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TaskInterface:
    """Build the minimal HCL interface I_t=(X_t, G_hat_t, K_hat_t).

    Dataset identity, gold labels, paths, and evaluator metadata remain in the
    internal runtime envelope. Only ``hcl_interface`` is model-visible.
    """

    def __init__(
        self,
        config: TaskInterfaceConfig | None = None,
        *,
        model: Any | None = None,
        structuring_template_content: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or TaskInterfaceConfig()
        if structuring_template_content is None:
            self.structuring_template = PromptTemplate.from_path(self.config.structuring_template_path)
        else:
            self.structuring_template = PromptTemplate.from_dict(
                structuring_template_content,
                template_id_fallback="candidate_task_interface_structuring",
            )
        self.model = model
        self._cache = _TaskInterfaceCache(self.config.cache_path)

    def build_taskinterfacechunk(self, example: dict[str, Any], stream_position: int) -> dict[str, Any]:
        mode = self.config.mode
        if mode not in {"rule", "llm", "auto"}:
            raise ValueError("TaskInterfaceConfig.mode must be one of: rule, llm, auto")
        visible_chars = self._raw_visible_input_chars(example)
        long_context_threshold = max(int(self.config.long_context_rule_threshold_chars), 0)
        if mode != "rule" and long_context_threshold and visible_chars > long_context_threshold:
            progress(
                f"task interface bypass mode=rule_long_context "
                f"task_id={example.get('task_id', '')} visible_chars={visible_chars} "
                f"threshold={long_context_threshold}"
            )
            return self._build_runtime_envelope(
                example,
                stream_position,
                hcl_interface=self._rule_hcl_interface(example),
                interface_mode="rule_long_context",
            )
        if mode == "llm":
            return self._build_chunk_with_llm(example, stream_position)
        if mode == "auto" and self.model is not None:
            return self._try_build_chunk_with_llm(example, stream_position)
        return self._build_chunk_with_rules(example, stream_position)

    def uses_llm_structuring(self, example: dict[str, Any]) -> bool:
        """Return whether this example can exercise the learnable structuring artifact."""
        mode = self.config.mode
        if mode == "rule":
            return False
        long_context_threshold = max(int(self.config.long_context_rule_threshold_chars), 0)
        if long_context_threshold and self._raw_visible_input_chars(example) > long_context_threshold:
            return False
        return mode == "llm" or (mode == "auto" and self.model is not None)

    def _build_chunk_with_rules(self, example: dict[str, Any], stream_position: int) -> dict[str, Any]:
        return self._build_runtime_envelope(
            example,
            stream_position,
            hcl_interface=self._rule_hcl_interface(example),
            interface_mode="rule",
        )

    def _try_build_chunk_with_llm(self, example: dict[str, Any], stream_position: int) -> dict[str, Any]:
        try:
            return self._build_chunk_with_llm(example, stream_position)
        except Exception as exc:
            progress(
                f"task interface fallback mode=rule task_id={example.get('task_id', '')} "
                f"reason={type(exc).__name__}:{exc}"
            )
            return self._build_runtime_envelope(
                example,
                stream_position,
                hcl_interface=self._rule_hcl_interface(example),
                interface_mode="rule_fallback",
            )

    def _build_chunk_with_llm(self, example: dict[str, Any], stream_position: int) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("TaskInterface mode llm requires a model.")
        cache_key = self._cache_key(example)
        cached = self._cache.get(cache_key)
        if cached is not None:
            assert_no_label_leakage(json.dumps(cached, ensure_ascii=False), phase="task_interface_cache")
            progress(
                f"task interface cache hit task_id={example.get('task_id', '')} "
                f"key={cache_key[:12]}"
            )
            return self._build_runtime_envelope(
                example,
                stream_position,
                hcl_interface=cached,
                interface_mode="llm_cache",
            )
        attempts = max(int(self.config.invalid_output_retries), 0) + 1
        diagnostics: list[str] = []
        hcl_interface: dict[str, Any] | None = None
        for attempt in range(attempts):
            is_retry = attempt > 0
            prompt = self._render_structuring_prompt(example, compact_retry=is_retry)
            token_limit = (
                self.config.retry_max_new_tokens if is_retry else self.config.max_new_tokens
            )
            raw_output = self.model.generate(
                prompt,
                state={
                    "task_interface_phase": "structure_task",
                    "task_interface_attempt": attempt + 1,
                    "internal_sample_id": example.get("task_id"),
                    "max_new_tokens": max(int(token_limit), 1),
                },
            )
            decoded, parse_error = _extract_json_with_diagnostics(raw_output)
            if parse_error is not None:
                diagnostics.append(f"attempt_{attempt + 1}:{parse_error}")
                continue
            try:
                hcl_interface = self._normalize_hcl_interface(decoded, example)
            except ValueError as exc:
                diagnostics.append(f"attempt_{attempt + 1}:schema_error:{exc}")
                continue
            break
        if hcl_interface is None:
            if self.config.fallback_to_rules_on_invalid:
                progress(
                    f"task interface fallback mode=rule task_id={example.get('task_id', '')} "
                    f"reason=invalid_llm_output attempts={attempts} "
                    f"diagnostics={'; '.join(diagnostics)}"
                )
                return self._build_runtime_envelope(
                    example,
                    stream_position,
                    hcl_interface=self._rule_hcl_interface(example),
                    interface_mode="rule_fallback",
                )
            raise ValueError(
                "Task Interface LLM output remained invalid after "
                f"{attempts} attempt(s): {'; '.join(diagnostics)}"
            )
        assert_no_label_leakage(
            json.dumps(hcl_interface, ensure_ascii=False),
            phase="task_interface_output",
        )
        self._cache.put(cache_key, hcl_interface)
        progress(
            f"task interface cache store task_id={example.get('task_id', '')} "
            f"key={cache_key[:12]}"
        )
        return self._build_runtime_envelope(
            example,
            stream_position,
            hcl_interface=hcl_interface,
            interface_mode="llm",
        )

    def _cache_key(self, example: dict[str, Any]) -> str:
        payload = {
            "schema": "hcl_task_interface_cache_v1",
            "visible_input": self._raw_visible_input(example),
            "attachments": _attachment_content_hashes(self._files(example)),
            "structuring_artifact": {
                "template_id": self.structuring_template.template_id,
                "description": self.structuring_template.description,
                "sections": self.structuring_template.sections,
            },
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _build_runtime_envelope(
        self,
        example: dict[str, Any],
        stream_position: int,
        *,
        hcl_interface: dict[str, Any],
        interface_mode: str,
    ) -> dict[str, Any]:
        question = str(example.get("question", ""))
        task_id = str(example.get("task_id") or f"internal_sample_{stream_position}")
        return {
            "hcl_interface": hcl_interface,
            "task_context": {
                "raw_question": question,
                "files": self._files(example),
                "capability_inputs": {
                    "text_candidates": _capability_text_candidates(example),
                },
            },
            "task_id": task_id,
            "question": question,
            "task_name": str(example.get("task_name", "")),
            "task_type": str(example.get("task_type", "")),
            "stream_position": stream_position,
            "metadata": {
                **dict(example.get("metadata", {})),
                "answer": example.get("answer"),
                "split": example.get("split", ""),
                "task_interface_mode": interface_mode,
            },
        }

    def _render_structuring_prompt(
        self,
        example: dict[str, Any],
        *,
        compact_retry: bool = False,
    ) -> str:
        prompt = self.structuring_template.render(
            {
                "raw_visible_input": json.dumps(
                    self._raw_visible_input(example),
                    ensure_ascii=False,
                    indent=2,
                )
            }
        )
        if compact_retry:
            prompt = "\n".join(
                (
                    prompt,
                    "[RETRY_AFTER_INVALID_OUTPUT]",
                    "The previous response was invalid or truncated. Return a much more compact object now.",
                    "Use one evidence item per raw visible observation, a summary of at most 12 words, "
                    "a goal of at most 20 words, and at most 4 genuine interaction constraints.",
                    "Do not explain, solve, repeat the schema, or enumerate problem facts as constraints.",
                    "Close every JSON string, array, and object.",
                )
            )
        assert_no_label_leakage(prompt, phase="task_interface_structuring")
        return prompt

    def _raw_visible_input(self, example: dict[str, Any]) -> dict[str, Any]:
        visible_context = self._bounded_visible_context(example)
        extra_example = dict(example)
        # visible_context has its own schema field; do not duplicate it as an
        # additional observation in structuring and cache-key payloads.
        extra_example["visible_context"] = ""
        return {
            "text": str(example.get("question", "")),
            "options": example.get("options", []),
            "visible_context": visible_context,
            "additional_observations": [
                {"modality": modality, "content": content, "source": source}
                for modality, content, source in _additional_visible_observations(extra_example)
            ],
            "attachments": [
                {
                    "modality": str(item.get("type", "file")),
                    "content": f"A {item.get('type', 'file')} is available as part of the current observation.",
                    "source": "file",
                }
                for item in self._files(example)
            ],
        }

    def _raw_visible_input_chars(self, example: dict[str, Any]) -> int:
        return len(
            json.dumps(
                self._raw_visible_input(example),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    def _rule_hcl_interface(self, example: dict[str, Any]) -> dict[str, Any]:
        bounded_example = dict(example)
        bounded_example["visible_context"] = self._bounded_visible_context(example)
        evidence = _evidence_from_example(bounded_example)
        goal = _recognizable_goal(str(example.get("question", "")), evidence)
        constraints = _recognizable_constraints(str(example.get("question", "")), evidence)
        return {
            "evidence": evidence,
            "recognizable_goal": goal,
            "recognizable_constraints": constraints,
        }

    def _bounded_visible_context(self, example: dict[str, Any]) -> object:
        return _prefilter_visible_context(
            example.get("visible_context", ""),
            question=str(example.get("question", "")),
            max_chars=max(int(self.config.visible_context_max_chars), 0),
            max_items=max(int(self.config.visible_context_max_items), 0),
        )

    def _normalize_hcl_interface(
        self,
        decoded: dict[str, Any],
        example: dict[str, Any],
    ) -> dict[str, Any]:
        required_keys = {"evidence", "recognizable_goal", "recognizable_constraints"}
        if set(decoded) != required_keys:
            raise ValueError(
                "Task Interface output must contain exactly evidence, recognizable_goal, "
                "and recognizable_constraints."
            )
        fallback = self._rule_hcl_interface(example)
        raw_evidence = decoded.get("evidence")
        evidence = _normalize_evidence(raw_evidence, fallback["evidence"])
        evidence = _preserve_observed_evidence(evidence, fallback["evidence"])
        valid_ids = {
            str(item.get("id", ""))
            for item in evidence.get("items", [])
            if isinstance(item, dict)
        }
        fallback_goal = _remap_supporting_evidence(
            fallback["recognizable_goal"],
            fallback["evidence"],
            evidence,
        )
        fallback_constraints = [
            _remap_supporting_evidence(item, fallback["evidence"], evidence)
            for item in fallback["recognizable_constraints"]
        ]
        goal = _normalize_goal(decoded.get("recognizable_goal"), fallback_goal, valid_ids)
        constraints = _normalize_constraints(
            decoded.get("recognizable_constraints"),
            fallback_constraints,
            valid_ids,
        )
        return {
            "evidence": evidence,
            "recognizable_goal": goal,
            "recognizable_constraints": constraints,
        }

    @staticmethod
    def to_model_visible(taskinterfacechunk: dict[str, Any]) -> dict[str, Any]:
        """Return exactly the minimal HCL interface, with no runtime metadata."""
        hcl_interface = taskinterfacechunk.get("hcl_interface")
        if not isinstance(hcl_interface, dict):
            return _empty_hcl_interface()
        return {
            "evidence": dict(hcl_interface.get("evidence", {})),
            "recognizable_goal": dict(hcl_interface.get("recognizable_goal", {})),
            "recognizable_constraints": [
                dict(item)
                for item in hcl_interface.get("recognizable_constraints", [])
                if isinstance(item, dict)
            ],
        }

    @staticmethod
    def visible_question(taskinterfacechunk: dict[str, Any]) -> str:
        visible = TaskInterface.to_model_visible(taskinterfacechunk)
        evidence = visible.get("evidence", {})
        items = evidence.get("items", []) if isinstance(evidence, dict) else []
        for item in items:
            if isinstance(item, dict) and item.get("modality") == "text":
                return str(item.get("content", ""))
        return ""

    def _files(self, example: dict[str, Any]) -> list[dict[str, str]]:
        files: list[dict[str, str]] = []
        image = example.get("image") or example.get("image_path")
        if image:
            files.append({"type": "image", "path": str(image)})
        file_path = example.get("file") or example.get("file_path")
        if file_path:
            files.append({"type": "document", "path": str(file_path)})
        return files


def _capability_text_candidates(example: dict[str, Any]) -> list[dict[str, str]]:
    """Preserve visible answer options as task-local matcher inputs."""
    options = example.get("options")
    candidates: list[dict[str, str]] = []
    if isinstance(options, dict):
        for key, value in options.items():
            text = str(value).strip()
            if text:
                candidates.append({"id": str(key), "text": text})
    elif isinstance(options, list):
        for index, value in enumerate(options):
            if isinstance(value, dict):
                candidate_id = str(value.get("id") or value.get("label") or index)
                text = str(value.get("text") or value.get("content") or value.get("value") or "").strip()
            else:
                candidate_id = str(index)
                text = str(value).strip()
            if text:
                candidates.append({"id": candidate_id, "text": text})
    # COCO's public category ontology is part of the task definition, not a
    # sample label.  Supplying it makes image/text capability matching usable
    # without exposing which categories or boxes are present in this example.
    if not candidates and str(example.get("task_type", "")).lower() == "coco_detection":
        categories = (
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
            "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
            "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
            "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
            "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
        )
        candidates = [{"id": category, "text": category} for category in categories]
    return candidates


class _TaskInterfaceCache(JsonObjectCache):
    """Append-only cache of successful, normalized Task Interface outputs."""

    def __init__(self, path: str | None) -> None:
        super().__init__(path, value_field="hcl_interface")


def _attachment_content_hashes(files: list[dict[str, str]]) -> list[dict[str, str]]:
    fingerprints: list[dict[str, str]] = []
    for item in files:
        path = Path(str(item.get("path", "")))
        digest = hashlib.sha256()
        status = "unavailable"
        try:
            stat = path.stat()
            memo_key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
            status = _ATTACHMENT_HASH_CACHE.get(memo_key, "")
            if not status:
                with path.open("rb") as f:
                    for block in iter(lambda: f.read(1024 * 1024), b""):
                        digest.update(block)
                status = digest.hexdigest()
                _ATTACHMENT_HASH_CACHE[memo_key] = status
        except OSError:
            pass
        fingerprints.append({"type": str(item.get("type", "file")), "content_sha256": status})
    return fingerprints


def infer_output_form(question: str) -> dict[str, Any]:
    """Infer a visible output constraint; this is not part of the interface schema."""
    text = str(question)
    option_letters: list[str] = []
    for match in re.finditer(r"(?:^|\n)\s*\(?([A-Ea-e])\)?\s*[\.:\)]\s+", text):
        letter = match.group(1).upper()
        if letter not in option_letters:
            option_letters.append(letter)
    lower = text.lower()
    if "json" in lower and "bbox" in lower:
        coordinate_text = (
            " using the 0-1000 image coordinate system"
            if "0-1000" in lower or "0 to 1000" in lower
            else ""
        )
        return {
            "kind": "json_bbox" if "object" not in lower and "label" not in lower else "json_objects",
            "valid_outputs": [],
            "constraint": (
                "The response should be valid JSON only, using bbox coordinates in "
                f"[x1, y1, x2, y2] format{coordinate_text}."
            ),
        }
    if "caption" in lower:
        return {
            "kind": "caption",
            "valid_outputs": [],
            "constraint": "The response should be one short natural-language image caption, without explanation.",
        }
    explicitly_requests_option = "option letter" in lower or "选项字母" in lower
    if len(option_letters) >= 2 or explicitly_requests_option:
        if len(option_letters) < 2:
            example_match = re.search(
                r"(?:e\.g\.|例如)[,，:]?\s*([A-E](?:\s*[,，/]\s*[A-E]){1,4})",
                text,
                flags=re.IGNORECASE,
            )
            if example_match:
                option_letters = re.findall(r"[A-E]", example_match.group(1).upper())
        if len(option_letters) < 2:
            option_letters = ["A", "B", "C", "D"]
        return {
            "kind": "single_option_label",
            "valid_outputs": option_letters,
            "constraint": f"The response should be exactly one visible option label from {option_letters}, without explanation.",
        }
    if "single word or phrase" in lower or "short answer" in lower:
        return {
            "kind": "short_text",
            "valid_outputs": [],
            "constraint": "The response should contain only the requested short word, phrase, or value, without explanation.",
        }
    numeric_cues = ("how many", "what number", "calculate", "compute", "find the value", "多少", "几个")
    if any(cue in lower for cue in numeric_cues):
        return {
            "kind": "numeric",
            "valid_outputs": [],
            "constraint": "The response should contain only the final plain numeric value, without explanation.",
        }
    return {"kind": "unspecified", "valid_outputs": [], "constraint": ""}


def _evidence_from_example(example: dict[str, Any]) -> dict[str, Any]:
    items: list[dict[str, str]] = []
    question = str(example.get("question", "")).strip()
    if question:
        items.append({"id": "x1", "modality": "text", "content": question, "source": "user"})
    for modality, content, source in _additional_visible_observations(example):
        items.append(
            {
                "id": f"x{len(items) + 1}",
                "modality": modality,
                "content": content,
                "source": source,
            }
        )
    if example.get("image") or example.get("image_path"):
        items.append(
            {
                "id": f"x{len(items) + 1}",
                "modality": "image",
                "content": "An image is available as part of the current observation.",
                "source": "file",
            }
        )
    if example.get("file") or example.get("file_path"):
        items.append(
            {
                "id": f"x{len(items) + 1}",
                "modality": "document",
                "content": "A document is available as part of the current observation.",
                "source": "file",
            }
        )
    modalities = [item["modality"] for item in items]
    summary = "No visible evidence is available."
    if modalities:
        summary = f"The current observation contains visible evidence with modalities: {', '.join(modalities)}."
    return {"items": items, "summary": summary}


def _additional_visible_observations(example: dict[str, Any]) -> list[tuple[str, str, str]]:
    observations: list[tuple[str, str, str]] = []
    options = example.get("options")
    if options not in (None, [], {}):
        observations.append(("text", f"Visible answer options: {_visible_value(options)}", "user"))
    visible_context = example.get("visible_context")
    if visible_context not in (None, "", [], {}):
        observations.append(("environment_observation", _visible_value(visible_context), "environment"))
    for key, modality, source in (
        ("tool_output", "tool_output", "tool"),
        ("environment_observation", "environment_observation", "environment"),
        ("user_feedback", "user_feedback", "user"),
        ("action_trace", "action_trace", "prior_interaction"),
    ):
        value = example.get(key)
        if value is not None and str(value).strip():
            observations.append((modality, str(value), source))
    return observations


def _visible_value(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _prefilter_visible_context(
    value: object,
    *,
    question: str,
    max_chars: int,
    max_items: int,
) -> object:
    """Select a bounded, query-connected subset of long paragraph lists."""
    if not isinstance(value, list) or (max_chars <= 0 and max_items <= 0):
        return value
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    item_limit = max_items if max_items > 0 else len(value)
    if len(serialized) <= max_chars or max_chars <= 0:
        return value[:item_limit]
    candidates = [
        {
            "index": index,
            "item": item,
            "text": _visible_value(item),
            "tokens": _context_tokens(_visible_value(item)),
            "title_tokens": _context_title_tokens(item),
        }
        for index, item in enumerate(value)
    ]
    question_tokens = _context_tokens(question)
    selected: list[dict[str, Any]] = []
    selected_chars = 2
    while candidates and len(selected) < item_limit:
        selected_body_tokens = set().union(*(item["tokens"] for item in selected)) if selected else set()

        def score(candidate: dict[str, Any]) -> tuple[float, float, int]:
            query_overlap = _token_overlap(question_tokens, candidate["tokens"])
            bridge_overlap = _token_overlap(candidate["title_tokens"], selected_body_tokens)
            return query_overlap + 1.5 * bridge_overlap, bridge_overlap, -int(candidate["index"])

        best = max(candidates, key=score)
        item_chars = len(json.dumps(best["item"], ensure_ascii=False, sort_keys=True)) + 1
        candidates.remove(best)
        if selected and selected_chars + item_chars > max_chars:
            continue
        selected.append(best)
        selected_chars += item_chars
        if selected_chars >= max_chars:
            break
    return [item["item"] for item in selected]


def _context_tokens(text: str) -> set[str]:
    stopwords = {
        "about",
        "after",
        "answer",
        "before",
        "context",
        "did",
        "does",
        "from",
        "only",
        "provided",
        "return",
        "shortest",
        "that",
        "the",
        "this",
        "using",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", str(text).lower())
        if len(token) > 1 and token not in stopwords
    }


def _context_title_tokens(item: object) -> set[str]:
    if not isinstance(item, dict):
        return set()
    title = item.get("title") or item.get("name") or ""
    return _context_tokens(str(title))


def _token_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left), 1)


def _recognizable_goal(question: str, evidence: dict[str, Any]) -> dict[str, Any]:
    text = str(question).strip()
    text_id = _first_evidence_id(evidence, "text")
    if not text:
        return {
            "status": "under_specified",
            "content": "The current evidence does not provide a complete executable goal.",
            "supporting_evidence": [],
        }
    output_form = infer_output_form(text)
    if output_form["kind"] == "single_option_label":
        content = "Select the option supported by the visible evidence."
    elif output_form["kind"] in {"json_bbox", "json_objects"}:
        content = "Return the requested visual localization result in valid JSON."
    elif output_form["kind"] == "caption":
        content = "Write a concise caption grounded in the visible image."
    elif output_form["kind"] == "numeric":
        content = "Compute the requested final numeric value from the visible evidence."
    elif output_form["kind"] == "short_text":
        content = "Provide the short answer requested by the visible evidence."
    else:
        content = "Answer the request expressed in the visible text evidence."
    explicit_cues = ("?", "please", "answer", "calculate", "compute", "find", "what", "which", "how", "请", "求", "多少", "哪个")
    status = "explicit" if any(cue in text.lower() for cue in explicit_cues) else "inferred"
    return {"status": status, "content": content, "supporting_evidence": [text_id] if text_id else []}


def _recognizable_constraints(question: str, evidence: dict[str, Any]) -> list[dict[str, Any]]:
    output_form = infer_output_form(question)
    if output_form["kind"] == "unspecified":
        return []
    text_id = _first_evidence_id(evidence, "text")
    return [
        {
            "id": "k1",
            "content": output_form["constraint"],
            "constraint_type": "output_format",
            "supporting_evidence": [text_id] if text_id else [],
        }
    ]


def _normalize_evidence(raw: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict) or not isinstance(raw.get("items"), list):
        return {"items": [dict(item) for item in fallback["items"]], "summary": fallback["summary"]}
    items: list[dict[str, str]] = []
    used_ids: set[str] = set()
    for index, item in enumerate(raw["items"]):
        if not isinstance(item, dict) or not str(item.get("content", "")).strip():
            continue
        item_id = str(item.get("id") or f"x{index + 1}")
        if item_id in used_ids:
            item_id = f"x{len(items) + 1}"
        used_ids.add(item_id)
        modality = str(item.get("modality", "other"))
        source = str(item.get("source", "unknown"))
        items.append(
            {
                "id": item_id,
                "modality": modality if modality in EVIDENCE_MODALITIES else "other",
                "content": str(item.get("content", "")),
                "source": source if source in EVIDENCE_SOURCES else "unknown",
            }
        )
    return {"items": items, "summary": str(raw.get("summary") or fallback["summary"])}


def _preserve_observed_evidence(evidence: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    items = [dict(item) for item in evidence.get("items", []) if isinstance(item, dict)]
    signatures = {(item.get("modality"), item.get("content")) for item in items}
    for observed in fallback.get("items", []):
        signature = (observed.get("modality"), observed.get("content"))
        if signature in signatures:
            continue
        item = dict(observed)
        item["id"] = _next_id("x", {str(current.get("id", "")) for current in items})
        items.append(item)
        signatures.add(signature)
    return {"items": items, "summary": str(evidence.get("summary") or fallback["summary"])}


def _normalize_goal(raw: Any, fallback: dict[str, Any], valid_ids: set[str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return dict(fallback)
    status = str(raw.get("status", "under_specified"))
    if status not in GOAL_STATUSES:
        status = "under_specified"
    supporting = _valid_supporting_ids(raw.get("supporting_evidence"), valid_ids)
    content = str(raw.get("content", "")).strip()
    if not content:
        return dict(fallback)
    if status == "under_specified" and fallback.get("status") == "explicit":
        return dict(fallback)
    return {"status": status, "content": content, "supporting_evidence": supporting}


def _normalize_constraints(raw: Any, fallback: list[dict[str, Any]], valid_ids: set[str]) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return [dict(item) for item in fallback]
    constraints: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or not str(item.get("content", "")).strip():
            continue
        constraint_id = str(item.get("id") or f"k{index + 1}")
        if constraint_id in used_ids:
            constraint_id = _next_id("k", used_ids)
        used_ids.add(constraint_id)
        constraint_type = str(item.get("constraint_type", "other"))
        constraints.append(
            {
                "id": constraint_id,
                "content": str(item.get("content", "")),
                "constraint_type": constraint_type if constraint_type in CONSTRAINT_TYPES else "other",
                "supporting_evidence": _valid_supporting_ids(item.get("supporting_evidence"), valid_ids),
            }
        )
    existing = {
        (item.get("constraint_type"), str(item.get("content", "")).strip().lower())
        for item in constraints
    }
    for fallback_item in fallback:
        if fallback_item.get("constraint_type") == "output_format":
            constraints = [
                item
                for item in constraints
                if item.get("constraint_type") != "output_format"
            ]
            used_ids = {str(item.get("id", "")) for item in constraints}
            existing = {
                (item.get("constraint_type"), str(item.get("content", "")).strip().lower())
                for item in constraints
            }
        signature = (
            fallback_item.get("constraint_type"),
            str(fallback_item.get("content", "")).strip().lower(),
        )
        if signature in existing:
            continue
        preserved = dict(fallback_item)
        preserved["id"] = _next_id("k", used_ids)
        used_ids.add(str(preserved["id"]))
        constraints.append(preserved)
        existing.add(signature)
    return constraints


def _valid_supporting_ids(raw: Any, valid_ids: set[str]) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if str(item) in valid_ids]


def _remap_supporting_evidence(
    item: dict[str, Any],
    source_evidence: dict[str, Any],
    target_evidence: dict[str, Any],
) -> dict[str, Any]:
    source_by_id = {
        str(source.get("id", "")): (source.get("modality"), source.get("content"))
        for source in source_evidence.get("items", [])
        if isinstance(source, dict)
    }
    target_by_signature = {
        (target.get("modality"), target.get("content")): str(target.get("id", ""))
        for target in target_evidence.get("items", [])
        if isinstance(target, dict)
    }
    remapped = dict(item)
    remapped["supporting_evidence"] = [
        target_by_signature[source_by_id[source_id]]
        for source_id in item.get("supporting_evidence", [])
        if source_id in source_by_id and source_by_id[source_id] in target_by_signature
    ]
    return remapped


def _first_evidence_id(evidence: dict[str, Any], modality: str) -> str:
    for item in evidence.get("items", []):
        if isinstance(item, dict) and item.get("modality") == modality:
            return str(item.get("id", ""))
    return ""


def _next_id(prefix: str, used_ids: set[str]) -> str:
    index = 1
    while f"{prefix}{index}" in used_ids:
        index += 1
    return f"{prefix}{index}"


def _empty_hcl_interface() -> dict[str, Any]:
    return {
        "evidence": {"items": [], "summary": "No visible evidence is available."},
        "recognizable_goal": {
            "status": "under_specified",
            "content": "The current evidence does not provide a complete executable goal.",
            "supporting_evidence": [],
        },
        "recognizable_constraints": [],
    }


def assert_no_label_leakage(prompt: str, *, phase: str) -> None:
    text = str(prompt)
    leaked: list[str] = []
    for key in FORBIDDEN_ASSERT_KEYS:
        # Runtime metadata is serialized by json.dumps as an unescaped quoted
        # object key (for example, `"split": "train"`).  Restrict the guard to
        # that representation so ordinary model-visible prose such as
        # `split: 12 and 8` and escaped JSON snippets inside evidence strings do
        # not trigger a false-positive leakage failure.
        key_pattern = rf"(?<!\\)[\"']{re.escape(key)}[\"']\s*:"
        if re.search(key_pattern, text, flags=re.IGNORECASE):
            leaked.append(key)
    for label in FORBIDDEN_TASK_LABELS:
        label_pattern = rf"(?<![a-z0-9_]){re.escape(label)}(?:[_-]\d+)?(?![a-z0-9_])"
        if re.search(label_pattern, text, flags=re.IGNORECASE):
            leaked.append(label)
    if leaked:
        raise ValueError(f"Label leakage in {phase} prompt: {leaked}")


def sanitize_label_leakage_for_visible_prompt(value: Any) -> Any:
    """Return a model-visible copy with internal labels and metadata removed."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for raw_key, raw_child in value.items():
            key = str(raw_key)
            if key in FORBIDDEN_VISIBLE_KEYS or key in {"internal_source", "metadata", "_anchor_meta"}:
                continue
            cleaned[raw_key] = sanitize_label_leakage_for_visible_prompt(raw_child)
        return cleaned
    if isinstance(value, list):
        return [sanitize_label_leakage_for_visible_prompt(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_label_leakage_for_visible_prompt(item) for item in value)
    if isinstance(value, str):
        text = value
        for label in FORBIDDEN_TASK_LABELS:
            label_pattern = rf"(?<![a-z0-9_]){re.escape(label)}(?:[_-]\d+)?(?![a-z0-9_])"
            text = re.sub(label_pattern, "benchmark", text, flags=re.IGNORECASE)
        return text
    return value


def _extract_json_with_diagnostics(text: str) -> tuple[dict[str, Any], str | None]:
    cleaned = str(text).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as direct_error:
        start = cleaned.find("{")
        if start < 0:
            return {}, "json_decode_error:no_object_start"
        try:
            data, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        except json.JSONDecodeError as extracted_error:
            likely_truncated = _looks_truncated(cleaned)
            kind = "likely_truncated_json" if likely_truncated else "json_decode_error"
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
    if not stripped:
        return True
    return not stripped.endswith("}") or stripped.count("{") > stripped.count("}")
