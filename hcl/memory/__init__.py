from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

from ..evaluator import answer_format_compliant, answer_matches
from ..skills import EvolvingSkillMemory
from ..task_interface import infer_output_form
from .abstract import AbstractMemory
from .anchor import AnchorMemory
from .raw import RawMemory


DEFAULT_MEMORY_DIR = Path(__file__).resolve().parent


@dataclass
class MemoryConfig:
    enabled: bool = True
    record_dir: str = str(DEFAULT_MEMORY_DIR)
    raw_capacity_per_task: int = 15
    abstract_capacity_per_task: int = 2
    anchor_capacity_per_task: int = 15
    retrieval_top_k: int = 2
    retrieval_candidate_limit: int = 12
    reset_on_start: bool = True
    abstract_mode: str = "rule"
    abstract_max_new_tokens: int = 512
    abstract_prompt_max_chars: int = 12000
    anchor_selection_mode: str = "llm"
    anchor_count_per_batch: int = 1
    anchor_candidate_limit: int = 0
    anchor_selection_max_new_tokens: int = 1024
    anchor_similarity_threshold: float = 0.82
    skill_enabled: bool = True
    skill_mode: str = "llm"
    skill_max_count: int = 6
    skill_max_new_tokens: int = 1536
    skill_min_source_tasks: int = 2
    skill_invalid_output_retries: int = 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ExperienceMemory:
    """Minimal paper-aligned memory: raw evidence, abstractions, and anchors."""

    def __init__(self, config: MemoryConfig | None = None, *, model: Any | None = None) -> None:
        self.config = config or MemoryConfig()
        if int(self.config.abstract_capacity_per_task) != 2:
            raise ValueError("abstract_capacity_per_task must be 2: main_policy + failure_avoidance")
        self.model = model
        root = Path(self.config.record_dir)
        self.raw = RawMemory(root / "raw_memory.jsonl", capacity_per_task=self.config.raw_capacity_per_task)
        self.abstract = AbstractMemory(
            root / "abstract_memory.json",
            model=model,
            mode=self.config.abstract_mode,
            max_new_tokens=self.config.abstract_max_new_tokens,
            prompt_max_chars=self.config.abstract_prompt_max_chars,
        )
        self.anchor = AnchorMemory(
            root / "anchors.jsonl",
            capacity_per_task=self.config.anchor_capacity_per_task,
            similarity_threshold=self.config.anchor_similarity_threshold,
        )
        self.skills = EvolvingSkillMemory(
            root / "skills.json",
            model=model,
            mode=self.config.skill_mode,
            max_skills=self.config.skill_max_count,
            max_new_tokens=self.config.skill_max_new_tokens,
            min_source_tasks=self.config.skill_min_source_tasks,
            invalid_output_retries=self.config.skill_invalid_output_retries,
        )
        self._last_retrieval_stats: dict[str, int] = {"pool_count": 0, "candidate_count": 0}
        if self.config.reset_on_start:
            self.clear()

    def clear(self) -> None:
        self.raw.clear()
        self.abstract.clear()
        self.anchor.clear()
        self.skills.clear()

    def record_training_interaction(
        self,
        task_name: str,
        example: dict[str, Any],
        prediction: dict[str, Any],
    ) -> None:
        """Write one completed train interaction; never call this from evaluation."""
        if self.config.enabled:
            self.raw.add(task_name, [example], [prediction])

    def consolidate_task(self, task_name: str, train_examples: list[dict[str, Any]]) -> dict[str, object]:
        """Merge two abstractions and evolve skills after a train batch."""
        if self.config.enabled:
            task_records = [
                item
                for item in self.raw.records
                if isinstance(item.get("internal_source"), dict)
                and item["internal_source"].get("task_name") == task_name
                and not item.get("is_anchor")
            ]
            updated = self.abstract.consolidate(task_name, train_examples, raw_records=task_records)
            skill_update: dict[str, object] = {
                "updated": False,
                "skill_count": len(self.skills.records),
                "reason": "skill_memory_disabled",
            }
            if self.config.skill_enabled:
                skill_update = self.skills.evolve(self.abstract.records)
            return {"abstract_count": len(updated), "skill_update": skill_update}
        return {"abstract_count": 0, "skill_update": {"updated": False, "reason": "memory_disabled"}}

    def finalize_task(self, task_name: str, anchor_examples: list[dict[str, Any]]) -> None:
        """Freeze regression anchors only after candidate optimization is complete."""
        if not self.config.enabled:
            return
        self.anchor.add(task_name, anchor_examples)

    def batch_anchor_selection_enabled(self) -> bool:
        mode = str(self.config.anchor_selection_mode or "split").lower()
        return mode not in {"off", "none", "disabled", "split", "static"}

    def select_batch_anchors(
        self,
        task_name: str,
        examples: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
        *,
        split: str,
        batch_index: int,
        batch_count: int,
    ) -> dict[str, Any]:
        if not self.config.enabled or not self.batch_anchor_selection_enabled():
            return {"selected": [], "reason": "batch_anchor_selection_disabled"}
        target_count = self._anchor_target_count(len(examples))
        if target_count <= 0:
            return {"selected": [], "reason": "anchor_target_count_zero"}
        all_candidates = _anchor_candidate_views(examples, predictions)
        if not all_candidates:
            return {"selected": [], "reason": "empty_anchor_candidate_pool"}
        candidates = _limit_anchor_candidates(
            all_candidates,
            limit=int(self.config.anchor_candidate_limit or 0),
            target_count=target_count,
        )
        mode = str(self.config.anchor_selection_mode or "llm").lower()
        selected_ids: list[str] = []
        raw_output = ""
        diagnostics: list[str] = []
        if mode in {"llm", "model", "ds", "deepseek"} and self.model is not None:
            prompt = _render_anchor_selection_prompt(
                task_name=task_name,
                split=split,
                batch_index=batch_index,
                batch_count=batch_count,
                target_count=target_count,
                candidates=candidates,
            )
            raw_output = self.model.generate(
                prompt,
                state={
                    "memory_phase": "anchor_selection",
                    "internal_sample_id": f"{task_name}:batch_{batch_index + 1}",
                    "max_new_tokens": max(int(self.config.anchor_selection_max_new_tokens), 1),
                },
            )
            selected_ids, diagnostics = _parse_anchor_selection(raw_output, candidates, target_count)
        if len(selected_ids) < target_count:
            selected_ids = _fill_anchor_selection(selected_ids, candidates, target_count)
            if not diagnostics:
                diagnostics.append("rule_fallback_or_fill")
        by_candidate_id = {str(item.get("task_id", "")): item for item in all_candidates}
        selected_examples = [
            {
                **dict(example),
                "_anchor_meta": {
                    "correct": bool(by_candidate_id.get(str(example.get("task_id", "")), {}).get("correct", False)),
                    "format_compliant": bool(
                        by_candidate_id.get(str(example.get("task_id", "")), {}).get("format_compliant", False)
                    ),
                    "selected_batch": batch_index + 1,
                },
            }
            for example in examples
            if str(example.get("task_id", "")) in set(selected_ids)
        ]
        added = self.anchor.add_batch(
            task_name,
            selected_examples,
            max_total_per_task=self.config.anchor_capacity_per_task,
        )
        return {
            "selected": [
                {
                    "task_id": item.get("task_id"),
                    "source_task_id": item.get("source_task_id"),
                    "task_name": item.get("task_name"),
                    "split": item.get("split"),
                }
                for item in selected_examples
            ],
            "added_count": len(added),
            "target_count": target_count,
            "candidate_count": len(candidates),
            "original_candidate_count": len(all_candidates),
            "anchor_candidate_limit": int(self.config.anchor_candidate_limit or 0),
            "selection_mode": mode,
            "diagnostics": diagnostics,
            "raw_output": raw_output,
            "anchor_summary": {"anchor": len(self.anchor.records)},
        }

    def _anchor_target_count(self, batch_size: int) -> int:
        if batch_size <= 0:
            return 0
        count = max(int(self.config.anchor_count_per_batch), 0)
        return min(count, batch_size)

    def retrieval_candidates(
        self,
        *,
        current_input: dict[str, Any] | None = None,
        workflow_decision: dict[str, object] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.config.enabled:
            return []
        pool = self.abstract.retrieval_candidates() + self.raw.retrieval_candidates()
        limit = max(int(self.config.retrieval_candidate_limit), 0)
        candidates = _prefilter_memory_candidates(
            pool,
            current_input=current_input or {},
            workflow_decision=workflow_decision or {},
            limit=limit,
        )
        self._last_retrieval_stats = {
            "pool_count": len(pool),
            "candidate_count": len(candidates),
        }
        return candidates

    def retrieval_stats(self) -> dict[str, int]:
        return dict(self._last_retrieval_stats)

    def available_skill_views(self) -> list[dict[str, object]]:
        if not self.config.enabled or not self.config.skill_enabled:
            return []
        return self.skills.available_skill_views()

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        """Capture an update candidate so it can be rejected without losing stable memory."""
        from copy import deepcopy

        return {
            "raw": deepcopy(self.raw.records),
            "abstract": deepcopy(self.abstract.records),
            "anchor": deepcopy(self.anchor.records),
            "skills": deepcopy(self.skills.records),
        }

    def restore(self, snapshot: dict[str, list[dict[str, Any]]]) -> None:
        from copy import deepcopy

        self.raw.records = deepcopy(snapshot.get("raw", []))
        self.abstract.records = deepcopy(snapshot.get("abstract", []))
        self.anchor.records = deepcopy(snapshot.get("anchor", []))
        self.skills.records = deepcopy(snapshot.get("skills", []))
        self.raw._persist()
        self.abstract._persist()
        self.anchor._persist()
        self.skills._persist()

    def historical_anchors(self) -> list[dict[str, Any]]:
        return self.anchor.all_examples() if self.config.enabled else []

    def summary(self) -> dict[str, int]:
        return {
            "raw": len(self.raw.records),
            "abstract": len(self.abstract.records),
            "anchor": len(self.anchor.records),
            "skill": len(self.skills.records),
        }


__all__ = ["AbstractMemory", "AnchorMemory", "ExperienceMemory", "MemoryConfig", "RawMemory"]


def _anchor_candidate_views(
    examples: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {str(item.get("task_id", "")): item for item in predictions}
    candidates: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        task_id = str(example.get("task_id", ""))
        prediction = by_id.get(task_id, {})
        predicted_answer = prediction.get("answer")
        target = example.get("answer")
        correct = answer_matches(predicted_answer, target, example=example)
        format_ok = answer_format_compliant(predicted_answer, target)
        candidates.append(
            {
                "task_id": task_id,
                "index": index,
                "question": str(example.get("question", ""))[:400],
                "gold_answer": target,
                "prediction": predicted_answer,
                "correct": correct,
                "format_compliant": format_ok,
                "output_chars": len(str(predicted_answer or "")),
                "task_type": str(example.get("task_type", "")),
            }
        )
    return candidates


def _render_anchor_selection_prompt(
    *,
    task_name: str,
    split: str,
    batch_index: int,
    batch_count: int,
    target_count: int,
    candidates: list[dict[str, Any]],
) -> str:
    payload = {
        "task_name_internal": task_name,
        "split": split,
        "batch_index": batch_index,
        "batch_count": batch_count,
        "target_count": target_count,
        "candidate_examples": candidates,
    }
    return "\n".join(
        [
            "You are selecting anchor memory for a continual-learning harness.",
            "Choose training examples that should be preserved for future regression checks.",
            "Use the previous training performance: correctness, format compliance, output length, and task diversity.",
            "Prioritize failures, format violations, brittle edge cases, and representative examples.",
            "Do not select by dataset identity alone. Do not invent task_ids.",
            f"Select exactly {target_count} task_id(s) unless fewer candidates exist.",
            "Return one valid JSON object only, without markdown.",
            "Schema: {\"selected_task_ids\":[\"...\"],\"rationale\":\"...\"}",
            "Input:",
            json.dumps(payload, ensure_ascii=False, indent=2),
        ]
    )


def _limit_anchor_candidates(
    candidates: list[dict[str, Any]],
    *,
    limit: int,
    target_count: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or len(candidates) <= limit:
        return list(candidates)
    effective_limit = max(limit, target_count)
    return _rank_anchor_candidates(candidates)[:effective_limit]


def _parse_anchor_selection(
    raw_output: str,
    candidates: list[dict[str, Any]],
    target_count: int,
) -> tuple[list[str], list[str]]:
    diagnostics: list[str] = []
    valid_ids = {str(item.get("task_id", "")) for item in candidates}
    try:
        decoded = _extract_json_object(raw_output)
    except json.JSONDecodeError as exc:
        return [], [f"json_decode_error:{exc.msg}"]
    selected = decoded.get("selected_task_ids")
    if not isinstance(selected, list):
        return [], ["schema_error:selected_task_ids_must_be_list"]
    ids: list[str] = []
    for item in selected:
        task_id = str(item)
        if task_id not in valid_ids:
            diagnostics.append(f"unknown_task_id:{task_id}")
            continue
        if task_id not in ids:
            ids.append(task_id)
        if len(ids) >= target_count:
            break
    return ids, diagnostics


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        decoded = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start < 0:
            raise
        decoded, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(decoded, dict):
        raise json.JSONDecodeError("top level is not an object", cleaned, 0)
    return decoded


def _fill_anchor_selection(
    selected_ids: list[str],
    candidates: list[dict[str, Any]],
    target_count: int,
) -> list[str]:
    selected = list(selected_ids)
    seen = set(selected)
    ranked = _rank_anchor_candidates(candidates)
    for item in ranked:
        task_id = str(item.get("task_id", ""))
        if task_id and task_id not in seen:
            selected.append(task_id)
            seen.add(task_id)
        if len(selected) >= target_count:
            break
    return selected[:target_count]


def _rank_anchor_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda item: (
            bool(item.get("correct", False)),
            bool(item.get("format_compliant", False)),
            -int(item.get("output_chars", 0) or 0),
            int(item.get("index", 0) or 0),
        ),
    )


def _prefilter_memory_candidates(
    pool: list[dict[str, Any]],
    *,
    current_input: dict[str, Any],
    workflow_decision: dict[str, object],
    limit: int,
) -> list[dict[str, Any]]:
    """Cheap lexical/output-form prefilter before the LLM memory selector."""
    if limit <= 0 or len(pool) <= limit:
        return list(pool)
    query_text = json.dumps(current_input, ensure_ascii=False) + " " + json.dumps(
        workflow_decision,
        ensure_ascii=False,
    )
    question = str(current_input.get("question") or current_input.get("goal") or query_text)
    query_tokens = _memory_tokens(query_text)
    query_output_form = infer_output_form(question)["kind"]

    def base_score(item: dict[str, Any]) -> float:
        item_tokens = _memory_tokens(_memory_candidate_text(item))
        overlap = _memory_jaccard(query_tokens, item_tokens)
        output_match = str(item.get("output_form") or "unspecified") == str(query_output_form)
        abstract_bonus = 0.35 if item.get("memory_kind") == "abstract" else 0.0
        failure_bonus = 0.15 if item.get("abstract_role") == "failure_avoidance" else 0.0
        verified_bonus = 0.10 if item.get("previous_attempt_correct") is True else 0.0
        return overlap * 4.0 + (1.5 if output_match else 0.0) + abstract_bonus + failure_bonus + verified_bonus

    remaining = list(pool)
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < limit:
        best = max(
            remaining,
            key=lambda item: (
                base_score(item)
                - 0.35
                * max(
                    (
                        _memory_jaccard(
                            _memory_tokens(_memory_candidate_text(item)),
                            _memory_tokens(_memory_candidate_text(chosen)),
                        )
                        for chosen in selected
                    ),
                    default=0.0,
                ),
                str(item.get("memory_id", "")),
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return selected


def _memory_candidate_text(item: dict[str, Any]) -> str:
    fields = (
        "question",
        "guidance",
        "applicable_when",
        "reusable_pattern",
        "failure_mode",
        "correct_feedback",
        "output_form",
    )
    return " ".join(str(item.get(field) or "") for field in fields)


def _memory_tokens(text: str) -> set[str]:
    # Include CJK characters and alphanumeric words; discard punctuation-only tokens.
    return set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", str(text).lower()))


def _memory_jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
