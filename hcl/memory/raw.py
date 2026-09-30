from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from ..evaluator import answer_format_compliant, answer_matches
from ..json_utils import atomic_write_jsonl
from ..task_interface import infer_output_form


class RawMemory:
    """Bounded interaction evidence selected by failure type and diversity."""

    def __init__(self, path: str | Path, *, capacity_per_task: int = 4) -> None:
        self.path = Path(path)
        self.capacity_per_task = max(int(capacity_per_task), 0)
        self.records = self._load()

    def clear(self) -> None:
        self.records = []
        self._persist()

    def add(
        self,
        task_name: str,
        examples: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
        *,
        anchor_ids: set[str] | None = None,
        replace_anchors: bool = False,
    ) -> None:
        anchor_ids = anchor_ids or set()
        by_id = {str(item.get("task_id")): item for item in predictions}
        retained = [item for item in self.records if _source(item, "task_name") != task_name]
        task_records = [dict(item) for item in self.records if _source(item, "task_name") == task_name]
        if replace_anchors:
            task_records = [item for item in task_records if not item.get("is_anchor")]
        by_task_id = {str(_source(item, "task_id")): item for item in task_records}
        for example in examples:
            task_id = str(example.get("task_id", ""))
            prediction = by_id.get(task_id, {})
            response = prediction.get("answer")
            by_task_id[task_id] = {
                "memory_id": _memory_id(task_id),
                "memory_type": "raw",
                "internal_source": {
                    "task_name": task_name,
                    "task_type": str(example.get("task_type", "")),
                    "task_id": task_id,
                    "split": str(example.get("split", "")),
                },
                "question": str(example.get("question", "")),
                "response": response,
                "feedback": example.get("answer"),
                "response_is_correct": answer_matches(response, example.get("answer"), example=example),
                "format_compliant": answer_format_compliant(response, example.get("answer")),
                "output_form": infer_output_form(str(example.get("question", "")))["kind"],
                "is_anchor": task_id in anchor_ids,
            }
        task_records = list(by_task_id.values())
        anchors = [item for item in task_records if item.get("is_anchor")]
        interactions = [item for item in task_records if not item.get("is_anchor")]
        anchor_slots = min(len(anchors), self.capacity_per_task)
        interaction_slots = max(self.capacity_per_task - anchor_slots, 0)
        bounded = anchors[:anchor_slots]
        if interaction_slots:
            bounded.extend(_stratified_interactions(interactions, interaction_slots))
        self.records = retained + bounded
        self._persist()

    def retrieval_candidates(self) -> list[dict[str, Any]]:
        """Expose a global, de-labelled pool; ranking belongs to the Router selector."""
        return [_retrieval_view(item) for item in self.records if not item.get("is_anchor")]

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    records.append(item)
        return records

    def _persist(self) -> None:
        atomic_write_jsonl(self.path, self.records)


def _retrieval_view(item: dict[str, Any]) -> dict[str, Any]:
    response_is_correct = item.get("response_is_correct")
    if not isinstance(response_is_correct, bool):
        response_is_correct = answer_matches(item.get("response"), item.get("feedback"))
    return {
        "memory_id": item.get("memory_id") or _memory_id(str(_source(item, "task_id"))),
        "memory_kind": "raw",
        "question": _bounded_text(item.get("question"), 1200),
        "previous_attempt": _bounded_text(item.get("response"), 512),
        "previous_attempt_correct": response_is_correct,
        "correct_feedback": _bounded_text(item.get("feedback"), 512),
        "output_form": infer_output_form(str(item.get("question", "")))["kind"],
        "usage_rule": (
            "correct_feedback is authoritative supervision; previous_attempt is only a historical attempt "
            "and may be incorrect. Do not copy an incorrect attempt."
        ),
    }


def _bounded_text(value: object, limit: int) -> object:
    if value is None or not isinstance(value, str):
        return value
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + " …[truncated]"


def _source(item: dict[str, Any], key: str) -> Any:
    internal = item.get("internal_source")
    if isinstance(internal, dict) and key in internal:
        return internal.get(key)
    return item.get(key)


def _memory_id(task_id: str) -> str:
    digest = hashlib.sha256(str(task_id).encode("utf-8")).hexdigest()[:12]
    return f"raw_{digest}"


def _stratified_interactions(
    records: list[dict[str, Any]],
    capacity: int,
) -> list[dict[str, Any]]:
    """Keep format failures, answer errors, and representative successes.

    Buckets are exclusive. Within each bucket greedy lexical diversity avoids
    spending several slots on near-identical examples. Unused quota is filled
    from the remaining globally diverse records.
    """
    if capacity <= 0:
        return []
    format_failures = [item for item in records if not bool(item.get("format_compliant", True))]
    answer_errors = [
        item
        for item in records
        if bool(item.get("format_compliant", True)) and not bool(item.get("response_is_correct", False))
    ]
    representative = [
        item
        for item in records
        if bool(item.get("format_compliant", True)) and bool(item.get("response_is_correct", False))
    ]
    format_quota = max(round(capacity * 0.20), 1)
    error_quota = max(round(capacity * 0.35), 1)
    representative_quota = max(capacity - format_quota - error_quota, 0)
    selected: list[dict[str, Any]] = []
    selected.extend(_diverse_take(format_failures, format_quota, selected))
    selected.extend(_diverse_take(answer_errors, error_quota, selected))
    selected.extend(_diverse_take(representative, representative_quota, selected))
    selected_ids = {str(_source(item, "task_id")) for item in selected}
    remaining = [item for item in records if str(_source(item, "task_id")) not in selected_ids]
    selected.extend(_diverse_take(remaining, capacity - len(selected), selected))
    return selected[:capacity]


def _diverse_take(
    candidates: list[dict[str, Any]],
    count: int,
    already_selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pool = list(candidates)
    chosen: list[dict[str, Any]] = []
    while pool and len(chosen) < count:
        references = already_selected + chosen
        if not references:
            best = pool[-1]
        else:
            best = max(
                pool,
                key=lambda item: (
                    min(
                        1.0 - _token_jaccard(_question_tokens(item), _question_tokens(reference))
                        for reference in references
                    ),
                    str(_source(item, "task_id")),
                ),
            )
        chosen.append(best)
        pool.remove(best)
    return chosen


def _question_tokens(item: dict[str, Any]) -> set[str]:
    return set(_normalize_text(str(item.get("question", ""))).split())


def _normalize_text(text: str) -> str:
    return " ".join(
        "".join(character.lower() if character.isalnum() else " " for character in str(text)).split()
    )


def _token_jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
