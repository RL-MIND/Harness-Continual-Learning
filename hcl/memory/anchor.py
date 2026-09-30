from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..json_utils import atomic_write_jsonl


class AnchorMemory:
    """Deduplicated, clustered historical cases for regression validation."""

    def __init__(
        self,
        path: str | Path,
        *,
        capacity_per_task: int = 15,
        similarity_threshold: float = 0.82,
    ) -> None:
        self.path = Path(path)
        self.capacity_per_task = max(int(capacity_per_task), 0)
        self.similarity_threshold = min(max(float(similarity_threshold), 0.0), 1.0)
        self.records = self._load()

    def clear(self) -> None:
        self.records = []
        self._persist()

    def add(self, task_name: str, examples: list[dict[str, Any]]) -> None:
        retained = [item for item in self.records if item.get("task_name") != task_name]
        selected = _clustered_anchor_selection(
            [dict(item) for item in examples],
            capacity=self.capacity_per_task,
            similarity_threshold=self.similarity_threshold,
        )
        self.records = retained + selected
        self._persist()

    def add_batch(
        self,
        task_name: str,
        examples: list[dict[str, Any]],
        *,
        max_total_per_task: int | None = None,
    ) -> list[dict[str, Any]]:
        if not examples:
            return []
        retained = [item for item in self.records if item.get("task_name") != task_name]
        task_records = [dict(item) for item in self.records if item.get("task_name") == task_name]
        previous_ids = {str(item.get("task_id", "")) for item in task_records}
        by_id = {str(item.get("task_id", "")): item for item in task_records}
        for example in examples:
            item = dict(example)
            task_id = str(item.get("task_id", ""))
            if not task_id:
                continue
            by_id[task_id] = item
        capacity = self.capacity_per_task
        if max_total_per_task is not None:
            capacity = max(int(max_total_per_task), 0)
        bounded = _clustered_anchor_selection(
            list(by_id.values()),
            capacity=capacity,
            similarity_threshold=self.similarity_threshold,
        )
        self.records = retained + bounded
        self._persist()
        kept_ids = {str(item.get("task_id", "")) for item in bounded}
        return [
            item
            for item in bounded
            if str(item.get("task_id", "")) not in previous_ids
            and str(item.get("task_id", "")) in kept_ids
        ]

    def all_examples(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.records]

    def task_ids(self, task_name: str) -> set[str]:
        return {
            str(item.get("task_id", ""))
            for item in self.records
            if item.get("task_name") == task_name
        }

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


def _clustered_anchor_selection(
    records: list[dict[str, Any]],
    *,
    capacity: int,
    similarity_threshold: float,
) -> list[dict[str, Any]]:
    if capacity <= 0:
        return []
    exact_unique: dict[str, dict[str, Any]] = {}
    for item in records:
        key = _normalized_question(item)
        if not key:
            key = f"task_id:{item.get('task_id', '')}"
        previous = exact_unique.get(key)
        if previous is None or _anchor_priority(item) > _anchor_priority(previous):
            exact_unique[key] = item
    unique = list(exact_unique.values())
    clusters: list[list[dict[str, Any]]] = []
    for item in sorted(unique, key=_anchor_priority, reverse=True):
        item_tokens = _question_tokens(item)
        target_cluster: list[dict[str, Any]] | None = None
        for cluster in clusters:
            representative_tokens = _question_tokens(cluster[0])
            if _token_jaccard(item_tokens, representative_tokens) >= similarity_threshold:
                target_cluster = cluster
                break
        if target_cluster is None:
            clusters.append([item])
        else:
            target_cluster.append(item)
    # First cover as many distinct clusters as possible, prioritizing failures.
    cluster_representatives = [max(cluster, key=_anchor_priority) for cluster in clusters]
    selected = sorted(cluster_representatives, key=_anchor_priority, reverse=True)[:capacity]
    if len(selected) < capacity:
        selected_ids = {str(item.get("task_id", "")) for item in selected}
        remaining = [item for item in unique if str(item.get("task_id", "")) not in selected_ids]
        remaining.sort(
            key=lambda item: (
                _minimum_distance(item, selected),
                _anchor_priority(item),
            ),
            reverse=True,
        )
        selected.extend(remaining[: capacity - len(selected)])
    return selected[:capacity]


def _anchor_priority(item: dict[str, Any]) -> tuple[int, int, str]:
    metadata = item.get("_anchor_meta")
    meta = metadata if isinstance(metadata, dict) else {}
    format_ok = bool(meta.get("format_compliant", True))
    correct = bool(meta.get("correct", True))
    difficulty = 2 if not format_ok else 1 if not correct else 0
    return difficulty, len(str(item.get("question", ""))), str(item.get("task_id", ""))


def _minimum_distance(item: dict[str, Any], selected: list[dict[str, Any]]) -> float:
    if not selected:
        return 1.0
    tokens = _question_tokens(item)
    return min(1.0 - _token_jaccard(tokens, _question_tokens(other)) for other in selected)


def _normalized_question(item: dict[str, Any]) -> str:
    return " ".join(
        "".join(
            character.lower() if character.isalnum() else " "
            for character in str(item.get("question", ""))
        ).split()
    )


def _question_tokens(item: dict[str, Any]) -> set[str]:
    return set(_normalized_question(item).split())


def _token_jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
