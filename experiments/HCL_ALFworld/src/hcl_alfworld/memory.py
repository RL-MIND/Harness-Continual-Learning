from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .schemas import HarnessState, StrategyArtifact, TaskEvidence
from .text import overlap_score


class ExperienceMemory:
    """Paper Eq. (3): traceable raw records, abstractions, and anchors."""

    def __init__(self, top_k_memory: int = 5, top_k_strategies: int = 3):
        self.top_k_memory = top_k_memory
        self.top_k_strategies = top_k_strategies

    @staticmethod
    def _memory_text(item: Dict[str, Any]) -> str:
        steps = item.get("steps", [])
        action_text = " ".join(step.get("action", "") for step in steps)
        observation_text = " ".join(step.get("observation", "") for step in steps[-3:])
        return (
            f"{item.get('goal', '')} {item.get('task_type', '')} {action_text} {observation_text}"
        )

    def raw_candidates(
        self, task: TaskEvidence, state: HarnessState, pool_size: int = 20
    ) -> List[Dict[str, Any]]:
        """Bound the prompt size; the LLM router makes the actual relevance decision."""
        query = f"{task.task_type} {task.goal} {task.evidence}"
        ranked: List[Tuple[float, Dict[str, Any]]] = []
        for item in state.raw_memory:
            score = overlap_score(query, self._memory_text(item))
            if item.get("task_type") == task.task_type:
                score += 1.0
            if item.get("success"):
                score += 0.15
            ranked.append((score, item))
        ranked.sort(key=lambda pair: (pair[0], pair[1].get("episode_id", "")), reverse=True)
        return [item for _, item in ranked[:pool_size]]

    def strategy_candidates(
        self, task: TaskEvidence, state: HarnessState, pool_size: int = 20
    ) -> List[StrategyArtifact]:
        query = f"{task.task_type} {task.goal} {task.evidence}"
        ranked: List[Tuple[float, StrategyArtifact]] = []
        for strategy in state.abstract_memory:
            text = " ".join(
                [strategy.task_type, strategy.title]
                + strategy.procedure
                + strategy.applicability
                + strategy.failure_boundaries
            )
            score = overlap_score(query, text)
            if strategy.task_type == task.task_type:
                score += 1.0
            score += min(strategy.validation_count, 10) * 0.01
            ranked.append((score, strategy))
        ranked.sort(key=lambda pair: (pair[0], pair[1].artifact_id), reverse=True)
        return [item for _, item in ranked[:pool_size]]


class HarnessStore:
    """Atomic persistence for stable state, candidates, decisions, and interaction audit."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.candidates_dir = self.directory / "candidates"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.stable_path = self.directory / "stable.json"

    @staticmethod
    def _write_json(path: Path, data: Dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)

    @staticmethod
    def _append_jsonl(path: Path, data: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n")

    def load(self, initial: HarnessState) -> HarnessState:
        if not self.stable_path.exists():
            self.save_stable(initial)
            return initial
        with self.stable_path.open("r", encoding="utf-8") as handle:
            return HarnessState.from_dict(json.load(handle))

    def save_stable(self, state: HarnessState) -> None:
        self._write_json(self.stable_path, state.to_dict())

    def save_candidate(self, candidate: Dict[str, Any]) -> Path:
        path = self.candidates_dir / f"{candidate['candidate_id']}.json"
        self._write_json(path, candidate)
        return path

    def append_decision(self, decision: Dict[str, Any]) -> None:
        self._append_jsonl(self.directory / "decisions.jsonl", decision)

    def append_interaction(self, episode: Dict[str, Any]) -> None:
        self._append_jsonl(self.directory / "interactions.jsonl", episode)
