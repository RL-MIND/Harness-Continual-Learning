from __future__ import annotations

import copy
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .embedding import EmbedderProtocol, cosine_similarity
from .llm import StructuredLLMProtocol
from .memory import HarnessStore
from .schemas import (
    CandidateUpdate,
    EpisodeRecord,
    EvaluationResult,
    HarnessState,
    RoutingContext,
    StepRecord,
    TaskEvidence,
)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


class RAGRouter:
    """Retrieve raw past episodes using cosine similarity only."""

    def __init__(self, embedder: EmbedderProtocol, config: Dict[str, Any]):
        self.embedder = embedder
        self.top_k = int(config.get("top_k", 3))
        configured_threshold = config.get("similarity_threshold")
        self.similarity_threshold = (
            float(configured_threshold) if configured_threshold is not None else None
        )

    @staticmethod
    def reroute_reasons(
        task: TaskEvidence,
        previous_task: Optional[TaskEvidence],
        steps: Sequence[StepRecord],
    ) -> List[str]:
        del task, steps
        return ["episode_start"] if previous_task is None else []

    @staticmethod
    def should_refresh_semantic_context(
        reasons: Sequence[str],
        current_step: int,
        last_semantic_route_step: int,
    ) -> bool:
        del current_step, last_semantic_route_step
        return "episode_start" in reasons

    @staticmethod
    def _retrieval_query(task: TaskEvidence) -> str:
        return task.goal.strip() or task.initial_observation.strip() or task.evidence.strip()

    def _rank(
        self,
        query_vector: Sequence[float],
        memories: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []
        for memory in memories:
            similarity = cosine_similarity(
                query_vector,
                memory.get("retrieval_embedding", []),
            )
            if (
                self.similarity_threshold is not None
                and similarity < self.similarity_threshold
            ):
                continue
            item = copy.deepcopy(memory)
            item["similarity"] = float(similarity)
            item["retrieval_score"] = float(similarity)
            ranked.append(item)
        ranked.sort(
            key=lambda item: (
                float(item["similarity"]),
                str(item.get("memory_id", "")),
            ),
            reverse=True,
        )
        return ranked[: self.top_k]

    def route(
        self,
        task: TaskEvidence,
        state: HarnessState,
        trigger_reasons: Optional[Sequence[str]] = None,
    ) -> RoutingContext:
        query = self._retrieval_query(task)
        query_vector = self.embedder.embed([query])[0] if state.raw_memory else []
        selected = self._rank(query_vector, state.raw_memory) if query_vector else []
        trace = {
            "component": "rag_router",
            "mode": "cosine_top_k",
            "llm_called": False,
            "trigger_reasons": list(trigger_reasons or ["episode_start"]),
            "retrieval_query": query,
            "similarity_threshold": self.similarity_threshold,
            "selected_memory_ids": [
                str(item.get("memory_id", "")) for item in selected
            ],
            "selected_similarities": [
                float(item.get("similarity", 0.0)) for item in selected
            ],
            "selected_strategy_ids": [],
            "selected_capability_ids": [],
            "semantic_context_source_step": task.step,
            "rationale": (
                "RAG retrieved the top-k raw past episodes using embedding cosine "
                "similarity only."
            ),
        }
        return RoutingContext(task, selected, [], [], trace)

    @staticmethod
    def reuse(
        task: TaskEvidence,
        state: HarnessState,
        previous_context: RoutingContext,
    ) -> RoutingContext:
        del state
        trace = copy.deepcopy(previous_context.trace)
        trace.update(
            {
                "mode": "episode_route_reuse",
                "llm_called": False,
                "trigger_reasons": [],
                "reused_from_step": trace.get("semantic_context_source_step", 0),
            }
        )
        return RoutingContext(
            task=task,
            memories=copy.deepcopy(previous_context.memories),
            strategies=[],
            capabilities=[],
            trace=trace,
        )


class RAGHarness:
    """Standard RAG over raw episode experiences behind the shared runner API."""

    def __init__(
        self,
        store_dir: str | Path,
        config: Dict[str, Any],
        llm: StructuredLLMProtocol,
        embedder: EmbedderProtocol,
    ):
        del llm
        self.config = config
        self.embedder = embedder
        self.store = HarnessStore(store_dir)
        self.router = RAGRouter(embedder, config)
        self.capacity = int(config.get("memory_capacity", 500))
        initial = HarnessState(
            routing_policy={
                "method": "rag",
                "retrieval": "cosine_top_k",
                "top_k": self.router.top_k,
                "similarity_threshold": self.router.similarity_threshold,
            },
            evaluator_rules={"method": "rag", "updates": "append_only"},
        )
        self.state = self.store.load(initial)

    def route(
        self,
        task: TaskEvidence,
        trigger_reasons: Optional[Sequence[str]] = None,
    ) -> RoutingContext:
        return self.router.route(task, self.state, trigger_reasons)

    def reuse_route(
        self,
        task: TaskEvidence,
        previous_context: RoutingContext,
    ) -> RoutingContext:
        return self.router.reuse(task, self.state, previous_context)

    @staticmethod
    def _trajectory(episode: EpisodeRecord) -> List[Dict[str, Any]]:
        return [
            {
                "step": index,
                "observation": step.observation,
                "action": step.action,
                "next_observation": step.next_observation,
                "reward": step.reward,
            }
            for index, step in enumerate(episode.steps)
        ]

    def learn(
        self,
        episode: EpisodeRecord,
        final_context: RoutingContext,
    ) -> Tuple[CandidateUpdate, EvaluationResult]:
        proposed = copy.deepcopy(self.state)
        goal = episode.goal.strip()
        embedding = self.embedder.embed([goal])[0]
        memory_id = f"rag-{uuid.uuid4().hex[:12]}"
        memory = {
            "memory_id": memory_id,
            "episode_id": episode.episode_id,
            "task_type": episode.task_type,
            "goal": goal,
            "success": episode.success,
            "failure_reason": episode.failure_reason,
            "memory_type": "rag_episode",
            "steps": self._trajectory(episode),
            "retrieval_key": goal,
            "retrieval_embedding": list(embedding),
            "created_at": _now_iso(),
        }
        proposed.raw_memory.append(memory)
        evicted_memory_ids: List[str] = []
        if self.capacity > 0 and len(proposed.raw_memory) > self.capacity:
            overflow = len(proposed.raw_memory) - self.capacity
            evicted_memory_ids = [
                str(item.get("memory_id", ""))
                for item in proposed.raw_memory[:overflow]
            ]
            proposed.raw_memory = proposed.raw_memory[overflow:]
        proposed.version = self.state.version + 1
        operation = {
            "kind": "add_rag_experience",
            "memory_id": memory_id,
            "success": episode.success,
            "evicted_memory_ids": evicted_memory_ids,
        }
        candidate = CandidateUpdate(
            candidate_id=f"rag-candidate-v{proposed.version}-{episode.episode_id}",
            base_version=self.state.version,
            proposed_state=proposed,
            operations=[operation],
            rationale=[
                "Stored the complete episode as an append-only RAG experience.",
                "No reflection, proceduralization, Q update, or semantic merge was applied.",
            ],
        )
        evaluation = EvaluationResult(
            candidate_id=candidate.candidate_id,
            decision="commit",
            current_score=float(episode.score),
            anchor_retention=1.0,
            regressions=[],
            diagnostics=[
                "RAG appends training experiences directly without HCL gating or RL."
            ],
            accepted_operation_indices=[0],
            rejected_operation_indices=[],
        )
        self.state = proposed
        self.store.save_candidate(candidate.to_dict())
        self.store.save_stable(self.state)
        self.store.append_interaction(episode.to_dict())
        self.store.append_decision(evaluation.to_dict())
        episode.harness_traces.append(
            {
                "rag_learning": {
                    "new_memory_id": memory_id,
                    "selected_memory_ids": [
                        str(item.get("memory_id", ""))
                        for item in final_context.memories
                    ],
                    "evicted_memory_ids": evicted_memory_ids,
                }
            }
        )
        return candidate, evaluation
