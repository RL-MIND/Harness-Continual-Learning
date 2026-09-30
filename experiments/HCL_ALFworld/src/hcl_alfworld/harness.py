from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

from .capability import default_capabilities
from .evaluator import ContinualEvaluator
from .llm import StructuredLLMProtocol
from .memory import ExperienceMemory, HarnessStore
from .optimizer import ContinualOptimizer
from .router import AdaptiveRouter
from .schemas import CandidateUpdate, EpisodeRecord, EvaluationResult, HarnessState, RoutingContext


class HCLHarness:
    """Closed HCL loop implementing stable/candidate state separation."""

    def __init__(
        self,
        store_dir: str | Path,
        config: Dict[str, Any],
        llm: StructuredLLMProtocol,
    ):
        self.config = config
        self.gate_mode = str(config.get("gate_mode", "stability")).strip().lower()
        if self.gate_mode not in {"plasticity", "stability"}:
            raise ValueError(
                "harness.gate_mode must be either 'plasticity' or 'stability'"
            )
        self.store = HarnessStore(store_dir)
        memory = ExperienceMemory(
            top_k_memory=int(config.get("top_k_memory", 5)),
            top_k_strategies=int(config.get("top_k_strategies", 3)),
        )
        self.router = AdaptiveRouter(
            llm,
            memory,
            memory_enabled=bool(config.get("memory_enabled", True)),
            raw_memory_retrieval_enabled=bool(
                config.get("raw_memory_retrieval_enabled", True)
            ),
            strategy_retrieval_enabled=bool(
                config.get("strategy_retrieval_enabled", True)
            ),
            capabilities_enabled=bool(config.get("capabilities_enabled", True)),
            candidate_pool_size=int(config.get("candidate_pool_size", 20)),
            gate_mode=self.gate_mode,
        )
        self.optimizer = ContinualOptimizer(
            llm,
            raw_memory_capacity=int(config.get("raw_memory_capacity", 500)),
            anchors_per_task=int(config.get("anchors_per_task", 5)),
            capability_learning_enabled=bool(
                config.get("capability_learning_enabled", True)
                and config.get("capabilities_enabled", True)
            ),
            max_capability_operations=int(config.get("max_capability_operations", 3)),
            provisional_min_positive_evidence=int(
                config.get("provisional_min_positive_evidence", 2)
            ),
            validated_min_positive_evidence=int(
                config.get("validated_min_positive_evidence", 3)
            ),
            validated_min_success_rate=float(
                config.get("validated_min_success_rate", 0.67)
            ),
            gate_mode=self.gate_mode,
        )
        self.evaluator = ContinualEvaluator(
            llm,
            min_current_score=float(config.get("min_current_score", 1.0)),
            max_anchor_regression=float(config.get("max_anchor_regression", 0.0)),
            provisional_min_positive_evidence=int(
                config.get("provisional_min_positive_evidence", 2)
            ),
            validated_min_positive_evidence=int(
                config.get("validated_min_positive_evidence", 3)
            ),
            validated_min_success_rate=float(
                config.get("validated_min_success_rate", 0.67)
            ),
            gate_mode=self.gate_mode,
        )
        initial = HarnessState(
            capabilities=default_capabilities(),
            routing_policy={
                "gate_mode": self.gate_mode,
                "top_k_memory": int(config.get("top_k_memory", 5)),
                "top_k_strategies": int(config.get("top_k_strategies", 3)),
                "raw_memory_retrieval_enabled": bool(
                    config.get("raw_memory_retrieval_enabled", True)
                ),
                "strategy_retrieval_enabled": bool(
                    config.get("strategy_retrieval_enabled", True)
                ),
            },
            evaluator_rules={
                "gate_mode": self.gate_mode,
                "min_current_score": float(config.get("min_current_score", 1.0)),
                "max_anchor_regression": float(config.get("max_anchor_regression", 0.0)),
            },
        )
        self.state = self.store.load(initial)

    def route(self, task, trigger_reasons=None) -> RoutingContext:
        return self.router.route(task, self.state, trigger_reasons)

    def reuse_route(
        self, task, previous_context: RoutingContext
    ) -> RoutingContext:
        return self.router.reuse(task, self.state, previous_context)

    def _partial_merge(
        self, candidate: CandidateUpdate, evaluation: EvaluationResult
    ) -> HarnessState:
        merged = copy.deepcopy(self.state)
        allowed = set(evaluation.partial_operation_kinds)
        candidate_state = candidate.proposed_state
        if "add_raw_memory" in allowed:
            merged.raw_memory = copy.deepcopy(candidate_state.raw_memory)
        if "add_strategy" in allowed or "refine_strategy" in allowed:
            by_id = {item.artifact_id: item for item in merged.abstract_memory}
            for strategy in candidate_state.abstract_memory:
                if not strategy.title.startswith("Failure boundary"):
                    continue
                if strategy.artifact_id in by_id:
                    existing = by_id[strategy.artifact_id]
                    existing.title = strategy.title
                    existing.procedure = list(strategy.procedure)
                    existing.applicability = list(strategy.applicability)
                    existing.failure_boundaries = list(strategy.failure_boundaries)
                    existing.evidence_episode_ids = list(strategy.evidence_episode_ids)
                else:
                    merged.abstract_memory.append(copy.deepcopy(strategy))
        if "record_capability_outcome" in allowed:
            outcome_ids = {
                operation.get("capability_id")
                for operation in candidate.operations
                if operation.get("kind") == "record_capability_outcome"
            }
            candidate_by_id = {
                item.capability_id: item for item in candidate_state.capabilities
            }
            for capability in merged.capabilities:
                if capability.capability_id not in outcome_ids:
                    continue
                candidate_capability = candidate_by_id[capability.capability_id]
                capability.usage_count = candidate_capability.usage_count
                capability.success_count = candidate_capability.success_count
                capability.failure_count = candidate_capability.failure_count
                capability.positive_evidence_episode_ids = list(
                    candidate_capability.positive_evidence_episode_ids
                )
                capability.negative_evidence_episode_ids = list(
                    candidate_capability.negative_evidence_episode_ids
                )
                capability.updated_at_version = candidate_capability.updated_at_version
        merged.version += 1
        return merged

    def _plasticity_merge(
        self, candidate: CandidateUpdate, evaluation: EvaluationResult
    ) -> HarnessState:
        """Apply only accepted delta operations while retaining monotonic evidence."""
        merged = copy.deepcopy(self.state)
        accepted = set(evaluation.accepted_operation_indices)
        candidate_state = candidate.proposed_state
        candidate_strategies = {
            item.artifact_id: item for item in candidate_state.abstract_memory
        }
        candidate_capabilities = {
            item.capability_id: item for item in candidate_state.capabilities
        }
        candidate_anchors = {item.anchor_id: item for item in candidate_state.anchors}

        if any(
            index in accepted and operation.get("kind") == "add_raw_memory"
            for index, operation in enumerate(candidate.operations)
        ):
            merged.raw_memory = copy.deepcopy(candidate_state.raw_memory)

        for index, operation in enumerate(candidate.operations):
            if index not in accepted:
                continue
            kind = str(operation.get("kind", ""))
            if kind in {"add_raw_memory", "record_capability_outcome"}:
                continue
            if kind in {"add_strategy", "refine_strategy"}:
                artifact_id = str(operation.get("artifact_id", ""))
                proposed = candidate_strategies.get(artifact_id)
                if proposed is None:
                    continue
                by_id = {item.artifact_id: pos for pos, item in enumerate(merged.abstract_memory)}
                if artifact_id in by_id:
                    merged.abstract_memory[by_id[artifact_id]] = copy.deepcopy(proposed)
                else:
                    merged.abstract_memory.append(copy.deepcopy(proposed))
            elif kind in {
                "create_capability",
                "refine_capability",
                "deprecate_capability",
            }:
                capability_id = str(operation.get("capability_id", ""))
                proposed = candidate_capabilities.get(capability_id)
                if proposed is None:
                    continue
                by_id = {item.capability_id: pos for pos, item in enumerate(merged.capabilities)}
                if capability_id in by_id:
                    merged.capabilities[by_id[capability_id]] = copy.deepcopy(proposed)
                else:
                    merged.capabilities.append(copy.deepcopy(proposed))
            elif kind == "update_capability_evidence":
                capability_id = str(operation.get("capability_id", ""))
                proposed = candidate_capabilities.get(capability_id)
                target = next(
                    (
                        item
                        for item in merged.capabilities
                        if item.capability_id == capability_id
                    ),
                    None,
                )
                if proposed is not None and target is not None:
                    target.validation_evidence = list(proposed.validation_evidence)
                    target.updated_at_version = proposed.updated_at_version
            elif kind == "add_anchor":
                anchor_id = str(operation.get("anchor_id", ""))
                proposed = candidate_anchors.get(anchor_id)
                if proposed is not None and all(
                    item.anchor_id != anchor_id for item in merged.anchors
                ):
                    merged.anchors.append(copy.deepcopy(proposed))

        outcome_ids = {
            str(operation.get("capability_id"))
            for index, operation in enumerate(candidate.operations)
            if index in accepted
            and operation.get("kind") == "record_capability_outcome"
        }
        for capability in merged.capabilities:
            if capability.capability_id not in outcome_ids:
                continue
            proposed = candidate_capabilities.get(capability.capability_id)
            if proposed is None:
                continue
            capability.usage_count = proposed.usage_count
            capability.success_count = proposed.success_count
            capability.failure_count = proposed.failure_count
            capability.positive_evidence_episode_ids = list(
                proposed.positive_evidence_episode_ids
            )
            capability.negative_evidence_episode_ids = list(
                proposed.negative_evidence_episode_ids
            )
            capability.updated_at_version = proposed.updated_at_version

        self.optimizer.capability_evolution.recalibrate_lifecycle(merged)
        merged.version += 1
        return merged

    def learn(
        self, episode: EpisodeRecord, final_context: RoutingContext
    ) -> tuple[CandidateUpdate, EvaluationResult]:
        try:
            candidate = self.optimizer.propose(self.state, episode, final_context)
        except RuntimeError as exc:
            message = str(exc)
            if not message.startswith("optimizer did not return valid JSON:"):
                raise
            if self.gate_mode == "plasticity":
                episode.harness_traces.append(
                    {
                        "optimizer": {
                            "component": "llm_optimizer",
                            "mode": "json_fallback_evidence_only",
                            "fallback": True,
                            "error": message,
                            "knowledge_type": "none",
                            "rationale": (
                                "Malformed optimizer reflection; grounded evidence "
                                "was retained without semantic operations."
                            ),
                            "capability_operations": [],
                            "strategy_diagnostic": (
                                "Skipped strategy: optimizer returned invalid JSON."
                            ),
                        }
                    }
                )
                candidate = self.optimizer.propose_evidence_only(
                    self.state,
                    episode,
                    "Optimizer JSON fallback: retained grounded evidence only.",
                )
                accepted = list(range(len(candidate.operations)))
                evaluation = EvaluationResult(
                    candidate_id=candidate.candidate_id,
                    decision="partial_merge",
                    current_score=float(episode.score),
                    anchor_retention=1.0,
                    regressions=[],
                    diagnostics=[
                        "Optimizer exhausted JSON retries; retained deterministic "
                        f"episode evidence. {message}"
                    ],
                    partial_operation_kinds=[
                        "add_raw_memory",
                        "record_capability_outcome",
                    ],
                    accepted_operation_indices=accepted,
                    rejected_operation_indices=[],
                )
                self.state = self._plasticity_merge(candidate, evaluation)
                self.store.save_candidate(candidate.to_dict())
                self.store.save_stable(self.state)
                self.store.append_interaction(episode.to_dict())
                self.store.append_decision(evaluation.to_dict())
                return candidate, evaluation
            # A malformed reflection must not abort a resumable experiment or
            # mutate the stable harness. The grounded episode remains in the
            # interaction audit and the no-op candidate makes the skipped
            # update explicit in episode/update artifacts.
            candidate = CandidateUpdate(
                candidate_id=(
                    f"candidate-v{self.state.version}-optimizer-fallback-"
                    f"{episode.episode_id}"
                ),
                base_version=self.state.version,
                proposed_state=copy.deepcopy(self.state),
                operations=[],
                rationale=[
                    "Optimizer JSON fallback: stable Harness was left unchanged."
                ],
            )
            evaluation = EvaluationResult(
                candidate_id=candidate.candidate_id,
                decision="reject",
                current_score=float(episode.score),
                anchor_retention=1.0,
                regressions=[],
                diagnostics=[
                    "Optimizer exhausted JSON retries; skipped this episode's "
                    f"Harness update. {message}"
                ],
                partial_operation_kinds=[],
            )
            episode.harness_traces.append(
                {
                    "optimizer": {
                        "component": "llm_optimizer",
                        "mode": "json_fallback_noop",
                        "fallback": True,
                        "error": message,
                        "knowledge_type": "none",
                        "rationale": (
                            "Malformed optimizer reflection; stable Harness was "
                            "preserved without applying candidate operations."
                        ),
                        "capability_operations": [],
                        "strategy_diagnostic": (
                            "Skipped strategy: optimizer returned invalid JSON."
                        ),
                    }
                }
            )
            self.store.save_candidate(candidate.to_dict())
            self.store.append_interaction(episode.to_dict())
            self.store.append_decision(evaluation.to_dict())
            return candidate, evaluation
        try:
            evaluation = self.evaluator.evaluate(self.state, candidate, episode)
        except RuntimeError as exc:
            message = str(exc)
            if (
                self.gate_mode != "plasticity"
                or not message.startswith("evaluator did not return valid JSON:")
            ):
                raise
            accepted = [
                index
                for index, operation in enumerate(candidate.operations)
                if operation.get("kind")
                in {"add_raw_memory", "record_capability_outcome"}
            ]
            rejected = [
                index
                for index in range(len(candidate.operations))
                if index not in accepted
            ]
            evaluation = EvaluationResult(
                candidate_id=candidate.candidate_id,
                decision="partial_merge" if accepted else "reject",
                current_score=float(episode.score),
                anchor_retention=1.0,
                regressions=[],
                diagnostics=[
                    "Evaluator exhausted JSON retries; retained deterministic "
                    f"evidence only. {message}"
                ],
                partial_operation_kinds=[
                    "add_raw_memory",
                    "record_capability_outcome",
                ],
                accepted_operation_indices=accepted,
                rejected_operation_indices=rejected,
            )
            episode.harness_traces.append(
                {
                    "evaluator": {
                        "component": "llm_evaluator",
                        "gate_mode": "plasticity",
                        "mode": "json_fallback_evidence_only",
                        "fallback": True,
                        "error": message,
                        "accepted_operation_indices": accepted,
                        "rejected_operation_indices": rejected,
                    }
                }
            )
        self.store.save_candidate(candidate.to_dict())

        if not bool(self.config.get("gate_enabled", True)) and evaluation.decision != "rollback":
            evaluation.decision = "commit"
            evaluation.diagnostics.append(
                "Semantic gate disabled by ablation; structural rollback remains mandatory."
            )

        if (
            self.gate_mode == "plasticity"
            and bool(self.config.get("gate_enabled", True))
            and evaluation.decision in {"commit", "partial_merge"}
        ):
            self.state = self._plasticity_merge(candidate, evaluation)
            self.store.save_stable(self.state)
        elif evaluation.decision == "commit":
            self.state = candidate.proposed_state
            self.store.save_stable(self.state)
        elif evaluation.decision == "partial_merge":
            self.state = self._partial_merge(candidate, evaluation)
            self.store.save_stable(self.state)
        self.store.append_interaction(episode.to_dict())
        self.store.append_decision(evaluation.to_dict())
        return candidate, evaluation


def make_continual_harness(
    method: str,
    store_dir: str | Path,
    config: Dict[str, Any],
    llm: StructuredLLMProtocol,
    *,
    embedder: Any = None,
    seed: int = 42,
) -> Any:
    """Construct one continual method behind the runner's common harness boundary."""
    normalized = method.strip().lower()
    if normalized == "hcl":
        return HCLHarness(store_dir, config, llm)
    if normalized == "rag":
        if embedder is None:
            raise ValueError("The RAG baseline requires an embedding provider.")
        from .rag_baseline import RAGHarness

        return RAGHarness(store_dir, config, llm, embedder)
    raise ValueError(
        f"Unknown continual_method: {method}. "
        "Expected 'hcl' or 'rag'."
    )
