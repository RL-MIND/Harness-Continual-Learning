from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List

from .capability_evolution import CapabilityEvolution
from .llm import StructuredLLMProtocol
from .schemas import (
    AnchorCase,
    CandidateUpdate,
    EpisodeRecord,
    HarnessState,
    RoutingContext,
    StrategyArtifact,
)
from .text import unique


class ContinualOptimizer:
    """Use LLM reflection to propose bounded edits without mutating stable state."""

    def __init__(
        self,
        llm: StructuredLLMProtocol,
        raw_memory_capacity: int = 500,
        anchors_per_task: int = 5,
        capability_learning_enabled: bool = True,
        max_capability_operations: int = 3,
        provisional_min_positive_evidence: int = 2,
        validated_min_positive_evidence: int = 3,
        validated_min_success_rate: float = 0.67,
        gate_mode: str = "stability",
    ):
        self.llm = llm
        self.raw_memory_capacity = raw_memory_capacity
        self.anchors_per_task = anchors_per_task
        self.capability_learning_enabled = capability_learning_enabled
        self.gate_mode = gate_mode
        self.capability_evolution = CapabilityEvolution(
            max_operations=max_capability_operations,
            provisional_min_positive_evidence=provisional_min_positive_evidence,
            validated_min_positive_evidence=validated_min_positive_evidence,
            validated_min_success_rate=validated_min_success_rate,
        )

    @staticmethod
    def _strategy_summary(item: StrategyArtifact) -> Dict[str, Any]:
        return {
            "artifact_id": item.artifact_id,
            "task_type": item.task_type,
            "title": item.title,
            "procedure": item.procedure,
            "applicability": item.applicability,
            "failure_boundaries": item.failure_boundaries,
            "validation_count": item.validation_count,
            "success_count": item.success_count,
        }

    @staticmethod
    def _clean_string_list(value: Any, limit: int = 30) -> List[str]:
        # OpenAI-compatible backends occasionally collapse a one-item JSON array
        # to a scalar string. Treat that as a recoverable shape variation.
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return unique(
            str(item).strip()[:500]
            for item in value[:limit]
            if isinstance(item, str) and item.strip()
        )

    @classmethod
    def _clean_procedure(cls, value: Any, limit: int = 30) -> List[str]:
        """Normalize common structured-step variants to procedure strings."""
        if isinstance(value, dict):
            value = value.get("steps", value.get("procedure", []))
        if isinstance(value, list):
            normalized: List[Any] = []
            for item in value[:limit]:
                if isinstance(item, str):
                    normalized.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                for key in ("instruction", "action", "description", "step"):
                    text = item.get(key)
                    if isinstance(text, str) and text.strip():
                        normalized.append(text)
                        break
            value = normalized
        return cls._clean_string_list(value, limit)

    def _apply_strategy_reflection(
        self,
        proposed: HarnessState,
        episode: EpisodeRecord,
        reflection: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        knowledge_type = str(reflection.get("knowledge_type", "none"))
        if episode.success and knowledge_type != "success_strategy":
            return None
        if not episode.success and knowledge_type != "failure_boundary":
            return None

        procedure = self._clean_procedure(reflection.get("procedure"))
        applicability = self._clean_string_list(reflection.get("applicability"), 15)
        boundaries = self._clean_string_list(reflection.get("failure_boundaries"), 20)
        if episode.success and not procedure:
            # A malformed semantic response must not abort a resumable experiment.
            # Skipping the strategy also prevents an empty artifact and its anchor
            # from contaminating stable state. Raw evidence and independently valid
            # capability proposals can still be reviewed by the evaluator.
            return None
        if not episode.success and not boundaries:
            boundaries = [episode.failure_reason or "The episode ended without task success."]

        # A failed rollout is evidence about a boundary, not permission to
        # overwrite a previously successful strategy. The stability mode keeps
        # the historical behavior for reproducible comparisons.
        requested_target_id = str(reflection.get("target_strategy_id", ""))
        if self.gate_mode == "plasticity" and not episode.success:
            requested_target = next(
                (
                    item
                    for item in proposed.abstract_memory
                    if item.artifact_id == requested_target_id
                    and item.task_type == episode.task_type
                ),
                None,
            )
            target_id = (
                requested_target_id
                if requested_target is not None
                and requested_target.title.startswith("Failure boundary")
                else ""
            )
        else:
            target_id = requested_target_id
        target = next(
            (
                item
                for item in proposed.abstract_memory
                if item.artifact_id == target_id and item.task_type == episode.task_type
            ),
            None,
        )
        if target is not None:
            target.title = str(reflection.get("title") or target.title)[:200]
            target.procedure = procedure or target.procedure
            target.applicability = unique(target.applicability + applicability)
            target.failure_boundaries = unique(target.failure_boundaries + boundaries)
            target.evidence_episode_ids = unique(
                target.evidence_episode_ids + [episode.episode_id]
            )
            target.validation_count += int(episode.success)
            target.success_count += int(episode.success)
            return {"kind": "refine_strategy", "artifact_id": target.artifact_id}

        title = str(reflection.get("title", "")).strip()[:200]
        if not title:
            title = (
                f"Reusable strategy for {episode.task_type}"
                if episode.success
                else f"Failure boundary for {episode.task_type}"
            )
        if not episode.success and not title.startswith("Failure boundary"):
            title = f"Failure boundary: {title}"
        strategy = StrategyArtifact(
            artifact_id=f"strategy-{uuid.uuid4().hex[:12]}",
            task_type=episode.task_type,
            title=title,
            procedure=procedure,
            applicability=applicability,
            failure_boundaries=boundaries,
            evidence_episode_ids=[episode.episode_id],
            validation_count=int(episode.success),
            success_count=int(episode.success),
            created_at_version=proposed.version,
        )
        proposed.abstract_memory.append(strategy)
        return {"kind": "add_strategy", "artifact_id": strategy.artifact_id}

    def propose_evidence_only(
        self,
        stable: HarnessState,
        episode: EpisodeRecord,
        rationale: str,
    ) -> CandidateUpdate:
        """Stage deterministic evidence when semantic reflection is unavailable."""
        proposed = copy.deepcopy(stable)
        proposed.version = stable.version + 1
        proposed.raw_memory.append(episode.to_dict())
        proposed.raw_memory = proposed.raw_memory[-self.raw_memory_capacity :]
        operations: List[Dict[str, Any]] = [
            {"kind": "add_raw_memory", "episode_id": episode.episode_id}
        ]
        if self.capability_learning_enabled:
            operations.extend(
                self.capability_evolution.record_observed_outcomes(proposed, episode)
            )
        return CandidateUpdate(
            candidate_id=f"candidate-v{proposed.version}-evidence-{uuid.uuid4().hex[:8]}",
            base_version=stable.version,
            proposed_state=proposed,
            operations=operations,
            rationale=[rationale],
        )

    @staticmethod
    def _apply_capability_evidence(
        proposed: HarnessState, reflection: Dict[str, Any], episode_id: str
    ) -> List[Dict[str, Any]]:
        updates = reflection.get("capability_evidence", [])
        if not isinstance(updates, list):
            return []
        by_id = {item.capability_id: item for item in proposed.capabilities}
        operations: List[Dict[str, Any]] = []
        for update in updates[:20]:
            if not isinstance(update, dict):
                continue
            capability_id = str(update.get("capability_id", ""))
            evidence = str(update.get("evidence", "")).strip()[:500]
            if capability_id not in by_id or not evidence:
                continue
            entry = f"{episode_id}: {evidence}"
            by_id[capability_id].validation_evidence = unique(
                by_id[capability_id].validation_evidence + [entry]
            )[-50:]
            operations.append(
                {"kind": "update_capability_evidence", "capability_id": capability_id}
            )
        return operations

    def propose(
        self,
        stable: HarnessState,
        episode: EpisodeRecord,
        final_context: RoutingContext,
    ) -> CandidateUpdate:
        proposed = copy.deepcopy(stable)
        proposed.version = stable.version + 1
        proposed.raw_memory.append(episode.to_dict())
        proposed.raw_memory = proposed.raw_memory[-self.raw_memory_capacity :]
        operations: List[Dict[str, Any]] = [
            {"kind": "add_raw_memory", "episode_id": episode.episode_id}
        ]
        if self.capability_learning_enabled:
            operations.extend(
                self.capability_evolution.record_observed_outcomes(proposed, episode)
            )

        reflection = self.llm.complete_json(
            "optimizer",
            (
                "You are the Continual Optimizer of an ALFWorld harness. Reflect on the entire "
                "grounded trajectory and propose reusable knowledge, not a transcript. For a "
                "successful episode, knowledge_type must be success_strategy and procedure must "
                "describe a generalized, ordered, conditional plan without scene-specific object "
                "IDs. For a failed episode, knowledge_type must be failure_boundary and you must "
                "not claim a successful procedure. Distinguish environmental evidence from "
                "speculation. You may refine one supplied strategy using target_strategy_id or "
                "create a new one. Also discover capabilities from the grounded environment "
                "trajectory and internal Task Interface, Router, Action Policy, Optimizer and "
                "Evaluator traces. The atomic_external capabilities are environment-defined and "
                "must never be recreated, renamed or deprecated. You may propose create/refine/"
                "deprecate operations for composite_external and internal capabilities. Create a "
                "capability only when positive raw-episode evidence supports a reusable function; "
                "do not equate your own self-description with evidence. A composite capability "
                "must cite existing dependency IDs. An internal capability must identify one of "
                "task_interface, router, action_policy, optimizer, evaluator as provider_component. "
                "New capabilities begin hypothesized; refinements may request provisional or "
                "validated status but evidence thresholds are enforced outside the model. "
                "Capability evidence may update only supplied IDs. "
                "Return JSON with knowledge_type, target_strategy_id, title, procedure, "
                "applicability, failure_boundaries, capability_evidence, capability_operations, "
                "rationale. Each capability operation must have operation and, for "
                "creation, kind, name, description, function, dependencies, procedure, inputs, "
                "outputs, preconditions, success_conditions, failure_modes, provider_component, "
                "positive_evidence_episode_ids, negative_evidence_episode_ids and confidence. "
                "procedure, applicability, and failure_boundaries must each be JSON arrays of "
                "plain strings; for a successful episode procedure must contain at least one "
                "ordered reusable step."
            ),
            {
                "episode": episode.to_dict(),
                "routed_context": final_context.to_dict(),
                "existing_strategies_for_task": [
                    self._strategy_summary(item)
                    for item in stable.abstract_memory
                    if item.task_type == episode.task_type
                ],
                "capabilities": [item.__dict__ for item in proposed.capabilities],
                "capability_lifecycle": {
                    "statuses": [
                        "hypothesized",
                        "provisional",
                        "validated",
                        "deprecated",
                    ],
                    "new_capability_status": "hypothesized",
                    "max_operations_this_episode": self.capability_evolution.max_operations,
                },
            },
        )
        strategy_operation = self._apply_strategy_reflection(proposed, episode, reflection)
        strategy_diagnostic = ""
        if episode.success and strategy_operation is None:
            if str(reflection.get("knowledge_type", "none")) != "success_strategy":
                strategy_diagnostic = "Skipped strategy: knowledge_type was not success_strategy."
            else:
                strategy_diagnostic = "Skipped strategy: response had no reusable procedure."
        episode.harness_traces.append(
            {
                "optimizer": {
                    "component": "llm_optimizer",
                    "knowledge_type": str(reflection.get("knowledge_type", "none")),
                    "rationale": str(reflection.get("rationale", "")),
                    "capability_operations": reflection.get("capability_operations", []),
                    "strategy_diagnostic": strategy_diagnostic,
                }
            }
        )
        for raw_episode in proposed.raw_memory:
            if raw_episode.get("episode_id") == episode.episode_id:
                raw_episode["harness_traces"] = copy.deepcopy(episode.harness_traces)
        if strategy_operation is not None:
            operations.append(strategy_operation)
        operations.extend(
            self._apply_capability_evidence(proposed, reflection, episode.episode_id)
        )
        if self.capability_learning_enabled:
            operations.extend(
                self.capability_evolution.apply_proposals(
                    proposed,
                    reflection,
                    model=str(getattr(self.llm, "default_model", "configured-llm")),
                )
            )

        # Anchor retention is a harness invariant, not an LLM preference. Stage one
        # anchor for every grounded success until the per-task capacity is reached.
        # If the evaluator rejects the candidate, the staged anchor is rejected with
        # it and stable state remains unchanged.
        if episode.success:
            per_task = [item for item in proposed.anchors if item.task_type == episode.task_type]
            strategy_id = strategy_operation.get("artifact_id") if strategy_operation else ""
            if len(per_task) < self.anchors_per_task and strategy_id:
                anchor = AnchorCase(
                    anchor_id=f"anchor-{uuid.uuid4().hex[:12]}",
                    task_type=episode.task_type,
                    goal=episode.goal,
                    evidence=(
                        episode.steps[-1].next_observation if episode.steps else episode.goal
                    ),
                    admissible_commands=(
                        episode.steps[-1].admissible_commands if episode.steps else []
                    ),
                    expected_capability_ids=[
                        item.capability_id for item in final_context.capabilities
                    ],
                    expected_strategy_ids=[strategy_id],
                    source_episode_id=episode.episode_id,
                )
                proposed.anchors.append(anchor)
                operations.append({"kind": "add_anchor", "anchor_id": anchor.anchor_id})

        return CandidateUpdate(
            candidate_id=f"candidate-v{proposed.version}-{uuid.uuid4().hex[:8]}",
            base_version=stable.version,
            proposed_state=proposed,
            operations=operations,
            rationale=[str(reflection.get("rationale", "LLM trajectory reflection."))],
        )
