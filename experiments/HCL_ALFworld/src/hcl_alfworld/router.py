from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .llm import StructuredLLMProtocol
from .memory import ExperienceMemory
from .schemas import (
    CapabilityUnit,
    HarnessState,
    RoutingContext,
    StepRecord,
    StrategyArtifact,
    TaskEvidence,
)


class AdaptiveRouter:
    """LLM semantic routing over bounded, auditable candidate pools."""

    def __init__(
        self,
        llm: StructuredLLMProtocol,
        memory: ExperienceMemory,
        memory_enabled: bool = True,
        raw_memory_retrieval_enabled: bool = True,
        strategy_retrieval_enabled: bool = True,
        capabilities_enabled: bool = True,
        candidate_pool_size: int = 20,
        gate_mode: str = "stability",
    ):
        self.llm = llm
        self.memory = memory
        self.memory_enabled = memory_enabled
        self.raw_memory_retrieval_enabled = raw_memory_retrieval_enabled
        self.strategy_retrieval_enabled = strategy_retrieval_enabled
        self.capabilities_enabled = capabilities_enabled
        self.candidate_pool_size = candidate_pool_size
        self.gate_mode = gate_mode

    @staticmethod
    def _raw_summary(item: Dict[str, Any]) -> Dict[str, Any]:
        steps = item.get("steps", [])
        return {
            "episode_id": item.get("episode_id"),
            "task_type": item.get("task_type"),
            "goal": item.get("goal"),
            "success": item.get("success"),
            "failure_reason": item.get("failure_reason"),
            "actions": [step.get("action") for step in steps],
        }

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
    def _capability_summary(item: CapabilityUnit) -> Dict[str, Any]:
        return {
            "capability_id": item.capability_id,
            "name": item.name,
            "kind": item.kind,
            "description": item.description,
            "function": item.function,
            "action_prefixes": item.action_prefixes,
            "dependencies": item.dependencies,
            "procedure": item.procedure,
            "preconditions": item.preconditions,
            "success_conditions": item.success_conditions,
            "failure_modes": item.failure_modes,
            "positive_evidence_episode_ids": item.positive_evidence_episode_ids,
            "negative_evidence_episode_ids": item.negative_evidence_episode_ids,
            "status": item.status,
            "confidence": item.confidence,
            "provider_component": item.provider_component,
            "validation_evidence": item.validation_evidence[-5:],
        }

    @staticmethod
    def _select_by_id(items: List[Any], selected: Any, attribute: str, limit: int) -> List[Any]:
        requested = selected if isinstance(selected, list) else []
        by_id = {str(getattr(item, attribute)): item for item in items}
        return [by_id[item_id] for item_id in requested if item_id in by_id][:limit]

    @staticmethod
    def _command_matches_prefix(command: str, prefix: str) -> bool:
        normalized_command = " ".join(command.lower().split())
        normalized_prefix = " ".join(prefix.lower().split())
        return bool(normalized_prefix) and (
            normalized_command == normalized_prefix
            or normalized_command.startswith(f"{normalized_prefix} ")
        )

    def _grounded_atomic_capabilities(
        self, task: TaskEvidence, state: HarnessState
    ) -> List[CapabilityUnit]:
        if not self.capabilities_enabled:
            return []
        return [
            item
            for item in state.capabilities
            if item.kind == "atomic_external"
            and item.availability == "available"
            and item.status != "deprecated"
            and any(
                self._command_matches_prefix(command, prefix)
                for command in task.admissible_commands
                for prefix in item.action_prefixes
            )
        ]

    def _semantic_capability_pool(self, state: HarnessState) -> List[CapabilityUnit]:
        if not self.capabilities_enabled:
            return []
        return [
            item
            for item in state.capabilities
            if item.kind != "atomic_external"
            and item.availability == "available"
            and item.status != "deprecated"
            and (
                self.gate_mode == "stability"
                or item.status in {"provisional", "validated"}
            )
        ]

    @staticmethod
    def _unique_capabilities(items: Sequence[CapabilityUnit]) -> List[CapabilityUnit]:
        unique: List[CapabilityUnit] = []
        seen = set()
        for item in items:
            if item.capability_id in seen:
                continue
            seen.add(item.capability_id)
            unique.append(item)
        return unique

    @staticmethod
    def reroute_reasons(
        task: TaskEvidence,
        previous_task: Optional[TaskEvidence],
        steps: Sequence[StepRecord],
    ) -> List[str]:
        """Return grounded events that justify refreshing semantic episode context."""
        if previous_task is None:
            return ["episode_start"]

        reasons: List[str] = []
        current = task.progress if isinstance(task.progress, dict) else {}
        previous = (
            previous_task.progress if isinstance(previous_task.progress, dict) else {}
        )

        def normalized_items(progress: Dict[str, Any], name: str) -> set[str]:
            value = progress.get(name, [])
            if not isinstance(value, list):
                return set()
            return {
                str(item).strip() for item in value if isinstance(item, str) and item.strip()
            }

        if bool(current.get("loop_risk")) and not bool(previous.get("loop_risk")):
            reasons.append("loop_risk_detected")

        current_failures = normalized_items(current, "failed_or_unproductive_actions")
        previous_failures = normalized_items(previous, "failed_or_unproductive_actions")
        if current_failures - previous_failures:
            reasons.append("new_failure_evidence")

        current_completed = normalized_items(current, "completed_subgoals")
        previous_completed = normalized_items(previous, "completed_subgoals")
        if current_completed - previous_completed:
            reasons.append("subgoal_completed")

        if normalized_items(current, "inventory") != normalized_items(
            previous, "inventory"
        ):
            reasons.append("inventory_changed")

        current_count = current.get("completed_count")
        previous_count = previous.get("completed_count")
        if (
            isinstance(current_count, int)
            and not isinstance(current_count, bool)
            and isinstance(previous_count, int)
            and not isinstance(previous_count, bool)
            and current_count != previous_count
        ):
            reasons.append("completion_count_changed")

        if steps:
            latest = steps[-1]
            latest_signature = (
                " ".join(latest.action.lower().split()),
                " ".join(latest.next_observation.lower().split()),
            )
            prior_signatures = {
                (
                    " ".join(item.action.lower().split()),
                    " ".join(item.next_observation.lower().split()),
                )
                for item in steps[:-1]
            }
            if latest_signature in prior_signatures:
                reasons.append("repeated_transition")

        if len(steps) >= 2:
            recent = steps[-2:]
            no_progress = all(
                item.reward <= 0.0
                and " ".join(item.observation.lower().split())
                == " ".join(item.next_observation.lower().split())
                for item in recent
            )
            if no_progress:
                reasons.append("consecutive_no_progress")

        return list(dict.fromkeys(reasons))

    @staticmethod
    def should_refresh_semantic_context(
        reasons: Sequence[str],
        current_step: int,
        last_semantic_route_step: int,
    ) -> bool:
        """Immediately honor new evidence; lightly debounce repeated-stall recovery."""
        if not reasons:
            return False
        recovery_reasons = {"repeated_transition", "consecutive_no_progress"}
        if not set(reasons).issubset(recovery_reasons):
            return True
        return current_step - last_semantic_route_step >= 2

    def route(
        self,
        task: TaskEvidence,
        state: HarnessState,
        trigger_reasons: Optional[Sequence[str]] = None,
    ) -> RoutingContext:
        raw_pool = (
            self.memory.raw_candidates(task, state, self.candidate_pool_size)
            if self.memory_enabled and self.raw_memory_retrieval_enabled
            else []
        )
        strategy_pool = (
            self.memory.strategy_candidates(task, state, self.candidate_pool_size)
            if self.memory_enabled and self.strategy_retrieval_enabled
            else []
        )
        capability_pool = self._semantic_capability_pool(state)
        atomic_capabilities = self._grounded_atomic_capabilities(task, state)
        reasons = list(trigger_reasons or ["explicit_route"])
        if not raw_pool and not strategy_pool and not capability_pool:
            trace = {
                "component": "adaptive_router",
                "mode": "deterministic_only",
                "llm_called": False,
                "trigger_reasons": reasons,
                "candidate_memory_ids": [],
                "candidate_strategy_ids": [],
                "candidate_capability_ids": [],
                "selected_memory_ids": [],
                "selected_strategy_ids": [],
                "selected_capability_ids": [
                    item.capability_id for item in atomic_capabilities
                ],
                "grounded_atomic_capability_ids": [
                    item.capability_id for item in atomic_capabilities
                ],
                "semantic_context_source_step": task.step,
                "rationale": (
                    "No semantic memory, strategy, or learned capability candidates were "
                    "available; atomic capabilities were grounded from admissible commands."
                ),
            }
            return RoutingContext(task, [], [], atomic_capabilities, trace)

        result = self.llm.complete_json(
            "router",
            (
                "You are the Adaptive Router of an ALFWorld continual-learning harness. "
                "Semantically select only context that can help the agent's next decision. "
                "Prefer evidence matching the current goal, progress state, objects and required "
                "operations; use the current episode_history to avoid loops and reject "
                "superficially similar but stale or contradictory memories. "
                "Select learned capabilities that are relevant to the episode or immediately "
                "needed for planning. Atomic environment capabilities have already been grounded "
                "deterministically from admissible commands and must not be selected here. Treat "
                "hypothesized and provisional model-generated capabilities as uncertain advice; "
                "prefer validated capabilities when equally applicable, and respect dependencies "
                "and failure boundaries. Internal capabilities describe reasoning support, while "
                "composite external capabilities describe multi-step skills. Return JSON with "
                "memory_ids, strategy_ids, capability_ids, rationale. Use only supplied IDs."
            ),
            {
                "task": task.to_dict(),
                "raw_memory_candidates": [self._raw_summary(item) for item in raw_pool],
                "strategy_candidates": [self._strategy_summary(item) for item in strategy_pool],
                "capability_candidates": [
                    self._capability_summary(item) for item in capability_pool
                ],
                "grounded_atomic_capabilities": [
                    self._capability_summary(item) for item in atomic_capabilities
                ],
                "selection_limits": {
                    "raw_memory": (
                        self.memory.top_k_memory if self.raw_memory_retrieval_enabled else 0
                    ),
                    "strategies": (
                        self.memory.top_k_strategies if self.strategy_retrieval_enabled else 0
                    ),
                },
            },
        )
        raw_by_id = {str(item.get("episode_id")): item for item in raw_pool}
        memory_ids = result.get("memory_ids", [])
        memories = [
            raw_by_id[item_id]
            for item_id in memory_ids if isinstance(item_id, str) and item_id in raw_by_id
        ][: self.memory.top_k_memory]
        strategies = self._select_by_id(
            strategy_pool,
            result.get("strategy_ids", []),
            "artifact_id",
            self.memory.top_k_strategies,
        )
        capabilities = self._select_by_id(
            capability_pool,
            result.get("capability_ids", []),
            "capability_id",
            len(capability_pool),
        )
        capabilities = self._unique_capabilities(capabilities + atomic_capabilities)
        trace = {
            "component": "adaptive_router",
            "mode": "semantic_route",
            "llm_called": True,
            "trigger_reasons": reasons,
            "candidate_memory_ids": [item.get("episode_id") for item in raw_pool],
            "candidate_strategy_ids": [item.artifact_id for item in strategy_pool],
            "candidate_capability_ids": [
                item.capability_id for item in capability_pool
            ],
            "selected_memory_ids": [item.get("episode_id") for item in memories],
            "selected_strategy_ids": [item.artifact_id for item in strategies],
            "selected_capability_ids": [item.capability_id for item in capabilities],
            "grounded_atomic_capability_ids": [
                item.capability_id for item in atomic_capabilities
            ],
            "semantic_context_source_step": task.step,
            "rationale": str(result.get("rationale", "")),
        }
        return RoutingContext(task, memories, strategies, capabilities, trace)

    def reuse(
        self,
        task: TaskEvidence,
        state: HarnessState,
        previous: RoutingContext,
    ) -> RoutingContext:
        """Reuse episode-level semantic context while refreshing grounded atomics."""
        available = {
            item.capability_id: item
            for item in state.capabilities
            if item.availability == "available" and item.status != "deprecated"
            and (
                item.kind == "atomic_external"
                or self.gate_mode == "stability"
                or item.status in {"provisional", "validated"}
            )
        }
        semantic_capabilities = [
            available[item.capability_id]
            for item in previous.capabilities
            if item.kind != "atomic_external" and item.capability_id in available
        ]
        atomic_capabilities = self._grounded_atomic_capabilities(task, state)
        capabilities = self._unique_capabilities(
            semantic_capabilities + atomic_capabilities
        )
        trace = {
            "component": "adaptive_router",
            "mode": "cached_context",
            "llm_called": False,
            "trigger_reasons": [],
            "candidate_memory_ids": [],
            "candidate_strategy_ids": [],
            "candidate_capability_ids": [],
            "selected_memory_ids": [
                item.get("episode_id") for item in previous.memories
            ],
            "selected_strategy_ids": [
                item.artifact_id for item in previous.strategies
            ],
            "selected_capability_ids": [
                item.capability_id for item in capabilities
            ],
            "grounded_atomic_capability_ids": [
                item.capability_id for item in atomic_capabilities
            ],
            "semantic_context_source_step": previous.trace.get(
                "semantic_context_source_step", previous.task.step
            ),
            "rationale": "Reused episode-level semantic context and refreshed atomic capabilities.",
        }
        return RoutingContext(
            task,
            list(previous.memories),
            list(previous.strategies),
            capabilities,
            trace,
        )
