from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Set

from .schemas import CapabilityUnit, EpisodeRecord, HarnessState
from .text import unique


CAPABILITY_KINDS = {"atomic_external", "composite_external", "internal"}
CAPABILITY_STATUSES = {"hypothesized", "provisional", "validated", "deprecated"}
INTERNAL_PROVIDERS = {
    "task_interface",
    "router",
    "action_policy",
    "optimizer",
    "evaluator",
}


class CapabilityEvolution:
    """Apply LLM-proposed capability evolution under evidence and graph invariants."""

    def __init__(
        self,
        max_operations: int = 3,
        provisional_min_positive_evidence: int = 2,
        validated_min_positive_evidence: int = 3,
        validated_min_success_rate: float = 0.67,
    ):
        self.max_operations = max_operations
        self.provisional_min_positive_evidence = provisional_min_positive_evidence
        self.validated_min_positive_evidence = validated_min_positive_evidence
        self.validated_min_success_rate = validated_min_success_rate

    @staticmethod
    def _strings(value: Any, limit: int = 30) -> List[str]:
        if not isinstance(value, list):
            return []
        return unique(
            str(item).strip()[:500]
            for item in value[:limit]
            if isinstance(item, str) and item.strip()
        )

    @staticmethod
    def _normalized_name(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    @staticmethod
    def _known_episode_ids(state: HarnessState) -> Set[str]:
        return {
            str(item.get("episode_id"))
            for item in state.raw_memory
            if item.get("episode_id")
        }

    @staticmethod
    def _episode_selected_capabilities(episode: EpisodeRecord) -> Set[str]:
        selected: Set[str] = set()
        for trace in episode.harness_traces:
            router = trace.get("router", {}) if isinstance(trace, dict) else {}
            for capability_id in router.get("selected_capability_ids", []):
                if isinstance(capability_id, str):
                    selected.add(capability_id)
        return selected

    @staticmethod
    def _episode_internal_components(episode: EpisodeRecord) -> Set[str]:
        used: Set[str] = set()
        for trace in episode.harness_traces:
            if not isinstance(trace, dict):
                continue
            for component in INTERNAL_PROVIDERS:
                if trace.get(component):
                    used.add(component)
        return used

    @staticmethod
    def _raw_episode_supports_provider(
        state: HarnessState, episode_id: str, provider: str
    ) -> bool:
        raw = next(
            (
                item
                for item in state.raw_memory
                if str(item.get("episode_id", "")) == episode_id
            ),
            None,
        )
        if raw is None:
            return False
        traces = raw.get("harness_traces", [])
        return any(isinstance(trace, dict) and trace.get(provider) for trace in traces)

    def record_observed_outcomes(
        self, state: HarnessState, episode: EpisodeRecord
    ) -> List[Dict[str, Any]]:
        selected = self._episode_selected_capabilities(episode)
        internal_components = self._episode_internal_components(episode)
        operations: List[Dict[str, Any]] = []
        for capability in state.capabilities:
            was_used = capability.capability_id in selected or (
                capability.kind == "internal"
                and capability.provider_component in internal_components
            )
            if not was_used:
                continue
            capability.usage_count += 1
            if episode.success:
                capability.success_count += 1
                capability.positive_evidence_episode_ids = unique(
                    capability.positive_evidence_episode_ids + [episode.episode_id]
                )
            else:
                capability.failure_count += 1
                capability.negative_evidence_episode_ids = unique(
                    capability.negative_evidence_episode_ids + [episode.episode_id]
                )
            capability.updated_at_version = state.version
            operations.append(
                {
                    "kind": "record_capability_outcome",
                    "capability_id": capability.capability_id,
                    "success": episode.success,
                    "episode_id": episode.episode_id,
                }
            )
        return operations

    def recalibrate_lifecycle(self, state: HarnessState) -> None:
        """Promote or demote learned capabilities from complete outcome evidence."""
        for capability in state.capabilities:
            if capability.kind == "atomic_external" or capability.status == "deprecated":
                continue
            capability.status = self._max_evidence_status(capability)
            positive = len(set(capability.positive_evidence_episode_ids))
            negative = len(set(capability.negative_evidence_episode_ids))
            # A small beta prior avoids assigning certainty from only a few
            # episodes while still allowing evidence-driven lifecycle changes.
            evidence_confidence = (positive + 1) / (positive + negative + 2)
            capability.confidence = self._bounded_confidence(
                capability.status, evidence_confidence
            )

    def _max_evidence_status(self, capability: CapabilityUnit) -> str:
        positive = len(set(capability.positive_evidence_episode_ids))
        negative = len(set(capability.negative_evidence_episode_ids))
        rate = positive / max(1, positive + negative)
        if (
            positive >= self.validated_min_positive_evidence
            and rate >= self.validated_min_success_rate
        ):
            return "validated"
        if positive >= self.provisional_min_positive_evidence:
            return "provisional"
        return "hypothesized"

    def _bounded_status(self, capability: CapabilityUnit, requested: str) -> str:
        order = {"hypothesized": 0, "provisional": 1, "validated": 2}
        evidence_max = self._max_evidence_status(capability)
        if requested not in order:
            requested = capability.status if capability.status in order else "hypothesized"
        return min((requested, evidence_max), key=lambda item: order[item])

    @staticmethod
    def _bounded_confidence(status: str, value: Any) -> float:
        try:
            confidence = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            confidence = 0.0
        ceiling = {"hypothesized": 0.49, "provisional": 0.79}.get(status, 1.0)
        return min(confidence, ceiling)

    def _create(
        self,
        state: HarnessState,
        proposal: Dict[str, Any],
        model: str,
        known_episode_ids: Set[str],
    ) -> Dict[str, Any] | None:
        kind = str(proposal.get("kind", ""))
        if kind not in {"composite_external", "internal"}:
            return None
        name = str(proposal.get("name", "")).strip()[:200]
        if not name:
            return None
        normalized_names = {
            self._normalized_name(item.name) for item in state.capabilities
            if item.status != "deprecated"
        }
        if self._normalized_name(name) in normalized_names:
            return None
        by_id = {item.capability_id: item for item in state.capabilities}
        dependencies = [
            item for item in self._strings(proposal.get("dependencies"), 20) if item in by_id
        ]
        provider = str(proposal.get("provider_component", "")).strip()
        if kind == "composite_external" and not dependencies:
            return None
        if kind == "internal" and provider not in INTERNAL_PROVIDERS:
            return None
        procedure = self._strings(proposal.get("procedure"), 30)
        inputs = self._strings(proposal.get("inputs"), 20)
        outputs = self._strings(proposal.get("outputs"), 20)
        success_conditions = self._strings(proposal.get("success_conditions"), 20)
        if not procedure or not inputs or not outputs or not success_conditions:
            return None
        positive = [
            item
            for item in self._strings(proposal.get("positive_evidence_episode_ids"), 50)
            if item in known_episode_ids
        ]
        negative = [
            item
            for item in self._strings(proposal.get("negative_evidence_episode_ids"), 50)
            if item in known_episode_ids
        ]
        if not positive:
            return None
        if kind == "internal":
            positive = [
                episode_id
                for episode_id in positive
                if self._raw_episode_supports_provider(state, episode_id, provider)
            ]
            if not positive:
                return None
        atomic_prefixes = {
            prefix
            for item in state.capabilities
            if item.kind == "atomic_external"
            for prefix in item.action_prefixes
        }
        capability = CapabilityUnit(
            capability_id=f"capability-{uuid.uuid4().hex[:12]}",
            name=name,
            kind=kind,
            function=str(proposal.get("function", proposal.get("description", "")))[:500],
            action_prefixes=[
                prefix
                for prefix in self._strings(proposal.get("action_prefixes"), 20)
                if prefix in atomic_prefixes
            ],
            input_output="LLM-mediated structured capability",
            preconditions=self._strings(proposal.get("preconditions"), 30),
            failure_modes=self._strings(proposal.get("failure_modes"), 30),
            version=f"harness-{state.version}",
            description=str(proposal.get("description", ""))[:1000],
            dependencies=dependencies,
            procedure=procedure,
            inputs=inputs,
            outputs=outputs,
            success_conditions=success_conditions,
            positive_evidence_episode_ids=positive,
            negative_evidence_episode_ids=negative,
            usage_count=len(set(positive + negative)),
            success_count=len(set(positive)),
            failure_count=len(set(negative)),
            confidence=self._bounded_confidence("hypothesized", proposal.get("confidence", 0.0)),
            status="hypothesized",
            provider_component=provider,
            model=model,
            prompt_version="capability-induction-v1",
            created_at_version=state.version,
            updated_at_version=state.version,
        )
        state.capabilities.append(capability)
        return {"kind": "create_capability", "capability_id": capability.capability_id}

    def _refine(
        self,
        state: HarnessState,
        proposal: Dict[str, Any],
        known_episode_ids: Set[str],
    ) -> Dict[str, Any] | None:
        capability_id = str(proposal.get("capability_id", ""))
        target = next(
            (item for item in state.capabilities if item.capability_id == capability_id), None
        )
        if target is None or target.kind == "atomic_external" or target.status == "deprecated":
            return None
        dependencies = self._strings(proposal.get("dependencies"), 20)
        known_capabilities = {item.capability_id for item in state.capabilities}
        if dependencies:
            target.dependencies = [
                item
                for item in dependencies
                if item in known_capabilities and item != capability_id
            ]
        for field_name in (
            "procedure",
            "inputs",
            "outputs",
            "preconditions",
            "failure_modes",
            "success_conditions",
        ):
            value = self._strings(proposal.get(field_name), 30)
            if value:
                setattr(target, field_name, value)
        for field_name in ("name", "description", "function"):
            value = str(proposal.get(field_name, "")).strip()
            if value:
                setattr(target, field_name, value[:1000])
        positive = [
            item
            for item in self._strings(proposal.get("positive_evidence_episode_ids"), 50)
            if item in known_episode_ids
        ]
        negative = [
            item
            for item in self._strings(proposal.get("negative_evidence_episode_ids"), 50)
            if item in known_episode_ids
        ]
        target.positive_evidence_episode_ids = unique(
            target.positive_evidence_episode_ids + positive
        )
        target.negative_evidence_episode_ids = unique(
            target.negative_evidence_episode_ids + negative
        )
        target.success_count = len(set(target.positive_evidence_episode_ids))
        target.failure_count = len(set(target.negative_evidence_episode_ids))
        target.usage_count = len(
            set(target.positive_evidence_episode_ids + target.negative_evidence_episode_ids)
        )
        requested_status = str(proposal.get("requested_status", target.status))
        target.status = self._bounded_status(target, requested_status)
        target.confidence = self._bounded_confidence(
            target.status, proposal.get("confidence", target.confidence)
        )
        target.updated_at_version = state.version
        return {
            "kind": "refine_capability",
            "capability_id": target.capability_id,
            "status": target.status,
        }

    def _deprecate(
        self,
        state: HarnessState,
        proposal: Dict[str, Any],
        known_episode_ids: Set[str],
    ) -> Dict[str, Any] | None:
        capability_id = str(proposal.get("capability_id", ""))
        target = next(
            (item for item in state.capabilities if item.capability_id == capability_id), None
        )
        rationale = str(proposal.get("rationale", "")).strip()
        negative = [
            item
            for item in self._strings(proposal.get("negative_evidence_episode_ids"), 50)
            if item in known_episode_ids
        ]
        if (
            target is None
            or target.kind == "atomic_external"
            or target.status == "deprecated"
            or not rationale
        ):
            return None
        target.negative_evidence_episode_ids = unique(
            target.negative_evidence_episode_ids + negative
        )
        if not target.negative_evidence_episode_ids:
            return None
        target.status = "deprecated"
        target.availability = "unavailable"
        target.confidence = 0.0
        target.failure_modes = unique(target.failure_modes + [f"Deprecated: {rationale[:500]}"])
        target.updated_at_version = state.version
        return {"kind": "deprecate_capability", "capability_id": target.capability_id}

    def apply_proposals(
        self,
        state: HarnessState,
        reflection: Dict[str, Any],
        model: str,
    ) -> List[Dict[str, Any]]:
        proposals = reflection.get("capability_operations", [])
        if not isinstance(proposals, list):
            return []
        known_episode_ids = self._known_episode_ids(state)
        operations: List[Dict[str, Any]] = []
        for proposal in proposals[: self.max_operations]:
            if not isinstance(proposal, dict):
                continue
            operation_kind = str(proposal.get("operation", ""))
            if operation_kind == "create":
                applied = self._create(state, proposal, model, known_episode_ids)
            elif operation_kind == "refine":
                applied = self._refine(state, proposal, known_episode_ids)
            elif operation_kind == "deprecate":
                applied = self._deprecate(state, proposal, known_episode_ids)
            else:
                applied = None
            if applied is not None:
                operations.append(applied)
        return operations
