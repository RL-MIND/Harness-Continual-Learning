from __future__ import annotations

from typing import Any, Dict, List

from .capability_evolution import CAPABILITY_KINDS, CAPABILITY_STATUSES, INTERNAL_PROVIDERS
from .llm import StructuredLLMProtocol
from .schemas import CandidateUpdate, EpisodeRecord, EvaluationResult, HarnessState


class ContinualEvaluator:
    """LLM semantic review constrained by non-negotiable state-integrity checks."""

    def __init__(
        self,
        llm: StructuredLLMProtocol,
        min_current_score: float = 1.0,
        max_anchor_regression: float = 0.0,
        provisional_min_positive_evidence: int = 2,
        validated_min_positive_evidence: int = 3,
        validated_min_success_rate: float = 0.67,
        gate_mode: str = "stability",
    ):
        self.llm = llm
        self.min_current_score = min_current_score
        self.max_anchor_regression = max_anchor_regression
        self.provisional_min_positive_evidence = provisional_min_positive_evidence
        self.validated_min_positive_evidence = validated_min_positive_evidence
        self.validated_min_success_rate = validated_min_success_rate
        self.gate_mode = gate_mode

    @staticmethod
    def _clean_string_list(value: Any) -> List[str]:
        """Normalize a JSON string or string array without splitting scalars."""
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        cleaned: List[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if text and text not in cleaned:
                cleaned.append(text)
        return cleaned

    def _validate_structure(
        self, stable: HarnessState, candidate: CandidateUpdate
    ) -> List[str]:
        problems: List[str] = []
        proposed = candidate.proposed_state
        if proposed.version != candidate.base_version + 1:
            problems.append("candidate version is not exactly base_version + 1")
        for label, identifiers in (
            ("strategy", [item.artifact_id for item in proposed.abstract_memory]),
            ("capability", [item.capability_id for item in proposed.capabilities]),
            ("anchor", [item.anchor_id for item in proposed.anchors]),
        ):
            if len(identifiers) != len(set(identifiers)):
                problems.append(f"duplicate {label} identifiers")
        stable_capabilities = {item.capability_id: item for item in stable.capabilities}
        proposed_capabilities = {item.capability_id: item for item in proposed.capabilities}
        missing_existing = set(stable_capabilities) - set(proposed_capabilities)
        if missing_existing:
            problems.append(f"candidate deleted capabilities: {sorted(missing_existing)}")

        raw_episode_ids = {
            str(item.get("episode_id"))
            for item in proposed.raw_memory
            if item.get("episode_id")
        }
        raw_by_id = {
            str(item.get("episode_id")): item
            for item in proposed.raw_memory
            if item.get("episode_id")
        }
        normalized_names: Dict[str, str] = {}
        atomic_prefixes = {
            prefix
            for item in proposed.capabilities
            if item.kind == "atomic_external"
            for prefix in item.action_prefixes
        }
        for capability in proposed.capabilities:
            if capability.kind not in CAPABILITY_KINDS:
                problems.append(
                    f"{capability.capability_id}: unsupported kind {capability.kind}"
                )
            if capability.status not in CAPABILITY_STATUSES:
                problems.append(
                    f"{capability.capability_id}: unsupported status {capability.status}"
                )
            if not 0.0 <= capability.confidence <= 1.0:
                problems.append(f"{capability.capability_id}: confidence outside [0, 1]")
            normalized_name = " ".join(capability.name.lower().split())
            if normalized_name in normalized_names and capability.status != "deprecated":
                problems.append(
                    f"duplicate capability names: {normalized_names[normalized_name]} and "
                    f"{capability.capability_id}"
                )
            normalized_names[normalized_name] = capability.capability_id
            unknown_dependencies = set(capability.dependencies) - set(proposed_capabilities)
            if unknown_dependencies:
                problems.append(
                    f"{capability.capability_id}: unknown dependencies "
                    f"{sorted(unknown_dependencies)}"
                )
            if capability.capability_id in capability.dependencies:
                problems.append(f"{capability.capability_id}: self dependency")
            unknown_evidence = (
                set(capability.positive_evidence_episode_ids)
                | set(capability.negative_evidence_episode_ids)
            ) - raw_episode_ids
            if unknown_evidence:
                problems.append(
                    f"{capability.capability_id}: unknown evidence episodes "
                    f"{sorted(unknown_evidence)}"
                )
            if capability.kind != "atomic_external":
                invented_prefixes = set(capability.action_prefixes) - atomic_prefixes
                if invented_prefixes:
                    problems.append(
                        f"{capability.capability_id}: invented action prefixes "
                        f"{sorted(invented_prefixes)}"
                    )
                if not capability.procedure:
                    problems.append(f"{capability.capability_id}: missing procedure")
                if not capability.inputs or not capability.outputs:
                    problems.append(f"{capability.capability_id}: missing inputs or outputs")
                if not capability.success_conditions:
                    problems.append(f"{capability.capability_id}: missing success conditions")
                positive = len(set(capability.positive_evidence_episode_ids))
                negative = len(set(capability.negative_evidence_episode_ids))
                rate = positive / max(1, positive + negative)
                if (
                    capability.status in {"provisional", "validated"}
                    and positive < self.provisional_min_positive_evidence
                ):
                    problems.append(
                        f"{capability.capability_id}: insufficient evidence for provisional"
                    )
                if capability.status == "validated" and (
                    positive < self.validated_min_positive_evidence
                    or rate < self.validated_min_success_rate
                ):
                    problems.append(
                        f"{capability.capability_id}: insufficient evidence for validated"
                    )
                if capability.kind == "internal":
                    if capability.provider_component not in INTERNAL_PROVIDERS:
                        problems.append(
                            f"{capability.capability_id}: unsupported internal provider"
                        )
                    unsupported = []
                    for episode_id in capability.positive_evidence_episode_ids:
                        traces = raw_by_id.get(episode_id, {}).get("harness_traces", [])
                        if not any(
                            isinstance(trace, dict)
                            and trace.get(capability.provider_component)
                            for trace in traces
                        ):
                            unsupported.append(episode_id)
                    if unsupported:
                        problems.append(
                            f"{capability.capability_id}: internal evidence lacks provider trace "
                            f"{unsupported}"
                        )
            previous = stable_capabilities.get(capability.capability_id)
            if previous is not None and previous.kind == "atomic_external":
                immutable_before = (
                    previous.name,
                    previous.kind,
                    previous.function,
                    previous.action_prefixes,
                    previous.preconditions,
                    previous.failure_modes,
                    previous.status,
                    previous.availability,
                )
                immutable_after = (
                    capability.name,
                    capability.kind,
                    capability.function,
                    capability.action_prefixes,
                    capability.preconditions,
                    capability.failure_modes,
                    capability.status,
                    capability.availability,
                )
                if immutable_before != immutable_after:
                    problems.append(
                        f"{capability.capability_id}: atomic capability semantics are immutable"
                    )

        graph = {
            item.capability_id: list(item.dependencies)
            for item in proposed.capabilities
            if item.status != "deprecated"
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            if any(visit(dependency) for dependency in graph.get(node, []) if dependency in graph):
                return True
            visiting.remove(node)
            visited.add(node)
            return False

        if any(visit(node) for node in graph if node not in visited):
            problems.append("capability dependency graph contains a cycle")
        for capability in proposed.capabilities:
            if capability.status == "deprecated":
                continue
            deprecated_dependencies = [
                dependency
                for dependency in capability.dependencies
                if proposed_capabilities[dependency].status == "deprecated"
            ]
            if deprecated_dependencies:
                problems.append(
                    f"{capability.capability_id}: depends on deprecated capabilities "
                    f"{deprecated_dependencies}"
                )
        return problems

    @staticmethod
    def _hard_anchor_failures(
        stable: HarnessState, candidate: CandidateUpdate
    ) -> Dict[str, str]:
        capability_ids = {item.capability_id for item in candidate.proposed_state.capabilities}
        strategy_ids = {item.artifact_id for item in candidate.proposed_state.abstract_memory}
        failures: Dict[str, str] = {}
        for anchor in stable.anchors:
            missing_capabilities = set(anchor.expected_capability_ids) - capability_ids
            missing_strategies = set(anchor.expected_strategy_ids) - strategy_ids
            if missing_capabilities or missing_strategies:
                failures[anchor.anchor_id] = (
                    f"missing_capabilities={sorted(missing_capabilities)}, "
                    f"missing_strategies={sorted(missing_strategies)}"
                )
        return failures

    @staticmethod
    def _candidate_summary(candidate: CandidateUpdate) -> Dict[str, Any]:
        state = candidate.proposed_state
        return {
            "candidate_id": candidate.candidate_id,
            "base_version": candidate.base_version,
            "proposed_version": state.version,
            "operations": candidate.operations,
            "rationale": candidate.rationale,
            "strategies": [
                {
                    "artifact_id": item.artifact_id,
                    "task_type": item.task_type,
                    "title": item.title,
                    "procedure": item.procedure,
                    "applicability": item.applicability,
                    "failure_boundaries": item.failure_boundaries,
                    "evidence_episode_ids": item.evidence_episode_ids,
                }
                for item in state.abstract_memory
            ],
            "capabilities": [
                {
                    "capability_id": item.capability_id,
                    "name": item.name,
                    "kind": item.kind,
                    "description": item.description,
                    "dependencies": item.dependencies,
                    "procedure": item.procedure,
                    "inputs": item.inputs,
                    "outputs": item.outputs,
                    "preconditions": item.preconditions,
                    "success_conditions": item.success_conditions,
                    "failure_modes": item.failure_modes,
                    "positive_evidence_episode_ids": item.positive_evidence_episode_ids,
                    "negative_evidence_episode_ids": item.negative_evidence_episode_ids,
                    "status": item.status,
                    "confidence": item.confidence,
                    "provider_component": item.provider_component,
                    "validation_evidence": item.validation_evidence[-10:],
                }
                for item in state.capabilities
            ],
        }

    @staticmethod
    def _clean_index_list(value: Any, upper_bound: int) -> List[int]:
        if not isinstance(value, list):
            return []
        cleaned: List[int] = []
        for item in value:
            if isinstance(item, bool):
                continue
            try:
                index = int(item)
            except (TypeError, ValueError):
                continue
            if 0 <= index < upper_bound and index not in cleaned:
                cleaned.append(index)
        return cleaned

    @classmethod
    def _candidate_delta(
        cls, stable: HarnessState, candidate: CandidateUpdate
    ) -> Dict[str, Any]:
        proposed = candidate.proposed_state
        strategies = {item.artifact_id: item for item in proposed.abstract_memory}
        capabilities = {item.capability_id: item for item in proposed.capabilities}
        operations: List[Dict[str, Any]] = []
        for index, operation in enumerate(candidate.operations):
            entry: Dict[str, Any] = {"operation_index": index, **operation}
            strategy = strategies.get(str(operation.get("artifact_id", "")))
            capability = capabilities.get(str(operation.get("capability_id", "")))
            if strategy is not None:
                entry["strategy"] = {
                    "artifact_id": strategy.artifact_id,
                    "task_type": strategy.task_type,
                    "title": strategy.title,
                    "procedure": strategy.procedure,
                    "applicability": strategy.applicability,
                    "failure_boundaries": strategy.failure_boundaries,
                    "evidence_episode_ids": strategy.evidence_episode_ids,
                }
            if (
                capability is not None
                and operation.get("kind") != "record_capability_outcome"
            ):
                entry["capability"] = next(
                    item
                    for item in cls._candidate_summary(candidate)["capabilities"]
                    if item["capability_id"] == capability.capability_id
                )
            operations.append(entry)
        return {
            "candidate_id": candidate.candidate_id,
            "base_version": candidate.base_version,
            "proposed_version": proposed.version,
            "operations": operations,
            "rationale": candidate.rationale,
            "stable_counts": {
                "raw_memory": len(stable.raw_memory),
                "strategies": len(stable.abstract_memory),
                "capabilities": len(stable.capabilities),
                "anchors": len(stable.anchors),
            },
        }

    @staticmethod
    def _affected_anchor_ids(
        stable: HarnessState, candidate: CandidateUpdate
    ) -> set[str]:
        affected_strategy_ids = {
            str(operation.get("artifact_id"))
            for operation in candidate.operations
            if operation.get("kind") == "refine_strategy"
        }
        affected_capability_ids = {
            str(operation.get("capability_id"))
            for operation in candidate.operations
            if operation.get("kind") in {"refine_capability", "deprecate_capability"}
        }
        return {
            anchor.anchor_id
            for anchor in stable.anchors
            if set(anchor.expected_strategy_ids) & affected_strategy_ids
            or set(anchor.expected_capability_ids) & affected_capability_ids
        }

    def _evaluate_stability(
        self,
        stable: HarnessState,
        candidate: CandidateUpdate,
        episode: EpisodeRecord,
    ) -> EvaluationResult:
        structural_diagnostics = self._validate_structure(stable, candidate)
        if structural_diagnostics:
            return EvaluationResult(
                candidate_id=candidate.candidate_id,
                decision="rollback",
                current_score=float(episode.score),
                anchor_retention=1.0,
                regressions=[],
                diagnostics=structural_diagnostics,
                partial_operation_kinds=[
                    "add_raw_memory",
                    "add_strategy",
                    "refine_strategy",
                    "record_capability_outcome",
                ],
            )

        hard_failures = self._hard_anchor_failures(stable, candidate)
        review = self.llm.complete_json(
            "evaluator",
            (
                "You are the Continual Evaluator for an ALFWorld harness. Judge whether the "
                "candidate contains grounded, reusable knowledge and preserves every historical "
                "anchor semantically. A successful episode does not justify overgeneralized or "
                "unsupported rules. A failed episode may contribute raw evidence and bounded "
                "failure lessons but never a success strategy. Review every create/refine/"
                "deprecate capability operation: require grounded episode evidence, useful and "
                "non-duplicative semantics, correct dependencies, calibrated confidence and a "
                "falsifiable success condition. Internal capability claims must be supported by "
                "observable component traces and downstream behavior, not model self-description. "
                "Capabilities whose kind is atomic_external are immutable contracts supplied by "
                "the ALFWorld environment: their validated status and confidence 1.0 do not require "
                "positive episode evidence, and an unused atomic_external capability is neither a "
                "regression nor a reason to reject, demote or deprecate a candidate. Apply evidence "
                "calibration requirements only to composite_external and internal capabilities. "
                "Return JSON with decision "
                "(commit, partial_merge, or reject), retained_anchor_ids, regressions, diagnostics, "
                "rationale. An anchor is retained only if the candidate still has semantically "
                "appropriate capabilities and strategy coverage for its goal. Use only supplied "
                "anchor IDs. retained_anchor_ids, regressions, and diagnostics must be JSON arrays "
                "of plain strings, even when they contain zero or one item."
            ),
            {
                "episode": episode.to_dict(),
                "candidate": self._candidate_summary(candidate),
                "historical_anchors": [item.__dict__ for item in stable.anchors],
                "hard_anchor_failures": hard_failures,
                "thresholds": {
                    "min_current_score": self.min_current_score,
                    "max_anchor_regression": self.max_anchor_regression,
                    "provisional_min_positive_evidence": (
                        self.provisional_min_positive_evidence
                    ),
                    "validated_min_positive_evidence": self.validated_min_positive_evidence,
                    "validated_min_success_rate": self.validated_min_success_rate,
                },
            },
        )
        known_anchor_ids = {item.anchor_id for item in stable.anchors}
        retained_response = self._clean_string_list(review.get("retained_anchor_ids", []))
        retained_ids = {
            item
            for item in retained_response
            if isinstance(item, str) and item in known_anchor_ids and item not in hard_failures
        }
        if stable.anchors:
            retention = len(retained_ids) / len(stable.anchors)
        else:
            retention = 1.0
        regression_messages = self._clean_string_list(review.get("regressions", []))
        regression_messages.extend(
            f"{anchor_id}: {reason}" for anchor_id, reason in hard_failures.items()
        )
        diagnostics = self._clean_string_list(review.get("diagnostics", []))
        if review.get("rationale"):
            diagnostics.append(f"LLM rationale: {review['rationale']}")
        episode.harness_traces.append(
            {
                "evaluator": {
                    "component": "llm_evaluator",
                    "proposed_decision": str(review.get("decision", "reject")),
                    "rationale": str(review.get("rationale", "")),
                    "retained_anchor_ids": list(retained_ids),
                }
            }
        )
        for raw_episode in candidate.proposed_state.raw_memory:
            if raw_episode.get("episode_id") == episode.episode_id:
                raw_episode["harness_traces"] = list(episode.harness_traces)

        proposed_decision = str(review.get("decision", "reject"))
        if proposed_decision not in {"commit", "partial_merge", "reject"}:
            proposed_decision = "reject"
            diagnostics.append("Evaluator LLM returned an unsupported decision.")
        regression = 1.0 - retention
        if regression > self.max_anchor_regression:
            decision = "partial_merge"
            diagnostics.append("Candidate exceeded allowed historical-anchor regression.")
        elif episode.score < self.min_current_score and episode.success:
            decision = "reject"
            diagnostics.append("Current-task score did not meet the hard commit threshold.")
        elif not episode.success and proposed_decision == "commit":
            decision = "partial_merge"
            diagnostics.append("A failed episode cannot be fully committed.")
        else:
            decision = proposed_decision

        return EvaluationResult(
            candidate_id=candidate.candidate_id,
            decision=decision,
            current_score=float(episode.score),
            anchor_retention=retention,
            regressions=regression_messages,
            diagnostics=diagnostics,
            partial_operation_kinds=[
                "add_raw_memory",
                "add_strategy",
                "refine_strategy",
                "record_capability_outcome",
            ],
        )

    def _evaluate_plasticity(
        self,
        stable: HarnessState,
        candidate: CandidateUpdate,
        episode: EpisodeRecord,
    ) -> EvaluationResult:
        structural_diagnostics = self._validate_structure(stable, candidate)
        if structural_diagnostics:
            return EvaluationResult(
                candidate_id=candidate.candidate_id,
                decision="rollback",
                current_score=float(episode.score),
                anchor_retention=1.0,
                regressions=[],
                diagnostics=structural_diagnostics,
            )

        operations = candidate.operations
        evidence_kinds = {"add_raw_memory", "record_capability_outcome"}
        evidence_indices = {
            index
            for index, operation in enumerate(operations)
            if operation.get("kind") in evidence_kinds
        }
        semantic_indices = set(range(len(operations))) - evidence_indices
        affected_anchor_ids = self._affected_anchor_ids(stable, candidate)
        affected_anchors = [
            anchor.__dict__
            for anchor in stable.anchors
            if anchor.anchor_id in affected_anchor_ids
        ]
        hard_failures = self._hard_anchor_failures(stable, candidate)
        review = self.llm.complete_json(
            "evaluator",
            (
                "You are the operation-level Continual Evaluator for an ALFWorld harness in "
                "plasticity mode. The evidence operations add_raw_memory and "
                "record_capability_outcome are already accepted and must not be judged. Judge "
                "only the supplied semantic delta, never reject an operation merely because an "
                "unchanged capability or strategy elsewhere in the stable state is hypothesized. "
                "Accept a successful strategy when its ordered reusable procedure is grounded in "
                "the successful trajectory. Accept a failure_boundary when it is a bounded, "
                "grounded lesson and does not claim success. A newly created hypothesized "
                "capability may be stored with one positive grounded episode; it does not need to "
                "meet provisional or validated thresholds yet. Judge capability creation, "
                "refinement and deprecation independently from strategies. Accept a staged anchor "
                "only when its associated successful strategy is accepted. Only anchors listed "
                "under affected_historical_anchors can regress; additive operations cannot regress "
                "an old anchor. The current episode failing is not by itself evidence that an "
                "unchanged historical strategy regressed. Return JSON with decision, "
                "accepted_operation_indices, rejected_operation_indices, regressed_anchor_ids, "
                "regressions, diagnostics, and rationale. Indices must refer to candidate_delta "
                "operations. Lists must be JSON arrays."
            ),
            {
                "episode": episode.to_dict(),
                "candidate_delta": self._candidate_delta(stable, candidate),
                "preaccepted_evidence_operation_indices": sorted(evidence_indices),
                "semantic_operation_indices": sorted(semantic_indices),
                "affected_historical_anchors": affected_anchors,
                "staged_new_anchors": [
                    item.__dict__
                    for item in candidate.proposed_state.anchors
                    if item.anchor_id not in {anchor.anchor_id for anchor in stable.anchors}
                ],
                "hard_anchor_failures": hard_failures,
                "thresholds": {
                    "min_current_score": self.min_current_score,
                    "max_anchor_regression": self.max_anchor_regression,
                    "provisional_min_positive_evidence": (
                        self.provisional_min_positive_evidence
                    ),
                    "validated_min_positive_evidence": self.validated_min_positive_evidence,
                    "validated_min_success_rate": self.validated_min_success_rate,
                },
            },
        )

        proposed_decision = str(review.get("decision", "reject"))
        requested_accepts = set(
            self._clean_index_list(
                review.get("accepted_operation_indices", []), len(operations)
            )
        ) & semantic_indices
        if "accepted_operation_indices" not in review and proposed_decision == "commit":
            requested_accepts = set(semantic_indices)
        requested_rejects = set(
            self._clean_index_list(
                review.get("rejected_operation_indices", []), len(operations)
            )
        ) & semantic_indices
        accepted = evidence_indices | (requested_accepts - requested_rejects)

        # A successful flag with a sub-threshold score is still evidence, but it
        # cannot promote semantic knowledge or an anchor.
        diagnostics = self._clean_string_list(review.get("diagnostics", []))
        if episode.success and episode.score < self.min_current_score:
            accepted &= evidence_indices
            diagnostics.append(
                "Current-task score did not meet the semantic promotion threshold; "
                "evidence was retained."
            )

        known_anchor_ids = {item.anchor_id for item in stable.anchors}
        reported_regressions = {
            item
            for item in self._clean_string_list(review.get("regressed_anchor_ids", []))
            if item in affected_anchor_ids
        }
        regressed_anchor_ids = reported_regressions | set(hard_failures)
        regression_messages = self._clean_string_list(review.get("regressions", []))
        regression_messages.extend(
            f"{anchor_id}: {reason}" for anchor_id, reason in hard_failures.items()
        )
        regression = len(regressed_anchor_ids) / max(1, len(known_anchor_ids))
        if regression > self.max_anchor_regression:
            for index, operation in enumerate(operations):
                if index not in accepted:
                    continue
                kind = operation.get("kind")
                target = str(
                    operation.get("artifact_id")
                    or operation.get("capability_id")
                    or ""
                )
                if kind == "refine_strategy" and any(
                    anchor.anchor_id in regressed_anchor_ids
                    and target in anchor.expected_strategy_ids
                    for anchor in stable.anchors
                ):
                    accepted.discard(index)
                if kind in {"refine_capability", "deprecate_capability"} and any(
                    anchor.anchor_id in regressed_anchor_ids
                    and target in anchor.expected_capability_ids
                    for anchor in stable.anchors
                ):
                    accepted.discard(index)
            diagnostics.append(
                "Regressing semantic operations were rejected; independent evidence "
                "and additive operations remained eligible."
            )

        # Keep the accepted capability subgraph closed when the evaluator
        # accepts a capability that depends on another newly proposed one.
        stable_capability_ids = {
            item.capability_id
            for item in stable.capabilities
            if item.status != "deprecated"
        }
        capabilities_by_id = {
            item.capability_id: item
            for item in candidate.proposed_state.capabilities
        }
        changed = True
        while changed:
            changed = False
            accepted_created_ids = {
                str(operation.get("capability_id"))
                for index, operation in enumerate(operations)
                if index in accepted and operation.get("kind") == "create_capability"
            }
            available_capability_ids = stable_capability_ids | accepted_created_ids
            for index, operation in enumerate(operations):
                if index not in accepted or operation.get("kind") not in {
                    "create_capability",
                    "refine_capability",
                }:
                    continue
                capability = capabilities_by_id.get(
                    str(operation.get("capability_id", ""))
                )
                if capability is not None and not set(capability.dependencies) <= (
                    available_capability_ids
                ):
                    accepted.discard(index)
                    diagnostics.append(
                        f"Capability operation {index} was rejected because an "
                        "accepted dependency was unavailable."
                    )
                    changed = True

        # An anchor cannot be accepted unless its expected semantic entities
        # already exist in stable state or are produced by an accepted operation.
        stable_strategy_ids = {item.artifact_id for item in stable.abstract_memory}
        stable_capability_ids = {item.capability_id for item in stable.capabilities}
        accepted_strategy_ids = {
            str(operation.get("artifact_id"))
            for index, operation in enumerate(operations)
            if index in accepted
            and operation.get("kind") in {"add_strategy", "refine_strategy"}
        }
        accepted_capability_ids = {
            str(operation.get("capability_id"))
            for index, operation in enumerate(operations)
            if index in accepted
            and operation.get("kind")
            in {"create_capability", "refine_capability", "deprecate_capability"}
        }
        anchors_by_id = {
            item.anchor_id: item for item in candidate.proposed_state.anchors
        }
        strategy_operation_indices: Dict[str, set[int]] = {}
        for index, operation in enumerate(operations):
            if operation.get("kind") not in {"add_strategy", "refine_strategy"}:
                continue
            strategy_operation_indices.setdefault(
                str(operation.get("artifact_id", "")), set()
            ).add(index)
        for index, operation in enumerate(operations):
            if index not in accepted or operation.get("kind") != "add_anchor":
                continue
            anchor = anchors_by_id.get(str(operation.get("anchor_id", "")))
            associated_strategy_accepted = anchor is not None and all(
                not strategy_operation_indices.get(strategy_id)
                or bool(strategy_operation_indices[strategy_id] & accepted)
                for strategy_id in anchor.expected_strategy_ids
            )
            if (
                anchor is None
                or not associated_strategy_accepted
                or not set(anchor.expected_strategy_ids)
                <= (stable_strategy_ids | accepted_strategy_ids)
                or not set(anchor.expected_capability_ids)
                <= (stable_capability_ids | accepted_capability_ids)
            ):
                accepted.discard(index)
                diagnostics.append(
                    f"Anchor operation {index} was rejected because its semantic "
                    "dependencies were not accepted."
                )

        if review.get("rationale"):
            diagnostics.append(f"LLM rationale: {review['rationale']}")
        rejected = set(range(len(operations))) - accepted
        decision = (
            "commit"
            if len(accepted) == len(operations)
            else "partial_merge"
            if accepted
            else "reject"
        )
        retention = (
            1.0
            if not known_anchor_ids
            else 1.0 - len(regressed_anchor_ids) / len(known_anchor_ids)
        )
        episode.harness_traces.append(
            {
                "evaluator": {
                    "component": "llm_evaluator",
                    "gate_mode": "plasticity",
                    "proposed_decision": proposed_decision,
                    "rationale": str(review.get("rationale", "")),
                    "accepted_operation_indices": sorted(accepted),
                    "rejected_operation_indices": sorted(rejected),
                    "regressed_anchor_ids": sorted(regressed_anchor_ids),
                }
            }
        )
        for raw_episode in candidate.proposed_state.raw_memory:
            if raw_episode.get("episode_id") == episode.episode_id:
                raw_episode["harness_traces"] = list(episode.harness_traces)
        return EvaluationResult(
            candidate_id=candidate.candidate_id,
            decision=decision,
            current_score=float(episode.score),
            anchor_retention=retention,
            regressions=regression_messages,
            diagnostics=diagnostics,
            partial_operation_kinds=sorted(evidence_kinds),
            accepted_operation_indices=sorted(accepted),
            rejected_operation_indices=sorted(rejected),
        )

    def evaluate(
        self,
        stable: HarnessState,
        candidate: CandidateUpdate,
        episode: EpisodeRecord,
    ) -> EvaluationResult:
        if self.gate_mode == "plasticity":
            return self._evaluate_plasticity(stable, candidate, episode)
        return self._evaluate_stability(stable, candidate, episode)
