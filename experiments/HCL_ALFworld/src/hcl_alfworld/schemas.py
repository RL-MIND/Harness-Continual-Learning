from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskEvidence:
    """Paper Eq. (2): evidence, recognizable goal, and constraints."""

    interaction_id: str
    task_type: str
    evidence: str
    goal: str
    constraints: List[str]
    admissible_commands: List[str]
    game_file: str = ""
    initial_observation: str = ""
    step: int = 0
    episode_history: List[Dict[str, Any]] = field(default_factory=list)
    progress: Dict[str, Any] = field(default_factory=dict)
    interface_trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepRecord:
    observation: str
    action: str
    next_observation: str
    admissible_commands: List[str]
    reward: float = 0.0


@dataclass
class EpisodeRecord:
    episode_id: str
    task_type: str
    goal: str
    game_file: str
    split: str
    steps: List[StepRecord]
    success: bool
    score: float
    failure_reason: str = ""
    harness_traces: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyArtifact:
    artifact_id: str
    task_type: str
    title: str
    procedure: List[str]
    applicability: List[str]
    failure_boundaries: List[str]
    evidence_episode_ids: List[str]
    validation_count: int = 0
    success_count: int = 0
    created_at_version: int = 0


@dataclass
class CapabilityUnit:
    capability_id: str
    name: str
    kind: str
    function: str
    action_prefixes: List[str]
    input_output: str
    preconditions: List[str]
    failure_modes: List[str]
    availability: str = "available"
    version: str = "1"
    validation_evidence: List[str] = field(default_factory=list)
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    procedure: List[str] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    success_conditions: List[str] = field(default_factory=list)
    positive_evidence_episode_ids: List[str] = field(default_factory=list)
    negative_evidence_episode_ids: List[str] = field(default_factory=list)
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    confidence: float = 0.0
    status: str = "hypothesized"
    provider_component: str = ""
    model: str = ""
    prompt_version: str = ""
    created_at_version: int = 0
    updated_at_version: int = 0


@dataclass
class AnchorCase:
    anchor_id: str
    task_type: str
    goal: str
    evidence: str
    admissible_commands: List[str]
    expected_capability_ids: List[str]
    expected_strategy_ids: List[str]
    source_episode_id: str


@dataclass
class HarnessState:
    """Paper Eq. (1), serialized stable harness state H_t."""

    version: int = 0
    raw_memory: List[Dict[str, Any]] = field(default_factory=list)
    abstract_memory: List[StrategyArtifact] = field(default_factory=list)
    capabilities: List[CapabilityUnit] = field(default_factory=list)
    anchors: List[AnchorCase] = field(default_factory=list)
    routing_policy: Dict[str, Any] = field(default_factory=dict)
    evaluator_rules: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HarnessState":
        capabilities = []
        for raw in data.get("capabilities", []):
            item = dict(raw)
            if item.get("kind") == "external":
                item["kind"] = "atomic_external"
                item.setdefault("status", "validated")
                item.setdefault("confidence", 1.0)
            capabilities.append(CapabilityUnit(**item))
        return cls(
            version=int(data.get("version", 0)),
            raw_memory=list(data.get("raw_memory", [])),
            abstract_memory=[StrategyArtifact(**x) for x in data.get("abstract_memory", [])],
            capabilities=capabilities,
            anchors=[AnchorCase(**x) for x in data.get("anchors", [])],
            routing_policy=dict(data.get("routing_policy", {})),
            evaluator_rules=dict(data.get("evaluator_rules", {})),
        )


@dataclass
class RoutingContext:
    task: TaskEvidence
    memories: List[Dict[str, Any]]
    strategies: List[StrategyArtifact]
    capabilities: List[CapabilityUnit]
    trace: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateUpdate:
    candidate_id: str
    base_version: int
    proposed_state: HarnessState
    operations: List[Dict[str, Any]]
    rationale: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResult:
    candidate_id: str
    decision: str
    current_score: float
    anchor_retention: float
    regressions: List[str]
    diagnostics: List[str]
    partial_operation_kinds: List[str] = field(default_factory=list)
    accepted_operation_indices: List[int] = field(default_factory=list)
    rejected_operation_indices: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeResult:
    record: EpisodeRecord
    routing_traces: List[Dict[str, Any]]
    candidate: Optional[CandidateUpdate] = None
    evaluation: Optional[EvaluationResult] = None
