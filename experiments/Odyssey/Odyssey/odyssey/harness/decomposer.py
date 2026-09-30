from __future__ import annotations

from typing import Any

from .common import call_json


DECOMPOSER_PROMPT = """You are the Subgoal Decomposer for a Minecraft continual-learning Harness.
Split long-horizon instructions into short observable Minecraft subgoals only when useful.
Each subgoal should be executable by one existing or easily synthesized skill.
Do not decompose simple one-step placement, mining, dropping, or diagnostic tasks.
If failure_feedback is present, revise the remaining plan instead of repeating a known-bad plan.
Return only JSON with:
{
  "should_decompose": true | false,
  "reason": "brief reason",
  "subgoals": ["short imperative subgoal"]
}

When completion_contract and contract_progress are present, preserve their
hard target counts. Plan only the remaining work and make each subgoal state
the remaining quantity; do not replace a delta goal with an absolute inventory
claim.
"""


class TaskDecomposer:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def decompose(
        self,
        *,
        instruction: str,
        events: list[tuple[str, dict[str, Any]]] | None,
        abstract_memory: list[str],
        capability_summary: dict[str, Any],
        failure_feedback: dict[str, Any] | None = None,
        completed_subgoals: list[str] | None = None,
        completion_contract: dict[str, Any] | None = None,
        contract_progress: dict[str, Any] | None = None,
    ) -> list[str]:
        result = call_json(
            DECOMPOSER_PROMPT,
            {
                "instruction": instruction,
                "current_events": events[-3:] if events else [],
                "retrieved_abstract_memory": abstract_memory,
                "capability_summary": capability_summary,
                "failure_feedback": failure_feedback or {},
                "completed_subgoals": completed_subgoals or [],
                "completion_contract": completion_contract or {},
                "contract_progress": contract_progress or {},
            },
            self.model_name,
        )
        subgoals = result.get("subgoals") or []
        if not result.get("should_decompose") or not isinstance(subgoals, list):
            return [instruction]
        clean = [str(item).strip() for item in subgoals if str(item).strip()]
        return clean or [instruction]
