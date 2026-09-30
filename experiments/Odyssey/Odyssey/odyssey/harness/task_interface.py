from __future__ import annotations

from typing import Any

from .common import call_json, event_digest


TASK_INTERFACE_PROMPT = """You are the Task Interface of a continual-learning Minecraft agent.
Convert the user's open-ended instruction and current evidence into a compact structured task state.
Do not choose a skill. Do not invent unavailable observations.
If tool_returns or abstract memory show that a previous interpretation was wrong, revise the goal, constraints, and success criteria accordingly.
For long tasks, describe only the current actionable subgoal passed as user_instruction, while preserving relevant context from the original trajectory.
Return only JSON with:
{
  "goal": "short actionable goal",
  "instruction": "original user instruction",
  "task_kind": "action" | "observation",
  "context": ["important evidence from observations, tool returns, memory"],
  "constraints": ["known constraints or empty list"],
  "success_criteria": ["observable success checks"],
  "missing_information": ["unknown but relevant information"],
  "risk_notes": ["execution risks"]
}

Set task_kind to `observation` only when the task asks to inspect, diagnose,
verify, or confirm an already-existing state and does not request any world
change. Otherwise set it to `action`. This is a semantic classification, not
a keyword match.

If completion_contract and contract_progress are present, they are
observation-grounded hard constraints from the original instruction. Preserve
their target counts and describe the current actionable goal in terms of the
remaining_count. Never treat a partial inventory as completion.

When capability_summary.base_state is present, it is a persistent,
coordinate-verified registry of world assets. Treat assets whose status is
"present" or "placed" as reusable. Do not propose duplicate placement of a
registered asset; a missing registered asset may only be repaired at its
assigned position.
"""


class TaskInterface:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_prompt = TASK_INTERFACE_PROMPT
        self.system_prompt = TASK_INTERFACE_PROMPT

    def structure(
        self,
        *,
        instruction: str,
        events: list[tuple[str, dict[str, Any]]] | None,
        tool_returns: list[dict[str, Any]] | None,
        abstract_memory: list[str],
        capability_summary: dict[str, Any] | None = None,
        completion_contract: dict[str, Any] | None = None,
        contract_progress: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "user_instruction": instruction,
            "current_events": event_digest(events),
            "tool_returns": tool_returns or [],
            "retrieved_abstract_memory": abstract_memory,
            "capability_summary": capability_summary or {},
            "completion_contract": completion_contract or {},
            "contract_progress": contract_progress or {},
        }
        state = call_json(self.system_prompt, payload, self.model_name)
        state.setdefault("goal", instruction)
        state.setdefault("instruction", instruction)
        state.setdefault("task_kind", "action")
        if state["task_kind"] not in {"action", "observation"}:
            state["task_kind"] = "action"
        state.setdefault("context", [])
        state.setdefault("constraints", [])
        state.setdefault("success_criteria", [])
        state.setdefault("missing_information", [])
        state.setdefault("risk_notes", [])
        return state
