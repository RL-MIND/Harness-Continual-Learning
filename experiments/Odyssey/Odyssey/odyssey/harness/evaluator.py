from __future__ import annotations

from typing import Any

from .common import INVENTORY_SCHEMA, call_json, event_digest


TASK_EVALUATOR_PROMPT = """You are the Continual Evaluator for a Minecraft Harness.
Judge whether the latest execution achieved the structured task state's success criteria.
Use only observable evidence from events, inventory, registered world assets in task_state,
chat logs, and tool errors. When route.decision is `observation_check`, no new execution is
expected: judge whether the requested read-only inspection is already established by the
current observations. Do not require a state change for such a task, and never invent
unobserved world state.
Return only JSON with:
{
  "success": true | false,
  "critique": "brief evidence-based critique",
  "evidence": ["specific observations supporting the judgment"],
  "regression_risks": ["risks noticed during this execution"]
}
"""


UPDATE_EVALUATOR_PROMPT = """You are the safety evaluator for Harness continual updates.
Decide whether the optimizer's candidate update is safe to apply.
Reject updates that are unsupported by evidence, likely to break existing skills/tasks, overwrite broad prompts with narrow hacks, or add unsafe/broken JavaScript.
For JavaScript skill updates, verify primitive usage against capability_summary.primitive_reference and nearby code examples.
Reject code that passes undefined values into placeItem, craftItem, mineBlock, or other primitives, or that places blocks without a valid Vec3 position.
Return only JSON with:
{
  "approved": true | false,
  "reason": "brief reason",
  "safe_parts": {
    "abstract_memory": {},
    "capability_updates": [],
    "prompt_updates": {}
  },
  "rejected_parts": ["short descriptions"]
}
"""


class ContinualEvaluator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_task_prompt = TASK_EVALUATOR_PROMPT
        self.base_update_prompt = UPDATE_EVALUATOR_PROMPT
        self.task_prompt = TASK_EVALUATOR_PROMPT
        self.update_prompt = UPDATE_EVALUATOR_PROMPT

    def judge_task(
        self,
        *,
        task_state: dict[str, Any],
        route: dict[str, Any],
        events: list[tuple[str, dict[str, Any]]] | None,
        inventory: dict[str, Any] | None,
    ) -> dict[str, Any]:
        result = call_json(
            self.task_prompt,
            {
                "task_state": task_state,
                "route": route,
                "events": event_digest(events),
                "final_inventory": inventory or {},
                "inventory_schema": INVENTORY_SCHEMA,
            },
            self.model_name,
        )
        result.setdefault("success", False)
        result.setdefault("critique", "")
        result.setdefault("evidence", [])
        result.setdefault("regression_risks", [])
        result["success"] = bool(result["success"])
        return result

    def judge_update(
        self,
        *,
        task_state: dict[str, Any],
        raw_record: dict[str, Any],
        candidate_update: dict[str, Any],
        abstract_memory: dict[str, Any],
        capability_summary: dict[str, Any],
    ) -> dict[str, Any]:
        result = call_json(
            self.update_prompt,
            {
                "task_state": task_state,
                "raw_record": raw_record,
                "candidate_update": candidate_update,
                "current_abstract_memory": abstract_memory,
                "capability_summary": capability_summary,
            },
            self.model_name,
        )
        result.setdefault("approved", False)
        result.setdefault("reason", "")
        result.setdefault("safe_parts", {})
        result.setdefault("rejected_parts", [])
        result["approved"] = bool(result["approved"])
        return result
