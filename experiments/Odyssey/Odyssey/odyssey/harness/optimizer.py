from __future__ import annotations

import json
from typing import Any

import odyssey.utils as U

from .common import call_json, now_iso


OPTIMIZER_PROMPT = """You are the Continual Optimizer for a Minecraft Harness.
Given the structured task, route, execution result, and raw trajectory, propose useful updates.
Abstract memory should capture reusable lessons from the trajectory and result; future routing should mainly use abstract memory.
Capability updates may add or revise a skill only when the raw evidence shows a stable reusable behavior or a missing capability.
When writing JavaScript skill code, use capability_summary.primitive_reference for valid primitive signatures and capability_summary.skill_code_examples as style examples.
Important runtime rule: capability_summary.primitive_reference is documentation only. It is not a JavaScript runtime object.
Never write JavaScript that accesses capability_summary.primitive_reference, destructures primitive functions from capability_summary, or expects capability_summary.task_state at runtime.
Generated JavaScript must follow the existing Odyssey skill-library style:
- Define exactly async function skillName(bot) { ... }.
- Call injected helpers directly, for example await craftItem(bot, "crafting_table", 1).
- Call injected helpers directly, for example await mineBlock(bot, "stone", 1).
- Call injected helpers directly, for example const position = await findSuitablePosition(bot); await placeItem(bot, "crafting_table", position).
- Use concrete Minecraft item/block names such as "birch_planks", "oak_log", or "crafting_table"; do not use generic names like "planks", "logs", or "wooden_planks".
- Primitive count arguments must be numbers, not objects.
Prefer Odyssey primitives such as placeItem, craftItem, mineBlock, exploreUntil, and findSuitablePosition over reimplementing Mineflayer logic.
When possible, include a primitive_plan, preconditions, and success_checks so the Harness can version and test the learned skill.
For placement skills, the common safe pattern is:
const position = await findSuitablePosition(bot);
await placeItem(bot, "item_name", position);
Do not invent primitive arguments or call a primitive with undefined values.
Prompt updates should be small and general, not task-specific hacks.
Return only JSON with:
{
  "abstract_memory": {
    "lessons": [{"content": "reusable lesson", "entities": ["canonical_item_id"], "skills": ["skillName"], "conditions": ["when this applies"], "confidence": 0.0}],
    "task_strategies": [],
    "failure_patterns": [],
    "skill_notes": []
  },
  "capability_updates": [
    {
      "action": "add_skill" | "update_skill",
      "name": "camelCaseSkillName",
      "description": "what this skill does",
      "code": "async function ...",
      "reason": "why this should exist",
      "preconditions": [],
      "success_checks": [],
      "primitive_plan": [],
      "test_spec": {"run_live": false}
    }
  ],
  "prompt_updates": {
    "task_interface": "",
    "adaptive_router": "",
    "continual_evaluator": "",
    "continual_optimizer": ""
  },
  "notes": "brief summary"
}

Each abstract-memory entry should be an object following the example schema.
Use confidence from 0 to 1. Include only entities, skills and conditions that
are supported by this trajectory. Do not generate ids, timestamps, evidence
counts, retrieval metadata, task_kind, outcome or status; the Harness attaches
those fields from the approved raw execution provenance. Legacy string entries
remain accepted but structured objects are preferred.
"""


class ContinualOptimizer:
    def __init__(self, model_name: str, ckpt_dir: str = "ckpt"):
        self.model_name = model_name
        self.base_prompt = OPTIMIZER_PROMPT
        self.system_prompt = OPTIMIZER_PROMPT
        self.root = U.f_mkdir(ckpt_dir, "harness")
        self.prompt_overrides_path = U.f_join(self.root, "prompt_overrides.json")
        if U.f_exists(self.prompt_overrides_path):
            self.prompt_overrides = U.load_json(self.prompt_overrides_path)
        else:
            self.prompt_overrides = {}

    def propose_update(
        self,
        *,
        task_state: dict[str, Any],
        route: dict[str, Any],
        raw_record: dict[str, Any],
        evaluation: dict[str, Any],
        abstract_memory: dict[str, Any],
        capability_summary: dict[str, Any],
    ) -> dict[str, Any]:
        update = call_json(
            self.system_prompt,
            {
                "task_state": task_state,
                "route": route,
                "raw_record": raw_record,
                "evaluation": evaluation,
                "current_abstract_memory": abstract_memory,
                "capability_summary": capability_summary,
            },
            self.model_name,
        )
        if not isinstance(update, dict):
            # LLM JSON can be syntactically valid while violating the schema.
            # Treat it as an empty proposal rather than breaking the task that
            # has already finished executing.
            update = {}
        update.setdefault("abstract_memory", {})
        update.setdefault("capability_updates", [])
        update.setdefault("prompt_updates", {})
        update.setdefault("notes", "")
        if not isinstance(update["abstract_memory"], dict):
            update["abstract_memory"] = {}
        if not isinstance(update["capability_updates"], list):
            update["capability_updates"] = []
        if not isinstance(update["prompt_updates"], dict):
            update["prompt_updates"] = {}
        if not isinstance(update["notes"], str):
            update["notes"] = ""
        return update

    def apply_prompt_updates(self, updates: dict[str, Any] | Any) -> dict[str, Any]:
        if not isinstance(updates, dict):
            return {}
        applied = {}
        for key, value in updates.items():
            if not value or not isinstance(value, str):
                continue
            self.prompt_overrides[key] = {
                "updated_at": now_iso(),
                "instruction": value,
            }
            applied[key] = value
        if applied:
            with open(self.prompt_overrides_path, "w", encoding="utf-8") as fp:
                json.dump(self.prompt_overrides, fp, ensure_ascii=False, indent=2)
        return applied
