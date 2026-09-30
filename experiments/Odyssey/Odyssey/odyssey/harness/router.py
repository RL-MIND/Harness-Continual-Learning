from __future__ import annotations

from typing import Any

from .common import call_json


ROUTER_PROMPT = """You are the Adaptive Router of a Minecraft continual-learning Harness.
Choose the best next executable capability using the structured task state, abstract memory, and capability map.
The final choice must be made by reasoning over the provided skills/tools, not by a fixed rule.
Prefer existing skills. A single listed primitive with concrete arguments may be
executed directly; use a new skill only when a reusable multi-step composition
is needed.
Do not select a skill when its stated effects conflict with an exact quantity
or an existing-world-object constraint in the task; prefer an eligible
primitive in that case.
Use capability_map.capability_layers to distinguish primitives, base skills, learned skills, and diagnostic skills.
If the task state is uncertain because inventory or nearby blocks are unknown, prefer an available diagnostic skill before requesting a new skill.
Use capability_map.skill_health to avoid skills with repeated recent failures when a healthier equivalent exists.
capability_map.blocked_skill_names contains skills that already failed without
making contract progress in this task. Never select them again in this task.
When the task explicitly asks to place or establish a durable world object for
later reuse, set persistent_asset.action to "place_or_reuse" and give its
concrete Minecraft block name. This is a model decision based on the task and
state, not a keyword rule in the executor. Otherwise use action "none".
For a selected capability, world_asset_dependencies lists only the abstract
dependencies needed for this particular invocation (for example a 3x3 craft
may need crafting_surface while a 2x2 craft does not). It must be a subset of
that capability's declared dependency roles.
Return only JSON with:
{
  "decision": "use_skill" | "use_primitive" | "request_new_skill" | "ask_user",
  "skill_name": "exact existing skill name when decision is use_skill, otherwise empty",
  "primitive_name": "exact primitive name when decision is use_primitive, otherwise empty",
  "primitive_args": ["arguments after bot, in signature order, only for use_primitive"],
  "reasoning": "brief reason",
  "expected_effect": "what should change in the world",
  "fallback": "what to try if this fails",
  "persistent_asset": {
    "action": "none" | "place_or_reuse",
    "block_name": "concrete Minecraft block name, or empty"
  },
  "world_asset_dependencies": ["abstract dependency roles needed now"],
  "new_skill_request": {
    "name": "camelCase name",
    "description": "needed behavior",
    "requirements": ["primitive/tool requirements"]
  }
}

completion_contract and contract_progress, when supplied, are hard,
observation-grounded completion constraints. You may choose only how to make
the remaining targets progress; you may not declare them complete. Do not use
ask_user while any contract target has remaining_count > 0 unless an explicit
user-only choice is genuinely required.
"""


class AdaptiveRouter:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_prompt = ROUTER_PROMPT
        self.system_prompt = ROUTER_PROMPT

    def route(
        self,
        *,
        task_state: dict[str, Any],
        abstract_memory: list[str],
        capability_summary: dict[str, Any],
        last_feedback: dict[str, Any] | None = None,
        completion_contract: dict[str, Any] | None = None,
        contract_progress: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        route = call_json(
            self.system_prompt,
            {
                "task_state": task_state,
                "retrieved_abstract_memory": abstract_memory,
                "capability_map": capability_summary,
                "last_feedback": last_feedback or {},
                "completion_contract": completion_contract or {},
                "contract_progress": contract_progress or {},
            },
            self.model_name,
        )
        route.setdefault("decision", "ask_user")
        route.setdefault("skill_name", "")
        route.setdefault("primitive_name", "")
        route.setdefault("primitive_args", [])
        route.setdefault("reasoning", "")
        route.setdefault("expected_effect", "")
        route.setdefault("fallback", "")
        route.setdefault("new_skill_request", {})
        route.setdefault("world_asset_dependencies", [])
        if not isinstance(route["world_asset_dependencies"], list):
            route["world_asset_dependencies"] = []
        route["world_asset_dependencies"] = [
            value for value in route["world_asset_dependencies"] if isinstance(value, str)
        ]
        route.setdefault("persistent_asset", {"action": "none", "block_name": ""})
        if not isinstance(route["persistent_asset"], dict):
            route["persistent_asset"] = {"action": "none", "block_name": ""}
        asset = route["persistent_asset"]
        if asset.get("action") not in {"none", "place_or_reuse"}:
            asset["action"] = "none"
        if not isinstance(asset.get("block_name"), str):
            asset["block_name"] = ""
        primitive_names = {
            item.get("name")
            for item in capability_summary.get("primitive_reference", [])
            if item.get("name")
        }
        blocked_skill_names = set(capability_summary.get("blocked_skill_names", []))
        if route["decision"] == "use_skill":
            if route.get("skill_name") in blocked_skill_names:
                route["decision"] = "blocked_skill"
                route["reasoning"] = "Selected skill is blocked for this task after a no-progress failure."
                return route
            candidates = set(capability_summary.get("all_skill_names", []))
            if route.get("skill_name") not in candidates:
                requested_name = route.get("skill_name") or "missingSkill"
                # Backwards compatibility for a model that still puts a
                # primitive in skill_name. It must provide arguments through
                # the explicit primitive route, so synthesize a composition
                # when those arguments are absent.
                if requested_name in primitive_names:
                    if isinstance(route.get("primitive_args"), list) and route["primitive_args"]:
                        route["decision"] = "use_primitive"
                        route["primitive_name"] = requested_name
                        route["skill_name"] = ""
                        return route
                    requested_name = f"compose{requested_name[:1].upper()}{requested_name[1:]}"
                route["decision"] = "request_new_skill"
                route["new_skill_request"] = {
                    "name": requested_name,
                    "description": "Router selected a skill that does not exist; create an equivalent capability.",
                    "requirements": [],
                }
                route["skill_name"] = ""
        if route["decision"] == "use_primitive":
            if route.get("primitive_name") not in primitive_names or not isinstance(route.get("primitive_args"), list):
                route["decision"] = "request_new_skill"
                route["new_skill_request"] = {
                    "name": "composeMissingPrimitive",
                    "description": "Primitive route was invalid; synthesize a validated composition instead.",
                    "requirements": [],
                }
                route["primitive_name"] = ""
                route["primitive_args"] = []
        return route
