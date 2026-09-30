from __future__ import annotations

import json
from typing import Any

from .common import call_json, slugify_name


SYNTHESIZER_PROMPT = """You are the Skill Synthesizer for a Minecraft Harness.
When a missing capability can be expressed as a short composition of known Odyssey primitives, output a primitive_plan.
Use only primitives and helpers listed in capability_summary.primitive_reference.
Important runtime rule: capability_summary.primitive_reference is documentation only. It is not a JavaScript runtime object.
Do not write code that reads from capability_summary, primitive_reference, or destructures primitive functions from them.
Generated JavaScript must follow the existing Odyssey skill-library style:
- Define exactly async function skillName(bot) { ... }.
- Call injected helpers directly, for example await craftItem(bot, "crafting_table", 1).
- Call injected helpers directly, for example await mineBlock(bot, "stone", 1).
- Call injected helpers directly, for example const position = await findSuitablePosition(bot); await placeItem(bot, "crafting_table", position).
- Use concrete Minecraft item/block names such as "birch_planks", "oak_log", or "crafting_table"; do not use generic names like "planks", "logs", or "wooden_planks".
- Primitive count arguments must be numbers, not objects.
- To search for a block before mining it, use a structured step such as
  {"op": "exploreUntilBlock", "block": "stone", "direction": [1, 0, 1], "max_time": 60, "save_as": "stone"}.
  Do not invent an exploreUntil callback or omit its direction argument.
Prefer primitive plans over handwritten JavaScript.
Return only JSON with:
{
  "can_synthesize": true | false,
  "reason": "brief reason",
  "skill": {
    "name": "camelCaseSkillName",
    "description": "what this skill does",
    "preconditions": [],
    "success_checks": [],
    "primitive_plan": [
      {"op": "findSuitablePosition", "save_as": "position"},
      {"op": "placeItem", "args": ["crafting_table", "$position"]}
    ]
  }
}
"""


class SkillSynthesizer:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def propose(
        self,
        *,
        task_state: dict[str, Any],
        route: dict[str, Any],
        capability_summary: dict[str, Any],
    ) -> dict[str, Any]:
        proposal = call_json(
            SYNTHESIZER_PROMPT,
            {
                "task_state": task_state,
                "route": route,
                "capability_summary": capability_summary,
            },
            self.model_name,
        )
        proposal.setdefault("can_synthesize", False)
        proposal.setdefault("reason", "")
        proposal.setdefault("skill", {})
        return proposal

    def render_update(self, proposal: dict[str, Any]) -> dict[str, Any] | None:
        if not proposal.get("can_synthesize"):
            return None
        skill = proposal.get("skill") or {}
        raw_name = skill.get("name")
        plan = skill.get("primitive_plan") or []
        if not raw_name or not isinstance(plan, list):
            return None
        name = slugify_name(raw_name)
        body = self._render_plan(plan)
        if not body:
            return None
        code = f"async function {name}(bot) {{\n{body}\n}}"
        return {
            "action": "add_skill",
            "name": name,
            "description": skill.get("description") or proposal.get("reason") or name,
            "code": code,
            "reason": proposal.get("reason", ""),
            "generation_mode": "primitive_plan_template",
            "primitive_plan": plan,
            "preconditions": skill.get("preconditions", []),
            "success_checks": skill.get("success_checks", []),
        }

    def render_ephemeral_primitive(
        self,
        primitive_name: str,
        primitive_args: list[Any],
    ) -> dict[str, Any] | None:
        """Compile one allow-listed primitive into non-persistent skill code."""
        body = self._render_plan([{"op": primitive_name, "args": primitive_args}])
        if not body:
            return None
        name = f"temporary{primitive_name[:1].upper()}{primitive_name[1:]}"
        return {
            "name": name,
            "code": f"async function {name}(bot) {{\n{body}\n}}",
            "generation_mode": "ephemeral_primitive",
            "primitive_plan": [{"op": primitive_name, "args": primitive_args}],
        }

    def _render_plan(self, plan: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for index, step in enumerate(plan):
            op = step.get("op")
            save_as = step.get("save_as")
            if op == "findSuitablePosition":
                target = self._js_identifier(save_as or f"position{index}")
                if not target:
                    return ""
                lines.append(f"  const {target} = await findSuitablePosition(bot);")
            elif op == "placeItem":
                args = step.get("args") or []
                if len(args) != 2:
                    return ""
                item, position = args
                lines.append(f"  await placeItem(bot, {self._literal(item)}, {self._value(position)});")
            elif op == "craftItem":
                args = step.get("args") or []
                if not 1 <= len(args) <= 2:
                    return ""
                count = args[1] if len(args) == 2 else 1
                lines.append(f"  await craftItem(bot, {self._literal(args[0])}, {self._number(count)});")
            elif op == "mineBlock":
                args = step.get("args") or []
                if not 1 <= len(args) <= 2:
                    return ""
                count = args[1] if len(args) == 2 else 1
                lines.append(f"  await mineBlock(bot, {self._literal(args[0])}, {self._number(count)});")
            elif op == "equipItem":
                args = step.get("args") or []
                if len(args) != 1 or not self._minecraft_name(args[0]):
                    return ""
                lines.append(f"  await equipItem(bot, {self._literal(args[0])});")
            elif op == "smeltItem":
                args = step.get("args") or []
                if (
                    len(args) != 3
                    or not self._minecraft_name(args[0])
                    or not self._minecraft_name(args[1])
                    or not self._is_number(args[2])
                ):
                    return ""
                lines.append(
                    f"  await smeltItem(bot, {self._literal(args[0])}, {self._literal(args[1])}, {self._number(args[2])});"
                )
            elif op == "exploreUntilBlock":
                block_name = step.get("block")
                direction = step.get("direction", [1, 0, 1])
                max_time = step.get("max_time", 60)
                target = self._js_identifier(save_as or f"foundBlock{index}")
                if (
                    not isinstance(block_name, str)
                    or not self._js_identifier(block_name)
                    or not isinstance(direction, list)
                    or len(direction) != 3
                    or not all(value in {-1, 0, 1} for value in direction)
                    or not isinstance(max_time, (int, float))
                    or not target
                ):
                    return ""
                block_literal = json.dumps(block_name, ensure_ascii=False)
                lines.extend(
                    [
                        f"  const {target} = await exploreUntil(bot, new Vec3({direction[0]}, {direction[1]}, {direction[2]}), {int(max_time)}, () => {{",
                        f"    return bot.findBlock({{ matching: mcData.blocksByName[{block_literal}].id, maxDistance: 32 }});",
                        "  });",
                    ]
                )
            elif op == "callSkill":
                name = self._js_identifier(step.get("name", ""))
                if not name:
                    return ""
                lines.append(f"  await {name}(bot);")
            elif op == "chat":
                lines.append(f"  bot.chat({self._literal(step.get('message', 'Done.'))});")
            else:
                return ""
        return "\n".join(lines)

    def _literal(self, value: Any) -> str:
        if isinstance(value, str) and value.startswith("$"):
            return self._value(value)
        return json.dumps(value, ensure_ascii=False)

    def _value(self, value: Any) -> str:
        if isinstance(value, str) and value.startswith("$"):
            return self._js_identifier(value[1:])
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def _number(self, value: Any) -> str:
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str) and value.isdigit():
            return value
        return "1"

    def _is_number(self, value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0

    def _minecraft_name(self, value: Any) -> bool:
        return isinstance(value, str) and bool(value) and all(char.islower() or char.isdigit() or char == "_" for char in value)

    def _js_identifier(self, value: str) -> str:
        cleaned = "".join(ch for ch in value if ch.isalnum() or ch == "_")
        if not cleaned or cleaned[0].isdigit():
            return ""
        return cleaned
