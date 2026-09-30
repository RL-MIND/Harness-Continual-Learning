from __future__ import annotations

import json
from typing import Any

import odyssey.utils as U

from .common import now_iso, slugify_name
from .diagnostics import DIAGNOSTIC_SKILLS


PRIMITIVE_REFERENCE = [
    {
        "name": "placeItem",
        "type": "primitive",
        "signature": "await placeItem(bot, name, position)",
        "args": [
            {"name": "bot", "type": "Bot"},
            {"name": "name", "type": "minecraft_item_name"},
            {"name": "position", "type": "Vec3"},
        ],
        "preconditions": ["item_in_inventory", "position_has_reference_block"],
        "effects": ["block_placed", "inventory_item_decremented"],
        "description": "Place an inventory item at a target Vec3 position.",
        "constraints": [
            "name must be a Minecraft item name string, for example \"crafting_table\".",
            "position must be a Vec3.",
            "Use findSuitablePosition(bot) for simple nearby placement.",
            "The item must already be in inventory; otherwise placeItem only chats a failure.",
        ],
        "example": (
            "const position = await findSuitablePosition(bot);\n"
            "await placeItem(bot, \"crafting_table\", position);"
        ),
    },
    {
        "name": "craftItem",
        "type": "primitive",
        "signature": "await craftItem(bot, name, count = 1)",
        "args": [
            {"name": "bot", "type": "Bot"},
            {"name": "name", "type": "minecraft_item_name"},
            {"name": "count", "type": "integer", "default": 1, "meaning": "number_of_recipe_executions"},
        ],
        "preconditions": ["recipe_exists", "ingredients_in_inventory", "crafting_table_nearby_for_3x3_recipes"],
        "effects": ["crafted_item_added", "ingredients_consumed"],
        "world_asset_requirements": [
            {"role": "crafting_surface", "accepted_blocks": ["crafting_table"]}
        ],
        "description": (
            "Craft an item using available inventory. count is the number of recipe executions, "
            "not the desired output-item quantity; e.g. 6 oak logs become 24 planks with "
            "craftItem(bot, 'oak_planks', 6)."
        ),
        "constraints": [
            "name must be a Minecraft item name string.",
            "count must be a number.",
            "count must equal recipe executions: divide desired output by that recipe's output per execution.",
            "For 3x3 recipes, place or find a crafting table before calling craftItem.",
        ],
        "example": "await craftItem(bot, \"stone_pickaxe\", 1); // one recipe execution",
    },
    {
        "name": "mineBlock",
        "type": "primitive",
        "signature": "await mineBlock(bot, name, count = 1)",
        "args": [
            {"name": "bot", "type": "Bot"},
            {"name": "name", "type": "minecraft_block_name"},
            {"name": "count", "type": "number", "default": 1},
        ],
        "preconditions": ["matching_block_nearby", "block_reachable", "tool_sufficient_when_required"],
        "effects": ["block_collected", "inventory_item_added"],
        "description": "Mine and collect nearby blocks by block name.",
        "constraints": [
            "name must be a Minecraft block name string.",
            "count must be a number.",
            "If no matching block is nearby, explore first with exploreUntil.",
        ],
        "example": "await mineBlock(bot, \"stone\", 1);",
    },
    {
        "name": "equipItem",
        "type": "primitive",
        "signature": "await equipItem(bot, itemName)",
        "args": [
            {"name": "bot", "type": "Bot"},
            {"name": "itemName", "type": "minecraft_item_name"},
        ],
        "preconditions": ["item_in_inventory"],
        "effects": ["main_hand_equipped"],
        "description": (
            "Equip an existing inventory item in the main hand. This primitive never "
            "crafts, places, mines, drops, or otherwise changes the world."
        ),
        "constraints": [
            "itemName must be an exact Minecraft item name string.",
            "The item must already be in inventory; a missing item returns a structured failed precondition.",
            "Use the observed main_hand slot or the primitive chat message as evidence of success.",
        ],
        "example": "await equipItem(bot, \"stone_axe\");",
    },
    {
        "name": "exploreUntil",
        "type": "primitive",
        "signature": "await exploreUntil(bot, direction, maxTime, callback)",
        "args": [
            {"name": "bot", "type": "Bot"},
            {"name": "direction", "type": "Vec3"},
            {"name": "maxTime", "type": "number"},
            {"name": "callback", "type": "function"},
        ],
        "preconditions": ["bot_can_move"],
        "effects": ["target_discovered_or_timeout"],
        "description": "Move while repeatedly checking a callback until a target is found.",
        "constraints": [
            "direction should be a Vec3.",
            "callback should return a block/entity/object when found.",
            "Common search radius inside callbacks is 32 blocks.",
        ],
        "example": (
            "const block = await exploreUntil(bot, new Vec3(1, 0, 1), 60, () => {\n"
            "  return bot.findBlock({ matching: mcData.blocksByName.oak_log.id, maxDistance: 32 });\n"
            "});"
        ),
    },
    {
        "name": "smeltItem",
        "type": "primitive",
        "signature": "await smeltItem(bot, itemName, fuelName, count)",
        "args": [
            {"name": "bot", "type": "Bot"},
            {"name": "itemName", "type": "minecraft_item_name"},
            {"name": "fuelName", "type": "minecraft_item_name"},
            {"name": "count", "type": "number"},
        ],
        "preconditions": ["furnace_available", "input_item_in_inventory", "fuel_in_inventory"],
        "effects": ["smelted_item_added", "fuel_consumed"],
        "world_asset_requirements": [
            {"role": "heat_source", "accepted_blocks": ["furnace"]}
        ],
        "contract_fallback": {
            "kind": "raw_to_ingot",
            "input_prefix": "raw_",
            "output_suffix": "_ingot",
            "fuel": "coal",
        },
        "description": "Smelt input items in a furnace with a fuel item.",
        "constraints": [
            "Requires a furnace placed nearby or available through existing furnace workflows.",
            "Item and fuel names must be Minecraft item name strings.",
        ],
        "example": "await smeltItem(bot, \"raw_iron\", \"coal\", 1);",
    },
    {
        "name": "findSuitablePosition",
        "type": "helper",
        "signature": "const position = await findSuitablePosition(bot)",
        "args": [{"name": "bot", "type": "Bot"}],
        "preconditions": ["nearby_space_available"],
        "effects": ["returns_placeable_Vec3"],
        "description": "Find a nearby Vec3 suitable for placing a block.",
        "constraints": [
            "Use before placeItem for ordinary block placement.",
            "The returned position is intended to have a valid neighboring reference block.",
        ],
        "example": "const position = await findSuitablePosition(bot);",
    },
]


EXAMPLE_SKILL_PRIORITY = [
    "placeChest",
    "placeRail",
    "craftWoodenPickaxe",
    "craftStonePickaxe",
    "craftCraftingTable",
    "mineWoodLog",
    "mineCobblestone",
    "collectCobblestone",
]


class CapabilityMap:
    """Skill and tool inventory exposed to the Harness router."""

    def __init__(
        self,
        skill_manager,
        ckpt_dir: str = "ckpt",
        include_harness_skills: bool = True,
    ):
        self.skill_manager = skill_manager
        self.root = U.f_mkdir(ckpt_dir, "harness", "skills")
        self.compositional_dir = U.f_mkdir(self.root, "compositional")
        self.description_dir = U.f_mkdir(self.root, "description")
        self.skills_path = U.f_join(self.root, "skills.json")
        self.versions_path = U.f_join(self.root, "versions.jsonl")
        self.harness_skills = self._load_harness_skills() if include_harness_skills else {}
        self._register_diagnostic_skills()
        if include_harness_skills:
            self._merge_harness_skills()

    def summarize(self, query: str, top_k: int = 24) -> dict[str, Any]:
        skills = self.skill_manager.skills
        names = sorted(skills.keys())
        retrieved_names: list[str] = []
        retrieved = []
        try:
            k_backup = self.skill_manager.retrieval_top_k
            self.skill_manager.retrieval_top_k = top_k
            codes, descriptions = self.skill_manager.retrieve_skills(query)
            self.skill_manager.retrieval_top_k = k_backup
            for code, description in zip(codes, descriptions):
                name = self._name_for_code(code)
                if name:
                    retrieved_names.append(name)
                retrieved.append({"name": name, "description": description})
        except Exception:  # noqa: BLE001
            retrieved = []
        if not retrieved:
            retrieved = [
                {"name": name, "description": skills[name].get("description", "")}
                for name in names[:top_k]
            ]
            retrieved_names = [item["name"] for item in retrieved if item["name"]]
        for name, entry in self._rank_harness_skills(query):
            if name not in retrieved_names:
                retrieved.insert(0, {"name": name, "description": entry.get("description", "")})
                retrieved_names.insert(0, name)
        retrieved = retrieved[:top_k]
        retrieved_names = retrieved_names[:top_k]
        return {
            "skill_count": len(names),
            "all_skill_names": names,
            "capability_layers": self._capability_layers(names),
            "candidate_skills": retrieved,
            "candidate_skill_names": retrieved_names,
            "primitive_reference": PRIMITIVE_REFERENCE,
            "skill_code_examples": self._select_skill_code_examples(query, retrieved_names),
            "external_tools": [
                "Mineflayer bot API",
                "Odyssey control primitives",
                "Minecraft inventory, crafting, mining, placing, and combat primitives",
            ],
        }

    def learned_skill_names(self) -> list[str]:
        return sorted(self.harness_skills.keys())

    def world_asset_requirements(self, route: dict[str, Any]) -> list[dict[str, Any]]:
        """Infer dependencies from the selected executable capability only.

        This intentionally does not inspect the natural-language task. Newly
        synthesized skills carry a primitive plan, so their dependencies are
        derived from the same primitive metadata as direct executions.
        """
        operations: list[str] = []
        if route.get("decision") == "use_primitive":
            operations.append(str(route.get("primitive_name", "")))
        elif route.get("decision") == "use_skill":
            skill = self.skill_manager.skills.get(route.get("skill_name", ""), {})
            plan = (skill.get("metadata", {}) or {}).get("primitive_plan", [])
            if isinstance(plan, list):
                operations.extend(str(step.get("op", "")) for step in plan if isinstance(step, dict))
        requested_roles = set(route.get("world_asset_dependencies", []))
        requirements: list[dict[str, Any]] = []
        for operation in operations:
            primitive = next((item for item in PRIMITIVE_REFERENCE if item["name"] == operation), None)
            for requirement in (primitive or {}).get("world_asset_requirements", []):
                if requirement.get("role") in requested_roles and requirement not in requirements:
                    requirements.append(dict(requirement))
        return requirements

    def diagnostic_skill_names(self) -> list[str]:
        return sorted(DIAGNOSTIC_SKILLS.keys())

    def get_skill_code(self, name: str) -> str:
        if name not in self.skill_manager.skills:
            raise KeyError(f"Unknown skill: {name}")
        return self.skill_manager.skills[name]["code"]

    def apply_skill_update(self, update: dict[str, Any]) -> dict[str, Any]:
        action = update.get("action")
        if action not in {"add_skill", "update_skill"}:
            return {"applied": False, "reason": f"unsupported action {action}"}

        raw_name = update.get("name") or update.get("skill_name")
        code = update.get("code")
        description = update.get("description") or ""
        if not raw_name or not code:
            return {"applied": False, "reason": "skill update requires name and code"}
        name = slugify_name(raw_name)
        if "async function" not in code or "(bot" not in code:
            return {"applied": False, "reason": "skill code must define an async function that accepts bot"}

        if action == "update_skill" and name not in self.skill_manager.skills:
            action = "add_skill"

        previous = self.harness_skills.get(name, {})
        metadata = previous.get("metadata", {}) if isinstance(previous, dict) else {}
        version = int(metadata.get("version", 0)) + 1
        metadata = {
            **metadata,
            "version": version,
            "updated_at": now_iso(),
            "generation_mode": update.get("generation_mode", "llm_code"),
            "reason": update.get("reason", ""),
            "test_status": update.get("test_status", "unknown"),
            "preconditions": update.get("preconditions", []),
            "success_checks": update.get("success_checks", []),
            "primitive_plan": update.get("primitive_plan", []),
        }
        self.harness_skills[name] = {"code": code, "description": description, "metadata": metadata}
        self.skill_manager.skills[name] = self.harness_skills[name]

        with open(U.f_join(self.compositional_dir, f"{name}.js"), "w", encoding="utf-8") as fp:
            fp.write(code.rstrip() + "\n")
        with open(U.f_join(self.description_dir, f"{name}.txt"), "w", encoding="utf-8") as fp:
            fp.write(description.rstrip() + "\n")
        with open(self.skills_path, "w", encoding="utf-8") as fp:
            json.dump(self.harness_skills, fp, ensure_ascii=False, indent=2)
        with open(self.versions_path, "a", encoding="utf-8") as fp:
            fp.write(
                json.dumps(
                    {
                        "time": now_iso(),
                        "name": name,
                        "version": version,
                        "action": action,
                        "description": description,
                        "generation_mode": metadata["generation_mode"],
                        "test_status": metadata["test_status"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        return {
            "applied": True,
            "action": action,
            "name": name,
            "version": version,
            "path": self.skills_path,
        }

    def _name_for_code(self, code: str) -> str | None:
        for name, entry in self.skill_manager.skills.items():
            if entry.get("code") == code:
                return name
        return None

    def _load_harness_skills(self) -> dict[str, Any]:
        if U.f_exists(self.skills_path):
            return U.load_json(self.skills_path)
        with open(self.skills_path, "w", encoding="utf-8") as fp:
            json.dump({}, fp, ensure_ascii=False, indent=2)
        return {}

    def _merge_harness_skills(self) -> None:
        for name, entry in self.harness_skills.items():
            if not isinstance(entry, dict):
                continue
            if "code" not in entry or "description" not in entry:
                continue
            self.skill_manager.skills[name] = entry

    def _rank_harness_skills(self, query: str) -> list[tuple[str, dict[str, Any]]]:
        terms = {term.lower() for term in query.replace("_", " ").split() if len(term) > 2}
        ranked = []
        for name, entry in self.harness_skills.items():
            text = f"{name} {entry.get('description', '')}".lower()
            score = sum(1 for term in terms if term in text)
            ranked.append((score, name, entry))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [(name, entry) for _, name, entry in ranked]

    def _capability_layers(self, all_names: list[str]) -> dict[str, Any]:
        learned = set(self.learned_skill_names())
        diagnostics = set(self.diagnostic_skill_names())
        return {
            "primitives": PRIMITIVE_REFERENCE,
            "diagnostic_skills": self.diagnostic_skill_names(),
            "base_skills": [name for name in all_names if name not in learned and name not in diagnostics],
            "learned_skills": sorted(learned),
        }

    def _register_diagnostic_skills(self) -> None:
        for name, entry in DIAGNOSTIC_SKILLS.items():
            self.skill_manager.skills[name] = entry

    def _select_skill_code_examples(self, query: str, retrieved_names: list[str]) -> list[dict[str, str]]:
        selected: list[str] = []
        query_terms = {term.lower() for term in query.replace("_", " ").split() if len(term) > 2}

        def add_if_available(name: str) -> None:
            if name in self.skill_manager.skills and name not in selected:
                selected.append(name)

        for name in EXAMPLE_SKILL_PRIORITY:
            text = name.lower()
            description = self.skill_manager.skills.get(name, {}).get("description", "").lower()
            if not query_terms or any(term in text or term in description for term in query_terms):
                add_if_available(name)

        for name in retrieved_names:
            if len(selected) >= 6:
                break
            add_if_available(name)

        examples = []
        for name in selected[:6]:
            entry = self.skill_manager.skills[name]
            code = entry.get("code", "").strip()
            if len(code) > 1800:
                code = code[:1800].rstrip() + "\n// ... truncated ..."
            examples.append(
                {
                    "name": name,
                    "description": entry.get("description", ""),
                    "code": code,
                }
            )
        return examples
