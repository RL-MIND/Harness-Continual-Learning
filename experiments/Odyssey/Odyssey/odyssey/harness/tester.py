from __future__ import annotations

import json
import re
from typing import Any

import odyssey.utils as U

from .capability import PRIMITIVE_REFERENCE
from .common import now_iso


class SkillTestGate:
    def __init__(self, ckpt_dir: str = "ckpt", resume: bool = True):
        self.root = U.f_mkdir(ckpt_dir, "harness", "tests")
        self.path = U.f_join(self.root, "skill_tests.jsonl")
        if not resume:
            U.f_remove(self.path)
        self.primitive_names = {item["name"] for item in PRIMITIVE_REFERENCE}

    def validate_update(
        self,
        update: dict[str, Any],
        known_skill_names: set[str] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(update, dict):
            result = {
                "time": now_iso(),
                "skill_name": "",
                "passed": False,
                "checks": [],
                "errors": ["capability update must be a JSON object"],
                "mode": "static",
            }
            self.append(result)
            return result
        code = update.get("code") or ""
        result = {
            "time": now_iso(),
            "skill_name": update.get("name", ""),
            "passed": True,
            "checks": [],
            "errors": [],
            "mode": "static",
        }
        self._check(result, "defines_async_function", "async function" in code and "(bot" in code)
        self._check(result, "no_undefined_literal", "undefined" not in code)
        self._check(result, "no_eval_or_require", "eval(" not in code and "require(" not in code)
        self._check(result, "no_process_or_fs", "process." not in code and "fs." not in code)
        self._check(result, "no_runtime_capability_summary_access", self._no_runtime_capability_summary_access(code))
        self._check(result, "no_generic_minecraft_item_names", self._no_generic_minecraft_item_names(code))
        self._check(result, "inventory_count_uses_item_ids_not_regex", self._inventory_count_looks_valid(code))
        self._check(result, "does_not_treat_equipment_4_as_offhand", self._does_not_treat_equipment_4_as_offhand(code))
        self._check(result, "balanced_braces", code.count("{") == code.count("}"))
        self._check(result, "primitive_calls_have_required_arguments", self._primitive_calls_look_valid(code))
        self._check(result, "placement_uses_vec3_source", self._placement_looks_valid(code))
        self._check(
            result,
            "no_unknown_harness_helpers",
            self._awaited_helpers_are_known(code, known_skill_names or set()),
        )
        result["passed"] = not result["errors"]
        self.append(result)
        return result

    def live_smoke_test(self, odyssey, update: dict[str, Any]) -> dict[str, Any]:
        test_spec = update.get("test_spec") or {}
        result = {
            "time": now_iso(),
            "skill_name": update.get("name", ""),
            "passed": False,
            "mode": "live_smoke",
            "checks": [],
            "errors": [],
        }
        if not test_spec.get("run_live"):
            result["errors"].append("live test not requested")
            return result
        try:
            if test_spec.get("reset"):
                options = {
                    "mode": test_spec.get("reset_mode", "soft"),
                    "wait_ticks": odyssey.env_wait_ticks,
                    "username": odyssey.username,
                }
                if isinstance(test_spec.get("inventory"), dict):
                    options["inventory"] = test_spec["inventory"]
                odyssey.env.reset(options=options)
            parsed = odyssey._parse_skill_code(update.get("code", ""))
            execution = odyssey.execute_harness_code(parsed, skill_name=update.get("name", "temporaryHarnessSkill"))
            result["passed"] = not execution.get("error")
            result["events_seen"] = len(execution.get("events") or [])
        except Exception as exc:  # noqa: BLE001
            result["errors"].append(str(exc))
        self.append(result)
        return result

    def append(self, result: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(result, ensure_ascii=False) + "\n")

    def _check(self, result: dict[str, Any], name: str, passed: bool) -> None:
        result["checks"].append({"name": name, "passed": bool(passed)})
        if not passed:
            result["errors"].append(name)

    def _primitive_calls_look_valid(self, code: str) -> bool:
        # Do not mistake `async function mineBlock(bot)` for a call to the
        # injected mineBlock primitive. The old regexp made primitive-name
        # collisions fail static validation for an unrelated reason.
        call_code = re.sub(r"(?:async\s+)?function\s+\w+\s*\([^)]*\)", "", code)
        for name in self.primitive_names:
            if name not in call_code:
                continue
            for args in re.findall(rf"(?<![.\w]){name}\s*\(([^)]*)\)", call_code):
                parts = self._split_args(args)
                if name in {"placeItem"} and len(parts) < 3:
                    return False
                if name in {"craftItem", "mineBlock"} and len(parts) < 2:
                    return False
                if name == "equipItem" and len(parts) != 2:
                    return False
                if name == "smeltItem" and len(parts) != 4:
                    return False
                if parts and parts[0] != "bot":
                    return False
                if any(part == "undefined" or part == "null" for part in parts):
                    return False
                if name in {"craftItem", "mineBlock"} and len(parts) >= 3 and parts[2].startswith("{"):
                    return False
                if name == "smeltItem" and len(parts) == 4 and parts[3].startswith("{"):
                    return False
        return True

    def _awaited_helpers_are_known(self, code: str, known_skill_names: set[str]) -> bool:
        """Reject invented global helpers before a candidate reaches Minecraft.

        Direct Mineflayer APIs are methods (for example ``bot.equip``), while
        globally awaited calls must be an injected primitive/helper, a known
        skill, or a function declared in the candidate itself.
        """
        declared = set(re.findall(r"(?:async\s+)?function\s+(\w+)\s*\(", code))
        allowed = self.primitive_names | {"findSuitablePosition"} | known_skill_names | declared
        awaited = re.findall(r"await\s+([A-Za-z_$][\w$]*)\s*\(", code)
        return all(name in allowed for name in awaited)

    def _placement_looks_valid(self, code: str) -> bool:
        if "placeItem" not in code:
            return True
        return "findSuitablePosition(bot)" in code or "new Vec3(" in code or "Block.position" in code

    def _no_runtime_capability_summary_access(self, code: str) -> bool:
        banned = [
            "capability_summary",
            "primitive_reference",
            "task_state",
            "success_criteria",
        ]
        return not any(token in code for token in banned)

    def _no_generic_minecraft_item_names(self, code: str) -> bool:
        banned_names = {"log", "logs", "plank", "planks", "wooden_planks"}
        for match in re.findall(r"""["']([a-z_]+)["']""", code):
            if match in banned_names:
                return False
        return True

    def _inventory_count_looks_valid(self, code: str) -> bool:
        for args in re.findall(r"bot\.inventory\.count\s*\(([^)]*)\)", code):
            stripped = args.strip()
            if not stripped:
                return False
            if stripped.startswith("/"):
                return False
            if re.match(r"""^["'][a-z_]+["']$""", stripped):
                return False
        return True

    def _does_not_treat_equipment_4_as_offhand(self, code: str) -> bool:
        lowered = code.lower()
        bad_patterns = [
            "equipment[4] is off",
            "equipment slot 4) to inventory",
            "equipment slot 4) from off",
            "equipment slot 4 to inventory",
            "slot 4 is off",
            "slot [4] is off",
        ]
        return not any(pattern in lowered for pattern in bad_patterns)

    def _split_args(self, args: str) -> list[str]:
        parts: list[str] = []
        current: list[str] = []
        quote = ""
        depth = 0
        escape = False
        for char in args:
            if escape:
                current.append(char)
                escape = False
                continue
            if char == "\\":
                current.append(char)
                escape = True
                continue
            if quote:
                current.append(char)
                if char == quote:
                    quote = ""
                continue
            if char in {"'", '"', "`"}:
                current.append(char)
                quote = char
                continue
            if char in "([{":
                depth += 1
                current.append(char)
                continue
            if char in ")]}":
                depth = max(0, depth - 1)
                current.append(char)
                continue
            if char == "," and depth == 0:
                value = "".join(current).strip()
                if value:
                    parts.append(value)
                current = []
                continue
            current.append(char)
        value = "".join(current).strip()
        if value:
            parts.append(value)
        return parts
