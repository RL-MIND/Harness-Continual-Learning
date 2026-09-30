from __future__ import annotations

import re
from typing import Any

from .common import call_json


TOOL_SUFFIXES = (
    "_axe",
    "_pickaxe",
    "_shovel",
    "_hoe",
    "_sword",
)
TOOL_NAMES = {"shield", "bow", "crossbow", "fishing_rod", "shears", "flint_and_steel"}

CHINESE_NUMBERS = {
    "\u4e00": 1,
    "\u4e24": 2,
    "\u4e8c": 2,
    "\u4e09": 3,
    "\u56db": 4,
    "\u4e94": 5,
    "\u516d": 6,
    "\u4e03": 7,
    "\u516b": 8,
    "\u4e5d": 9,
    "\u5341": 10,
}

SEMANTIC_CONTRACT_VERSION = 2

SEMANTIC_CONTRACT_PROMPT = """You compile a Minecraft instruction into a small,
observation-grounded completion contract. The instruction can be Chinese or English.
Understand its semantic roles; never infer a resource target merely because a tool name
contains a material word (for example, a wooden pickaxe is a tool, not a request for logs).

Return only JSON:
{
  "kind": "inventory_delta" | "none",
  "targets": [
    {"item": "canonical_minecraft_item_id", "amount": positive_integer}
  ],
  "protected_tools": ["canonical_tool_item_id"]
}

Use `inventory_delta` only when the instruction has an observable inventory outcome.
For a placement, navigation, inspection, combat, or building-only task whose outcome
cannot be established from item counts, return kind `none` and empty targets.
Use canonical lower-case Minecraft item ids with underscores. For gathering or crafting,
the target is only the newly obtained resource or product explicitly requested by the
instruction. For smelting, include only the output item. Never include consumed recipe
ingredients, fuel, raw inputs, implicit preparation steps, or intermediate products as
targets, even if they are needed to execute the task. Include only tools explicitly
required to be retained and present in the baseline inventory. Do not invent targets,
tools, quantities, or world observations."""

ITEM_ID = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def build_completion_contract(
    instruction: str,
    baseline_inventory: dict[str, Any] | None,
    *,
    model_name: str | None = None,
) -> dict[str, Any] | None:
    """Compile task semantics with an LLM, then validate a deterministic contract.

    The LLM interprets natural language once at transaction creation. The resulting
    contract is subsequently evaluated only against observed inventory, so the model
    never gets to declare a task complete. If semantic compilation is unavailable or
    invalid, return only a narrow unambiguous fallback rather than guessing from words
    inside Chinese tool names.
    """
    baseline = _normalize_inventory(baseline_inventory)
    if model_name:
        try:
            proposal = call_json(
                SEMANTIC_CONTRACT_PROMPT,
                {
                    "instruction": instruction,
                    "baseline_inventory": baseline,
                    "known_baseline_tools": [name for name in baseline if _is_tool(name)],
                },
                model_name,
            )
            contract = _validated_semantic_contract(instruction, baseline, proposal)
            if contract is not None:
                return contract
        except Exception:  # Semantic parsing must not make a task un-runnable.
            pass
    return _narrow_rule_fallback(instruction, baseline)


def _validated_semantic_contract(
    instruction: str,
    baseline: dict[str, int],
    proposal: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(proposal, dict):
        return None
    if proposal.get("kind") == "none":
        return None
    if proposal.get("kind") != "inventory_delta":
        return None

    targets = []
    seen_items = set()
    for target in proposal.get("targets", []):
        if not isinstance(target, dict):
            return None
        item = target.get("item")
        direction = target.get("direction")
        amount = target.get("amount")
        if (
            not isinstance(item, str)
            or not ITEM_ID.fullmatch(item)
            or item in seen_items
            or direction not in {None, "increase"}
            or not isinstance(amount, int)
            or isinstance(amount, bool)
            or not 1 <= amount <= 64
        ):
            return None
        seen_items.add(item)
        initial = int(baseline.get(item, 0))
        targets.append(
            {
                "item": item,
                "baseline_count": initial,
                "delta": amount,
                "target_count": initial + amount,
                "comparison": ">=",
            }
        )
    if not targets:
        return None

    protected_tools = {}
    for item in proposal.get("protected_tools", []):
        if not isinstance(item, str) or item in protected_tools:
            return None
        if not _is_tool(item) or int(baseline.get(item, 0)) <= 0:
            return None
        protected_tools[item] = int(baseline[item])
    return {
        "type": "inventory_delta",
        "instruction": instruction,
        "baseline_inventory": baseline,
        "targets": targets,
        "protected_tools": protected_tools,
        "semantic_source": "llm_validated",
        "semantic_contract_version": SEMANTIC_CONTRACT_VERSION,
        "notes": [
            "Targets are fixed from the original task baseline.",
            "Targets specify their own comparison direction against that baseline.",
        ],
    }


def _narrow_rule_fallback(instruction: str, baseline: dict[str, int]) -> dict[str, Any] | None:
    """Preserve only the unambiguous raw-iron fallback when the LLM is unavailable."""
    targets = [
        target
        for target in (_parse_smelting_targets(instruction, baseline) or [])
        if target.get("comparison") == ">="
    ]
    if not targets:
        return None
    protected_tools = {
        name: count
        for name, count in baseline.items()
        if _mentions_tool_retention(instruction) and _is_tool(name)
    }
    return {
        "type": "inventory_delta",
        "instruction": instruction,
        "baseline_inventory": baseline,
        "targets": targets,
        "protected_tools": protected_tools,
        "semantic_source": "narrow_rule_fallback",
        "semantic_contract_version": SEMANTIC_CONTRACT_VERSION,
        "notes": [
            "LLM semantic compilation was unavailable; only an unambiguous smelting contract was used.",
            "Targets are fixed from the original task baseline.",
        ],
    }


def evaluate_completion_contract(
    contract: dict[str, Any] | None,
    inventory: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not contract:
        return None
    current = _normalize_inventory(inventory)
    evidence: list[str] = []
    success = True

    for target in contract.get("targets", []):
        item = str(target.get("item", ""))
        required = int(target.get("target_count", 0))
        actual = int(current.get(item, 0))
        comparison = target.get("comparison", ">=")
        passed = actual <= required if comparison == "<=" else actual >= required
        if not passed:
            success = False
        evidence.append(f"{item}: current {actual}, target {comparison} {required}")

    for item, required_count in (contract.get("protected_tools") or {}).items():
        actual = int(current.get(item, 0))
        required = int(required_count)
        if actual < required:
            success = False
        evidence.append(f"tool {item}: current {actual}, baseline {required}")

    return {
        "success": success,
        "critique": _contract_critique(success),
        "evidence": evidence,
        "regression_risks": [],
        "contract": contract,
    }


def contract_progress(
    contract: dict[str, Any] | None,
    inventory: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return remaining observation-grounded work without redefining a goal."""
    if not contract:
        return None
    current = _normalize_inventory(inventory)
    targets = []
    for target in contract.get("targets", []):
        item = str(target.get("item", ""))
        required = int(target.get("target_count", 0))
        actual = int(current.get(item, 0))
        comparison = target.get("comparison", ">=")
        satisfied = actual <= required if comparison == "<=" else actual >= required
        remaining = max(0, actual - required) if comparison == "<=" else max(0, required - actual)
        targets.append(
            {
                **target,
                "current_count": actual,
                "remaining_count": remaining,
                "satisfied": satisfied,
            }
        )
    protected_items = []
    for item, baseline_count in (contract.get("protected_tools") or {}).items():
        actual = int(current.get(item, 0))
        required = int(baseline_count)
        protected_items.append(
            {
                "item": item,
                "baseline_count": required,
                "current_count": actual,
                "satisfied": actual >= required,
            }
        )
    return {
        "contract_type": contract.get("type"),
        "targets": targets,
        "protected_items": protected_items,
        "satisfied": all(target["satisfied"] for target in targets)
        and all(item["satisfied"] for item in protected_items),
    }


def apply_contract_result(evaluation: dict[str, Any], contract_result: dict[str, Any] | None) -> dict[str, Any]:
    if not contract_result:
        return evaluation
    merged = dict(evaluation)
    merged["contract_evaluation"] = contract_result
    merged["success"] = bool(contract_result["success"])
    if contract_result["success"]:
        merged["critique"] = contract_result["critique"]
        merged["evidence"] = contract_result["evidence"]
        merged["regression_risks"] = contract_result["regression_risks"]
    else:
        merged["critique"] = (
            (merged.get("critique") or "").strip()
            + " Contract check failed: "
            + "; ".join(contract_result.get("evidence", []))
        ).strip()
    return merged


def _parse_smelting_targets(instruction: str, baseline: dict[str, int]) -> list[dict[str, Any]] | None:
    """Parse raw-iron smelting into directional inventory constraints.

    Coal is consumable material. Its inventory count may decrease by any
    amount (or remain unchanged when a furnace already has fuel), and it is
    never a protected tool or a completion target.
    """
    text = instruction.lower()
    if not any(token in text for token in ("smelt", "\u7194\u70bc", "\u70e7")):
        return None
    if not any(token in text for token in ("raw_iron", "\u7c97\u94c1")):
        return None
    amount = _parse_amount(instruction) or 1
    raw_iron = baseline.get("raw_iron", 0)
    iron_ingot = baseline.get("iron_ingot", 0)
    return [
        {
            "item": "iron_ingot",
            "baseline_count": iron_ingot,
            "delta": amount,
            "target_count": iron_ingot + amount,
            "comparison": ">=",
        },
        {
            "item": "raw_iron",
            "baseline_count": raw_iron,
            "delta": -amount,
            "target_count": max(0, raw_iron - amount),
            "comparison": "<=",
        },
    ]


def _parse_amount(instruction: str) -> int | None:
    match = re.search(r"(\d+)\s*(?:\u4efd|\u4e2a|\u5757|\u6839)", instruction)
    if match:
        return int(match.group(1))
    for text, value in sorted(CHINESE_NUMBERS.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"{text}\s*(?:\u4efd|\u4e2a|\u5757|\u6839)", instruction):
            return value
    match = re.search(r"(\d+)", instruction)
    return int(match.group(1)) if match else None

def _mentions_tool_retention(instruction: str) -> bool:
    return any(
        token in instruction
        for token in (
            "\u4fdd\u7559\u6240\u6709\u5de5\u5177",
            "\u4e0d\u8981\u4e22\u5f03\u4efb\u4f55\u5de5\u5177",
            "\u4e0d\u4e22\u5f03\u5de5\u5177",
        )
    )


def _is_tool(item_name: str) -> bool:
    return item_name.endswith(TOOL_SUFFIXES) or item_name in TOOL_NAMES


def _normalize_inventory(inventory: dict[str, Any] | None) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, value in (inventory or {}).items():
        try:
            result[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return result


def _contract_critique(success: bool) -> str:
    return "Completion contract satisfied." if success else "Completion contract not yet satisfied."
