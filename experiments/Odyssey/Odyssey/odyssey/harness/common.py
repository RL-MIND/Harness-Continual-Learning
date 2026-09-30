from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from langchain.schema import HumanMessage, SystemMessage

from odyssey.agents.llama import call_with_messages
from odyssey.utils.json_utils import fix_and_parse_json


EQUIPMENT_SLOT_NAMES = [
    "head",
    "chest",
    "legs",
    "feet",
    "main_hand",
    "off_hand",
]

EQUIPMENT_SLOT_SCHEMA = {
    "array_order": EQUIPMENT_SLOT_NAMES,
    "description": (
        "status.equipment is ordered as [head, chest, legs, feet, main_hand, off_hand]. "
        "These slots are equipment/held-item views, not extra item-count containers."
    ),
}

INVENTORY_SCHEMA = {
    "description": (
        "inventory is the item-count view reported by Mineflayer inventory.items(); "
        "a held hotbar item may also appear in status.equipment_slots.main_hand."
    ),
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def slugify_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "", name.replace("-", "_").replace(" ", "_"))
    if not cleaned:
        raise ValueError("empty name")
    if cleaned[0].isdigit():
        cleaned = f"skill_{cleaned}"
    return cleaned


def event_digest(events: list[tuple[str, dict[str, Any]]] | None) -> dict[str, Any]:
    if not events:
        return {
            "events": [],
            "latest_observation": None,
            "chat": [],
            "errors": [],
            "inventory_schema": INVENTORY_SCHEMA,
            "equipment_slot_schema": EQUIPMENT_SLOT_SCHEMA,
        }
    chat = []
    errors = []
    latest = None
    compact_events = []
    for event_type, event in events:
        if event_type == "onChat":
            chat.append(event.get("onChat"))
        elif event_type == "onError":
            errors.append(event.get("onError"))
        elif event_type == "observe":
            latest = _annotate_observation(event)
        compact_events.append({"type": event_type, "value": _annotate_observation(event)})
    return {
        "events": compact_events[-12:],
        "latest_observation": latest,
        "chat": [item for item in chat if item],
        "errors": [item for item in errors if item],
        "inventory_schema": INVENTORY_SCHEMA,
        "equipment_slot_schema": EQUIPMENT_SLOT_SCHEMA,
    }


def _annotate_observation(event: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(event, dict):
        return event
    status = event.get("status")
    if not isinstance(status, dict):
        return event
    equipment = status.get("equipment")
    if not isinstance(equipment, list):
        return event

    annotated = dict(event)
    annotated_status = dict(status)
    slots = {}
    for index, name in enumerate(EQUIPMENT_SLOT_NAMES):
        slots[name] = equipment[index] if index < len(equipment) else None
    annotated_status["equipment_slots"] = slots
    annotated_status["equipment_slot_schema"] = EQUIPMENT_SLOT_SCHEMA
    annotated["status"] = annotated_status
    return annotated


def dumps_compact(data: Any, limit: int | None = None) -> str:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if limit and len(text) > limit:
        return text[:limit] + "\n...<truncated>"
    return text


def call_json(system_prompt: str, user_payload: dict[str, Any], model_name: str, max_retries: int = 3) -> dict[str, Any]:
    last_error: Exception | None = None
    user_content = dumps_compact(user_payload, limit=28000)
    for _ in range(max_retries):
        response = call_with_messages(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ],
            model_name=model_name,
        ).content
        try:
            return fix_and_parse_json(response)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            user_content = (
                user_content
                + "\n\nThe previous response was not valid JSON. Return only one JSON object."
            )
    raise RuntimeError(f"Failed to parse LLM JSON response: {last_error}")
