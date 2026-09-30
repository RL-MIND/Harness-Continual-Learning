from __future__ import annotations

import json
from typing import Any

import odyssey.utils as U

from .common import event_digest, now_iso


def latest_observation(events: list[tuple[str, dict[str, Any]]] | None) -> dict[str, Any]:
    if not events:
        return {}
    for event_type, event in reversed(events):
        if event_type == "observe":
            return event
    return {}


def inventory_from(events: list[tuple[str, dict[str, Any]]] | None) -> dict[str, Any]:
    return latest_observation(events).get("inventory", {}) or {}


class ExecutionRecorder:
    def __init__(self, ckpt_dir: str = "ckpt", resume: bool = True):
        self.root = U.f_mkdir(ckpt_dir, "harness", "execution_records")
        self.path = U.f_join(self.root, "records.jsonl")
        if not resume:
            U.f_remove(self.path)

    def build_record(
        self,
        *,
        instruction: str,
        step_index: int,
        route: dict[str, Any],
        events_before: list[tuple[str, dict[str, Any]]] | None,
        events_after: list[tuple[str, dict[str, Any]]] | None,
        execution: dict[str, Any],
        evaluation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        before_obs = latest_observation(events_before)
        after_obs = latest_observation(events_after)
        digest_before = event_digest(events_before)
        digest_after = event_digest(events_after)
        annotated_before_obs = digest_before.get("latest_observation") or before_obs
        annotated_after_obs = digest_after.get("latest_observation") or after_obs
        return {
            "time": now_iso(),
            "instruction": instruction,
            "step_index": step_index,
            "selected_skill": route.get("skill_name", ""),
            "decision": route.get("decision", ""),
            "pre_inventory": before_obs.get("inventory", {}) or {},
            "post_inventory": after_obs.get("inventory", {}) or {},
            "pre_status": annotated_before_obs.get("status", {}) or {},
            "post_status": annotated_after_obs.get("status", {}) or {},
            "nearby_blocks_after": after_obs.get("voxels", [])[:40] if isinstance(after_obs.get("voxels", []), list) else [],
            "chat": digest_after.get("chat", []),
            "errors": digest_after.get("errors", []),
            "execution_error": execution.get("error", ""),
            "success": bool((evaluation or {}).get("success", False)),
            "evaluation": evaluation or {},
        }

    def append(self, record: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
