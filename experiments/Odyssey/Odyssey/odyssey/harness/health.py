from __future__ import annotations

import json
from typing import Any

import odyssey.utils as U

from .common import now_iso


class SkillHealthStore:
    def __init__(self, ckpt_dir: str = "ckpt", resume: bool = True):
        self.root = U.f_mkdir(ckpt_dir, "harness")
        self.path = U.f_join(self.root, "skill_health.json")
        if not resume:
            U.f_remove(self.path)
        self.health = U.load_json(self.path) if U.f_exists(self.path) else {}

    def summarize(self, skill_names: list[str]) -> dict[str, Any]:
        return {name: self.health.get(name, self._empty()) for name in skill_names if name}

    def repeatedly_failed_contexts(self, minimum_failures: int = 2) -> dict[str, dict[str, Any]]:
        """Return conservative historical quarantines for never-successful skills."""
        result = {}
        for name, entry in self.health.items():
            if (
                int(entry.get("failure_count", 0)) >= minimum_failures
                and int(entry.get("success_count", 0)) == 0
            ):
                contexts = entry.get("failure_contexts") or []
                result[name] = contexts[-1] if contexts else {
                    "skill_name": name,
                    "reason": "repeated_historical_failures",
                    "critique": entry.get("last_critique", ""),
                }
        return result

    def update(
        self,
        skill_name: str,
        success: bool,
        evaluation: dict[str, Any] | None = None,
        failure_context: dict[str, Any] | None = None,
    ) -> None:
        if not skill_name:
            return
        entry = self.health.setdefault(skill_name, self._empty())
        if success:
            entry["success_count"] += 1
        else:
            entry["failure_count"] += 1
        total = entry["success_count"] + entry["failure_count"]
        entry["confidence"] = round(entry["success_count"] / total, 3) if total else 0.0
        entry["last_success"] = bool(success)
        entry["last_updated"] = now_iso()
        if evaluation:
            entry["last_critique"] = evaluation.get("critique", "")
            risks = evaluation.get("regression_risks") or []
            if risks:
                entry["last_risks"] = risks
        if failure_context and not success:
            contexts = entry.setdefault("failure_contexts", [])
            contexts.append(failure_context)
            entry["failure_contexts"] = contexts[-5:]
        self.save()

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as fp:
            json.dump(self.health, fp, ensure_ascii=False, indent=2)

    def _empty(self) -> dict[str, Any]:
        return {
            "success_count": 0,
            "failure_count": 0,
            "confidence": 0.0,
            "last_success": None,
            "last_critique": "",
            "last_risks": [],
            "last_updated": None,
            "failure_contexts": [],
        }
