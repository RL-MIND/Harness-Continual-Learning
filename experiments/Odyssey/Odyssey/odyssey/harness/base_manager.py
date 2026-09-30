from __future__ import annotations

import json
import re
from typing import Any

import odyssey.utils as U

from .common import now_iso


_ASSET_NAME = re.compile(r"^[a-z0-9_]+$")
_READY_STATUSES = {"present", "placed"}


class BaseManager:
    """Persistent registry of model-designated world assets.

    This layer deliberately has no knowledge of task wording or Minecraft
    facility categories. The model designates an asset in its route; selected
    capabilities declare abstract world dependencies. The manager binds and
    verifies concrete coordinates, and may rebind a stale coordinate only to
    an actually observed same-type block near the durable base anchor.
    """

    def __init__(self, ckpt_dir: str = "ckpt", resume: bool = True):
        self.root = U.f_mkdir(ckpt_dir, "harness")
        self.path = U.f_join(self.root, "base_state.json")
        self.state = self._load() if resume else self._empty_state()

    def maintain(self, block_name: str, events: list[tuple[str, dict[str, Any]]]) -> dict[str, Any] | None:
        """Create or repair one model-designated persistent asset."""
        if not self._valid_name(block_name):
            return None
        self.reconcile(events)
        entry = self.state["assets"].get(block_name, {})
        position = self._position(entry.get("position")) if entry.get("registered") else None
        return {
            "code": self._maintain_code(
                self.state.get("anchor"),
                block_name,
                position,
                self._rejected_positions(),
            ),
            "asset": block_name,
        }

    def assets_for_requirements(self, requirements: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
        selected: list[dict[str, Any]] = []
        missing: list[str] = []
        for requirement in requirements:
            role = str(requirement.get("role", "world_asset"))
            accepted = [name for name in requirement.get("accepted_blocks", []) if self._valid_name(name)]
            match = next(
                (
                    {"role": role, "block_name": name, "position": self._position(entry.get("position"))}
                    for name in accepted
                    for entry in [self.state["assets"].get(name, {})]
                    if entry.get("status") in _READY_STATUSES and self._position(entry.get("position"))
                ),
                None,
            )
            if match:
                selected.append(match)
            else:
                missing.append(role)
        return selected, missing

    def reach_code(self, assets: list[dict[str, Any]]) -> str:
        calls = []
        for asset in assets:
            calls.append(
                "  await useRegisteredAsset(bot, "
                + json.dumps(asset["block_name"])
                + ", "
                + json.dumps(asset["position"])
                + ");"
            )
        return "async function reachRegisteredAssets(bot) {\n" + "\n".join(calls) + "\n}"

    def protection_call(self) -> str:
        """Return a Harness prologue that protects registered coordinates.

        Failed assets remain protected at their durable coordinate: an empty
        stale coordinate is harmless, while silently allowing a temporarily
        unloaded/reappearing asset to be mined would violate the registry.
        """
        assets = []
        for block_name, entry in self.state.get("assets", {}).items():
            if not isinstance(entry, dict) or not entry.get("registered"):
                continue
            position = self._position(entry.get("position"))
            if self._valid_name(block_name) and position:
                assets.append({"block_name": block_name, "position": position})
        assets.sort(
            key=lambda asset: (
                asset["block_name"],
                asset["position"]["x"],
                asset["position"]["y"],
                asset["position"]["z"],
            )
        )
        return f"await protectRegisteredAssets(bot, {json.dumps(assets)});"

    def reconcile(self, events: list[tuple[str, dict[str, Any]]] | None) -> None:
        report = self._latest_report(events)
        if not report:
            return
        anchor = self._position(report.get("anchor"))
        if anchor:
            self.state["anchor"] = anchor
        name = report.get("asset")
        position = self._position(report.get("position"))
        if self._valid_name(name) and position:
            status = str(report.get("status", "unknown"))
            existing = self.state["assets"].get(name, {})
            existing_position = self._position(existing.get("position"))
            if status in _READY_STATUSES:
                self.state["assets"][name] = {
                    "position": position,
                    "status": status,
                    "registered": True,
                    "verified_at": now_iso(),
                }
                self.state["rejected_positions"].pop(self._position_key(position), None)
            elif existing.get("registered") and existing_position == position:
                # A failed repair at the registered coordinate is observable.
                # Relocation is accepted only through the READY branch above,
                # which requires a concrete same-type world observation from
                # maintainRegisteredAsset rather than a guessed coordinate.
                existing["status"] = f"repair_failed:{status}"
                existing["verified_at"] = now_iso()
                self.state["assets"][name] = existing
            elif self._rejectable_status(status):
                # A candidate is not part of the durable registry until it has
                # actually been observed. Remember failed candidates so the
                # next attempt can make progress instead of looping forever.
                self.state["rejected_positions"][self._position_key(position)] = {
                    "position": position,
                    "asset": name,
                    "status": status,
                    "detail": str(report.get("detail", ""))[:500],
                    "rejected_at": now_iso(),
                }
        self._save()

    def asset_ready(self, name: str) -> bool:
        entry = self.state["assets"].get(name, {})
        return entry.get("status") in _READY_STATUSES and bool(self._position(entry.get("position")))

    def public_state(self) -> dict[str, Any]:
        return {
            "anchor": self.state.get("anchor"),
            "assets": self.state.get("assets", {}),
            "policy": "Registered assets use verified world coordinates; stale coordinates may rebind only to an observed same-type block near the base anchor. The model chooses persistent assets; capabilities declare dependencies.",
        }

    def _maintain_code(
        self,
        anchor: dict[str, int] | None,
        name: str,
        position: dict[str, int] | None,
        rejected_positions: list[dict[str, int]],
    ) -> str:
        return (
            "async function maintainRegisteredWorldAsset(bot) {\n"
            "  await maintainRegisteredAsset(bot, "
            f"{json.dumps(anchor)}, {json.dumps(name)}, {json.dumps(position)}, "
            f"{json.dumps(rejected_positions)});\n"
            "}"
        )

    def _latest_report(self, events: list[tuple[str, dict[str, Any]]] | None) -> dict[str, Any] | None:
        for event_type, event in reversed(events or []):
            if event_type != "onSave" or not isinstance(event, dict):
                continue
            raw = event.get("onSave")
            if not isinstance(raw, str) or not raw.startswith("harness_asset:"):
                continue
            try:
                report = json.loads(raw[len("harness_asset:"):])
            except json.JSONDecodeError:
                continue
            if isinstance(report, dict):
                return report
        return None

    def _load(self) -> dict[str, Any]:
        try:
            with open(self.path, "r", encoding="utf-8") as fp:
                value = json.load(fp)
        except (OSError, json.JSONDecodeError):
            return self._empty_state()
        if not isinstance(value, dict):
            return self._empty_state()
        if isinstance(value.get("assets"), dict):
            value.setdefault("rejected_positions", {})
            for entry in value["assets"].values():
                if isinstance(entry, dict) and entry.get("status") in _READY_STATUSES:
                    entry.setdefault("registered", True)
            return value
        # One-way migration from the earlier fixed-facility state format.
        facilities = value.get("facilities")
        if isinstance(facilities, dict):
            for entry in facilities.values():
                if isinstance(entry, dict) and entry.get("status") in _READY_STATUSES:
                    entry.setdefault("registered", True)
            return {
                "version": 3,
                "anchor": value.get("anchor"),
                "assets": facilities,
                "rejected_positions": {},
                "updated_at": now_iso(),
            }
        return self._empty_state()

    def _save(self) -> None:
        self.state["version"] = 3
        self.state["updated_at"] = now_iso()
        with open(self.path, "w", encoding="utf-8") as fp:
            json.dump(self.state, fp, ensure_ascii=False, indent=2)

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "version": 3,
            "anchor": None,
            "assets": {},
            "rejected_positions": {},
            "updated_at": now_iso(),
        }

    def _rejected_positions(self) -> list[dict[str, int]]:
        records = self.state.get("rejected_positions", {})
        if not isinstance(records, dict):
            return []
        positions = []
        for record in records.values():
            position = self._position(record.get("position")) if isinstance(record, dict) else None
            if position and position not in positions:
                positions.append(position)
        return positions

    @staticmethod
    def _position_key(position: dict[str, int]) -> str:
        return f"{position['x']},{position['y']},{position['z']}"

    @staticmethod
    def _rejectable_status(status: str) -> bool:
        return status in {"placement_failed", "placement_unverified"} or status.startswith("conflict:")

    @staticmethod
    def _valid_name(value: Any) -> bool:
        return isinstance(value, str) and bool(_ASSET_NAME.fullmatch(value))

    @staticmethod
    def _position(value: Any) -> dict[str, int] | None:
        if not isinstance(value, dict):
            return None
        try:
            return {axis: int(value[axis]) for axis in ("x", "y", "z")}
        except (KeyError, TypeError, ValueError):
            return None
