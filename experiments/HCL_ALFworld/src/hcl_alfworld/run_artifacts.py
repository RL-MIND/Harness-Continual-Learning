from __future__ import annotations

import copy
import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _now() -> datetime:
    return datetime.now().astimezone()


def _json_fingerprint(config: Dict[str, Any]) -> str:
    ignored = {"_config_path", "_project_root", "output_dir"}
    comparable = copy.deepcopy(
        {key: value for key, value in config.items() if key not in ignored}
    )
    # Transport controls may be safely tuned when resuming; they do not change
    # the model, prompts, games, Harness state, or evaluation semantics.
    agent = comparable.get("agent", {})
    agent.pop("request_timeout_seconds", None)
    agent.pop("transport_max_retries", None)
    agent.pop("transport_retry_base_seconds", None)
    agent.pop("transport_retry_max_seconds", None)
    agent.pop("transport_retry_jitter", None)
    # Progress was enabled by default before the explicit switch was added.
    # Treat an omitted switch and explicit true as the same resume protocol;
    # explicit false remains fingerprint-significant.
    if agent.get("episode_progress_enabled") is True:
        agent.pop("episode_progress_enabled", None)
    # Evaluation concurrency changes throughput only. It is safe to tune between
    # resume attempts because committed results retain their original order and
    # every worker reads the same frozen Harness snapshot.
    execution = comparable.get("execution", {})
    execution.pop("evaluation_workers", None)
    if not execution:
        comparable.pop("execution", None)
    encoded = json.dumps(
        comparable, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RunArtifacts:
    """Own one timestamped run directory and its resumable audit artifacts."""

    def __init__(
        self,
        base_dir: str | Path,
        config: Dict[str, Any],
        resume_dir: Optional[str | Path] = None,
    ):
        self.config = config
        self._write_lock = threading.RLock()
        self.base_dir = Path(base_dir).resolve()
        self.fingerprint = _json_fingerprint(config)
        if resume_dir is None:
            self.run_dir = self._new_run_dir()
            self.attempt = 1
            self.metadata: Dict[str, Any] = {
                "schema_version": 1,
                "run_id": self.run_dir.name,
                "status": "created",
                "created_at": _now().isoformat(),
                "updated_at": _now().isoformat(),
                "completed_at": None,
                "attempts": self.attempt,
                "config_path": config.get("_config_path"),
                "config_fingerprint": self.fingerprint,
                "checkpoint_granularity": "episode",
            }
            self._prepare_directories()
            self._write_config_snapshot()
            self._write_json_atomic(self.run_dir / "run.json", self.metadata)
        else:
            candidate = Path(resume_dir)
            if not candidate.is_absolute():
                candidate = Path(config["_project_root"]) / candidate
            self.run_dir = candidate.resolve()
            metadata_path = self.run_dir / "run.json"
            checkpoint_path = self.run_dir / "checkpoint.json"
            if not metadata_path.exists() or not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Resume directory lacks run.json or checkpoint.json: {self.run_dir}"
                )
            self.metadata = self._read_json(metadata_path)
            if self.metadata.get("config_fingerprint") != self.fingerprint:
                raise ValueError(
                    "Resume config does not match the config that created this run."
                )
            self.attempt = int(self.metadata.get("attempts", 1)) + 1
            self.metadata["attempts"] = self.attempt
            self.metadata["status"] = "resuming"
            self.metadata["checkpoint_granularity"] = "episode"
            self.metadata["updated_at"] = _now().isoformat()
            self._prepare_directories()
            self._write_json_atomic(metadata_path, self.metadata)

        self.episode_sequence = self._next_sequence(self.run_dir / "episodes", "*.json")
        self.update_sequence = self._next_sequence(self.run_dir / "updates", "*.json")

    def _new_run_dir(self) -> Path:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        timestamp = _now().strftime("%Y%m%d_%H%M%S_%f")
        candidate = self.base_dir / timestamp
        suffix = 1
        while candidate.exists():
            candidate = self.base_dir / f"{timestamp}_{suffix}"
            suffix += 1
        return candidate

    def _prepare_directories(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "episodes").mkdir(exist_ok=True)
        (self.run_dir / "updates").mkdir(exist_ok=True)

    def _write_config_snapshot(self) -> None:
        snapshot = {
            key: value
            for key, value in self.config.items()
            if key not in {"_config_path", "_project_root"}
        }
        path = self.run_dir / "config_snapshot.yaml"
        temporary = path.with_suffix(".yaml.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(snapshot, handle, allow_unicode=True, sort_keys=False)
        temporary.replace(path)

    @staticmethod
    def _next_sequence(directory: Path, pattern: str) -> int:
        return len(list(directory.glob(pattern))) + 1

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise TypeError(f"Expected a JSON object in {path}")
        return value

    @staticmethod
    def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)

    @staticmethod
    def _append_jsonl(path: Path, data: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()

    def event(self, kind: str, message: str, **details: Any) -> Dict[str, Any]:
        payload = {
            "timestamp": _now().isoformat(),
            "attempt": self.attempt,
            "kind": kind,
            "message": message,
            "details": details,
        }
        with self._write_lock:
            self._append_jsonl(self.run_dir / "events.jsonl", payload)
            detail_text = json.dumps(details, ensure_ascii=False, sort_keys=True)
            with (self.run_dir / "run.log").open("a", encoding="utf-8") as handle:
                handle.write(
                    f"[{payload['timestamp']}] attempt={self.attempt} {kind}: "
                    f"{message} {detail_text}\n"
                )
                handle.flush()
        return payload

    def write_episode(self, payload: Dict[str, Any]) -> Path:
        enriched = {
            "logged_at": _now().isoformat(),
            "attempt": self.attempt,
            **payload,
        }
        with self._write_lock:
            self._append_jsonl(self.run_dir / "episodes.jsonl", enriched)
            episode_id = str(payload["record"]["episode_id"])
            path = self.run_dir / "episodes" / f"{self.episode_sequence:06d}_{episode_id}.json"
            self.episode_sequence += 1
            self._write_json_atomic(path, enriched)
        return path

    def write_update(self, payload: Dict[str, Any]) -> Path:
        enriched = {
            "logged_at": _now().isoformat(),
            "attempt": self.attempt,
            **payload,
        }
        with self._write_lock:
            self._append_jsonl(self.run_dir / "updates.jsonl", enriched)
            episode_id = str(payload["episode_id"])
            path = self.run_dir / "updates" / f"{self.update_sequence:06d}_{episode_id}.json"
            self.update_sequence += 1
            self._write_json_atomic(path, enriched)
        return path

    def save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        payload = {
            "schema_version": 1,
            "saved_at": _now().isoformat(),
            "attempt": self.attempt,
            **checkpoint,
        }
        self._write_json_atomic(self.run_dir / "checkpoint.json", payload)

    def load_checkpoint(self) -> Dict[str, Any]:
        return self._read_json(self.run_dir / "checkpoint.json")

    def set_status(self, status: str, **details: Any) -> None:
        self.metadata["status"] = status
        self.metadata["updated_at"] = _now().isoformat()
        if status == "completed":
            self.metadata["completed_at"] = _now().isoformat()
        if details:
            self.metadata.setdefault("status_details", {}).update(details)
        self._write_json_atomic(self.run_dir / "run.json", self.metadata)
