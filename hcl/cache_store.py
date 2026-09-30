from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any, Callable

from .json_utils import json_clone


SchemaError = Callable[[dict[str, Any]], object | None]


class JsonObjectCache:
    """Append-only JSONL cache for dict values keyed by ``cache_key``."""

    def __init__(
        self,
        path: str | Path | None,
        *,
        value_field: str,
        schema_error: SchemaError | None = None,
    ) -> None:
        self.path = Path(path) if path else None
        self.value_field = value_field
        self.schema_error = schema_error
        self.entries: dict[str, dict[str, Any]] = {}
        self._lock = RLock()
        if self.path is None or not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                cache_key = str(row.get("cache_key", ""))
                value = row.get(self.value_field)
                if not cache_key or not isinstance(value, dict):
                    continue
                if self.schema_error is not None and self.schema_error(value) is not None:
                    continue
                self.entries[cache_key] = dict(value)

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            value = self.entries.get(key)
            return json_clone(value) if value is not None else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            if self.path is None or key in self.entries:
                return
            if self.schema_error is not None and self.schema_error(value) is not None:
                return
            stored = json_clone(value)
            self.entries[key] = stored
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"cache_key": key, self.value_field: stored}, ensure_ascii=False) + "\n")


class TextValueCache:
    """Append-only JSONL cache for non-empty string values keyed by ``cache_key``."""

    def __init__(self, path: str | Path | None, *, value_field: str) -> None:
        self.path = Path(path) if path else None
        self.value_field = value_field
        self.entries: dict[str, str] = {}
        self._lock = RLock()
        if self.path is None or not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                cache_key = str(row.get("cache_key", ""))
                value = row.get(self.value_field)
                if not cache_key or not isinstance(value, str) or value == "":
                    continue
                self.entries[cache_key] = value

    def get(self, key: str) -> str | None:
        with self._lock:
            return self.entries.get(key)

    def put(self, key: str, value: str) -> None:
        with self._lock:
            if self.path is None or key in self.entries or not value:
                return
            self.entries[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"cache_key": key, self.value_field: value}, ensure_ascii=False) + "\n")
