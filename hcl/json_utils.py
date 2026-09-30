from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_json_hash(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Atomically replace one JSON document in its destination directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Atomically replace a JSONL snapshot without publishing a partial line."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
