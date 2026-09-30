from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    parent = config.pop("extends", None)
    if parent:
        parent_config = load_config(path.parent / parent)
        config = _deep_merge(parent_config, config)
    root = path.parent
    for candidate in (path.parent, *path.parents):
        if (candidate / "pyproject.toml").exists():
            root = candidate
            break
    config["_config_path"] = str(path)
    config["_project_root"] = str(root.resolve())
    return config


def resolve_project_path(config: Dict[str, Any], key: str) -> Path:
    value = Path(str(config[key]))
    if value.is_absolute():
        return value
    return Path(config["_project_root"]) / value
