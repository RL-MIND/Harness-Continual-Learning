from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .json_utils import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a resumable HCL ablation matrix sequentially")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    matrix_path = Path(args.config).resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    run_root = Path(str(matrix["run_root"])).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    state_dir = run_root / "matrix_state"
    log_dir = run_root / "run_logs"
    resolved_dir = run_root / "resolved_configs"
    for path in (state_dir, log_dir, resolved_dir):
        path.mkdir(parents=True, exist_ok=True)
    atomic_write_json(state_dir / "matrix.status.json", {"state": "running", "pid": os.getpid()})

    base = matrix["base_config"]
    variants = matrix["variants"]
    for index, variant in enumerate(variants):
        name = str(variant["name"])
        variant_state = state_dir / f"{name}.status.json"
        completed_marker = state_dir / f"{name}.completed.json"
        if completed_marker.exists():
            print(f"[ablation-matrix] skip completed variant={name}", flush=True)
            continue
        config = copy.deepcopy(base)
        config["project_name"] = f"HCL ablation: {name}"
        config["optimizer"]["components"] = list(variant["optimizer_components"])
        _deep_merge(config["router"], variant.get("router_overrides", {}))
        _deep_merge(config["memory"], variant.get("memory_overrides", {}))
        storage = run_root / name
        config["storage_dir"] = str(storage)
        if config["model"].get("log_path", "auto") is not None:
            config["model"]["log_path"] = str(storage / "llm_calls.txt")
        config["router"]["route_name"] = f"hcl_ablation_{name}"
        config["prediction_cache"] = {
            "enabled": True,
            "path": str(storage / "prediction_cache" / "read_only_predictions.jsonl"),
        }
        config["checkpoint"] = {
            "enabled": True,
            "path": str(storage / "checkpoint.json"),
            "resume": True,
        }
        resolved_path = resolved_dir / f"{index:02d}_{name}.json"
        atomic_write_json(resolved_path, config)
        manifest = {
            "variant": name,
            "optimizer_components": config["optimizer"]["components"],
            "memory_enabled": config["memory"].get("enabled"),
            "memory_selector_runtime_enabled": config["router"].get("use_memory_selector"),
            "external_capability_enabled": config["router"].get("capability_enabled"),
            "semantic_search_enabled": config["router"].get("semantic_search_enabled"),
            "cross_modal_match_enabled": config["router"].get("cross_modal_match_enabled"),
            "resolved_config": str(resolved_path),
        }
        atomic_write_json(storage / "experiment_manifest.json", manifest)
        log_path = log_dir / f"{index:02d}_{name}.log"
        atomic_write_json(variant_state, {
            "state": "running", "pid": os.getpid(), "variant": name,
            "config": str(resolved_path), "log": str(log_path),
        })
        print(f"[ablation-matrix] start {index + 1}/{len(variants)} variant={name}", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            project_root = Path(__file__).resolve().parent.parent
            process = subprocess.run(
                [
                    sys.executable, "-u", "-m", "hcl.cli",
                    "run", "--config", str(resolved_path), "--splits", "train,test",
                ],
                cwd=project_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "PYTHONPATH": os.pathsep.join(
                        part
                        for part in (str(project_root), os.environ.get("PYTHONPATH", ""))
                        if part
                    ),
                },
                check=False,
            )
        if process.returncode != 0:
            failure = {"state": "failed", "variant": name, "exit_code": process.returncode, "log": str(log_path)}
            atomic_write_json(variant_state, failure)
            atomic_write_json(state_dir / "matrix.status.json", failure)
            return process.returncode
        success = {"state": "completed", "variant": name, "exit_code": 0, "log": str(log_path)}
        atomic_write_json(variant_state, success)
        atomic_write_json(completed_marker, success)
    atomic_write_json(state_dir / "matrix.status.json", {"state": "completed", "pid": os.getpid()})
    return 0


def _deep_merge(target: dict[str, Any], override: object) -> None:
    if not isinstance(override, dict):
        return
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


if __name__ == "__main__":
    raise SystemExit(main())
