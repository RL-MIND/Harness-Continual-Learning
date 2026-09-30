from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict

from .config import load_config, resolve_project_path
from .environment import validate_data_dir
from .runner import ExperimentRunner
from .sequence import build_sequence


def _print(data: Dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))


def _local_model_report(model_config: Dict[str, Any], project_root: str) -> Dict[str, Any]:
    configured_path = str(model_config.get("model_path") or "")
    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = Path(project_root) / path
    path = path.resolve()
    config_present = (path / "config.json").is_file()
    tokenizer_present = (path / "tokenizer_config.json").is_file()
    weights_present = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
    return {
        "path": str(path),
        "config_present": config_present,
        "tokenizer_present": tokenizer_present,
        "weights_present": weights_present,
        "valid": bool(configured_path and config_present and tokenizer_present and weights_present),
    }


def command_doctor(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    data = validate_data_dir(resolve_project_path(config, "data_dir"))
    agent_config = dict(config.get("agent", {}))
    agent_provider = str(agent_config.get("provider", "openai")).lower()
    report = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "config": config["_config_path"],
        "data": data,
        "agent_provider": agent_provider,
    }
    models_valid = True
    local_provider_used = agent_provider == "transformers"
    if agent_provider == "transformers":
        report["agent_model"] = _local_model_report(
            agent_config, config["_project_root"]
        )
        models_valid = report["agent_model"]["valid"]
    else:
        api_key_env = str(agent_config.get("api_key_env") or "OPENAI_API_KEY")
        report["api_key_env"] = api_key_env
        report["api_key_present"] = bool(os.environ.get(api_key_env))
    if str(config.get("continual_method", "hcl")).lower() in {"rag"}:
        embedding_config = dict(config.get("embedding", {}))
        embedding_provider = str(embedding_config.get("provider", "openai")).lower()
        report["embedding_provider"] = embedding_provider
        if embedding_provider == "transformers":
            local_provider_used = True
            report["embedding_model"] = _local_model_report(
                embedding_config, config["_project_root"]
            )
            models_valid = models_valid and report["embedding_model"]["valid"]
    try:
        import textworld

        report["alfworld"] = importlib.metadata.version("alfworld")
        report["textworld"] = getattr(textworld, "__version__", "installed")
    except ImportError as exc:
        report["dependency_error"] = str(exc)
    if local_provider_used:
        local_dependencies = {}
        missing_dependencies = []
        for name in ("torch", "transformers", "accelerate"):
            try:
                local_dependencies[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                missing_dependencies.append(name)
        report["local_dependencies"] = local_dependencies
        if missing_dependencies:
            report["dependency_error"] = (
                "Missing local inference dependencies: " + ", ".join(missing_dependencies)
            )
    _print(report)
    return 0 if data["valid"] and models_valid and "dependency_error" not in report else 1


def command_build_sequence(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    sequence = config["sequence"]
    manifest = build_sequence(
        data_dir=resolve_project_path(config, "data_dir"),
        output_path=resolve_project_path(config, "sequence_manifest"),
        task_order=sequence["task_order"],
        train_episodes_per_task=int(sequence["train_episodes_per_task"]),
        eval_episodes_per_task=sequence["eval_episodes_per_task"],
        train_split=str(sequence["train_split"]),
        eval_split=str(sequence["eval_split"]),
        seed=int(config.get("seed", 42)),
    )
    _print(
        {
            "manifest": str(resolve_project_path(config, "sequence_manifest")),
            "phase_count": len(manifest["phases"]),
            "task_order": manifest["task_order"],
        }
    )
    return 0


def command_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.transport_max_retries is not None:
        if args.transport_max_retries < 0:
            raise ValueError("--transport-max-retries must be non-negative.")
        config.setdefault("agent", {})["transport_max_retries"] = int(
            args.transport_max_retries
        )
    if args.resume_disable_raw_memory_retrieval and not args.resume:
        raise ValueError(
            "--resume-disable-raw-memory-retrieval requires --resume; "
            "configs/hcl.yaml already disables raw-memory retrieval for a new run."
        )
    manifest = resolve_project_path(config, "sequence_manifest")
    if config.get("run_mode") != "standard_evaluation" and not manifest.exists():
        sequence = config["sequence"]
        build_sequence(
            resolve_project_path(config, "data_dir"),
            manifest,
            sequence["task_order"],
            int(sequence["train_episodes_per_task"]),
            sequence["eval_episodes_per_task"],
            str(sequence["train_split"]),
            str(sequence["eval_split"]),
            int(config.get("seed", 42)),
        )
    runner = ExperimentRunner(
        config,
        resume_dir=args.resume,
        raw_memory_retrieval_override=(
            False if args.resume_disable_raw_memory_retrieval else None
        ),
    )
    results = runner.run()
    _print(
        {
            "output_dir": str(runner.output_dir),
            "metrics": results["metrics"],
            "harness_version": results["harness_version"],
            "elapsed_seconds": results["elapsed_seconds"],
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Harness Continual Learning on ALFWorld")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler, help_text in (
        ("doctor", command_doctor, "verify Python, dependencies, config, and ALFWorld data"),
        ("build-sequence", command_build_sequence, "build a deterministic continual task stream"),
        ("run", command_run, "run learning and seen-task evaluation after every phase"),
    ):
        child = subparsers.add_parser(name, help=help_text)
        child.add_argument("--config", default="configs/hcl.yaml")
        if name == "run":
            child.add_argument(
                "--resume",
                default=None,
                help="resume a timestamped run directory from its last episode checkpoint",
            )
            child.add_argument(
                "--resume-disable-raw-memory-retrieval",
                action="store_true",
                help=(
                    "resume with raw trajectories stored but excluded from online routing; "
                    "records an intentional mid-run protocol change"
                ),
            )
            child.add_argument(
                "--transport-max-retries",
                type=int,
                default=None,
                help=(
                    "override transient transport retries; safe to change between "
                    "resume attempts"
                ),
            )
        child.set_defaults(handler=handler)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
