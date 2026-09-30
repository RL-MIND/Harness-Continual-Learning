from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from .dataset import DatasetLoader
from .evaluator import EvaluationRecorder, ExactMatchEvaluator
from .models import ChatCompletionsAPIModel, LocalTransformersModel, RuleBasedSmokeModel
from .memory import MemoryConfig
from .optimizer import Optimizer
from .pipeline import HCLPipeline, PipelineResult
from .progress import progress
from .router import RouterConfig
from .task_interface import TaskInterfaceConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal HCL CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run the configured HCL task flow")
    run.add_argument("--config", required=True, help="Path to config JSON")
    run.add_argument("--splits", default="train,test", help="Comma-separated flow names to run, e.g. train,test")
    run.add_argument("--limit-per-task", type=int, default=None, help="Maximum examples per task and split")
    run.add_argument("--method", choices=["hcl", "hcl_memory"], default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).resolve()
    config = _read_json(config_path)
    if args.method is not None:
        config["method"] = args.method
    base_dir = _config_base_dir(config_path)
    _run_lock_handle = _acquire_run_storage_lock(config, base_dir, config_path=config_path)
    model_config = config.get("model", {})
    backend = model_config.get("backend", "smoke") if isinstance(model_config, dict) else "smoke"
    progress(f"run start config={config_path} backend={backend} limit_per_task={args.limit_per_task}")
    _prepare_run_storage(config, base_dir)
    _prepare_checkpoint_resume(config, base_dir)
    model = _load_model(
        config.get("model"),
        base_dir=base_dir,
        storage_dir=config.get("storage_dir"),
    )
    judge_model = None
    if isinstance(config.get("judge_model"), dict):
        judge_model = _load_model(
            config.get("judge_model"),
            base_dir=base_dir,
            storage_dir=config.get("storage_dir"),
        )
    selection_model = None
    if isinstance(config.get("selection_model"), dict):
        if config.get("selection_model") == config.get("judge_model"):
            selection_model = judge_model
        else:
            selection_model = _load_model(
                config.get("selection_model"),
                base_dir=base_dir,
                storage_dir=config.get("storage_dir"),
            )
    memory_model = None
    if isinstance(config.get("memory_model"), dict):
        if config.get("memory_model") == config.get("selection_model"):
            memory_model = selection_model
        elif config.get("memory_model") == config.get("judge_model"):
            memory_model = judge_model
        else:
            memory_model = _load_model(
                config.get("memory_model"),
                base_dir=base_dir,
                storage_dir=config.get("storage_dir"),
            )
    pipeline = _build_pipeline(
        config,
        model=model,
        judge_model=judge_model,
        selection_model=selection_model,
        memory_model=memory_model,
        base_dir=base_dir,
    )
    selected_steps = {part.strip() for part in args.splits.split(",") if part.strip()}
    summaries: list[dict[str, Any]] = []
    for step in _select_flow_steps(config, selected_steps):
        progress(f"flow step start name={step['name']} phase={step['phase']}")
        result = _run_step(
            config,
            step=step,
            pipeline=pipeline,
            base_dir=base_dir,
            limit_per_task=args.limit_per_task,
        )
        summary = _summary(result, pipeline)
        summary["step"] = step["name"]
        summaries.append(summary)
        metrics = result.evaluation.get("metrics", {})
        progress(
            f"flow step done name={step['name']} "
            f"primary_metric={metrics.get('primary_metric_name')} "
            f"primary_score={float(metrics.get('primary_score') or 0.0):.4f} "
            f"harness_version={pipeline.harness_version}"
        )
    progress("run complete")
    print(json.dumps({"project_name": config.get("project_name", Path(__file__).resolve().parent.name), "runs": summaries}, ensure_ascii=False, indent=2))
    return 0


def _acquire_run_storage_lock(
    config: dict[str, Any],
    base_dir: Path,
    *,
    config_path: Path,
) -> Any | None:
    """Prevent two CLI processes from mutating the same experiment storage."""
    storage_dir = config.get("storage_dir")
    if not storage_dir:
        return None
    resolved_storage = _resolve_path(str(storage_dir), base_dir)
    # Keep the lock outside the directory that reset_run_on_start may delete.
    # A lock on an unlinked inode would not block a second process from
    # creating and locking a new file at the old path.
    lock_dir = resolved_storage.parent / ".hcl_run_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    storage_digest = hashlib.sha256(str(resolved_storage).encode("utf-8")).hexdigest()[:12]
    lock_path = lock_dir / f"{resolved_storage.name}.{storage_digest}.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "unknown owner"
        handle.close()
        raise RuntimeError(
            f"Experiment storage is already locked path={resolved_storage} owner={owner}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "config": str(config_path),
                "storage_dir": str(resolved_storage),
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    handle.flush()
    progress(f"run storage lock acquired path={lock_path} pid={os.getpid()}")
    return handle


def _build_pipeline(
    config: dict[str, Any],
    *,
    model: Any,
    judge_model: Any | None = None,
    selection_model: Any | None = None,
    memory_model: Any | None = None,
    base_dir: Path,
) -> HCLPipeline:
    task_interface_values = _resolve_paths_in_dict(config.get("task_interface", {}), base_dir)
    if "cache_path" not in task_interface_values and config.get("storage_dir"):
        task_interface_values["cache_path"] = str(
            _resolve_path(str(config["storage_dir"]), base_dir) / "task_interface" / "cache.jsonl"
        )
    task_interface_config = TaskInterfaceConfig(**task_interface_values)
    router_values = _resolve_paths_in_dict(config.get("router", {}), base_dir)
    if "memory_selector_cache_path" not in router_values and config.get("storage_dir"):
        router_values["memory_selector_cache_path"] = str(
            _resolve_path(str(config["storage_dir"]), base_dir) / "router" / "memory_selector_cache.jsonl"
        )
    if "workflow_selector_cache_path" not in router_values and config.get("storage_dir"):
        router_values["workflow_selector_cache_path"] = str(
            _resolve_path(str(config["storage_dir"]), base_dir) / "router" / "workflow_selector_cache.jsonl"
        )
    if "skill_selector_cache_path" not in router_values and config.get("storage_dir"):
        router_values["skill_selector_cache_path"] = str(
            _resolve_path(str(config["storage_dir"]), base_dir) / "router" / "skill_selector_cache.jsonl"
        )
    if "tool_selector_cache_path" not in router_values and config.get("storage_dir"):
        router_values["tool_selector_cache_path"] = str(
            _resolve_path(str(config["storage_dir"]), base_dir) / "router" / "tool_selector_cache.jsonl"
        )
    if "tool_argument_cache_path" not in router_values and config.get("storage_dir"):
        router_values["tool_argument_cache_path"] = str(
            _resolve_path(str(config["storage_dir"]), base_dir) / "router" / "tool_argument_cache.jsonl"
        )
    if "final_generation_cache_path" not in router_values and config.get("storage_dir"):
        router_values["final_generation_cache_path"] = str(
            _resolve_path(str(config["storage_dir"]), base_dir) / "router" / "final_generation_cache.jsonl"
        )
    router_config = RouterConfig(**router_values)
    optimizer_config = _component_config_with_storage_default(config, "optimizer", base_dir)
    evaluator_config = _component_config_with_storage_default(config, "evaluator", base_dir)
    memory_config = _component_config_with_storage_default(config, "memory", base_dir)
    prediction_cache_config = _prediction_cache_config(config, base_dir)
    execution_config = config.get("execution", {})
    if not isinstance(execution_config, dict):
        raise ValueError("execution config must be a JSON object")
    method = str(config.get("method", "hcl")).lower()
    if method not in {"hcl", "hcl_memory"}:
        raise ValueError(f"Unsupported method: {method}. Expected 'hcl' or 'hcl_memory'.")
    if method == "hcl_memory":
        optimizer_config["components"] = []
    return HCLPipeline(
        model,
        judge_model=judge_model,
        selection_model=selection_model,
        memory_model=memory_model,
        judge_memory=bool(config.get("judge_memory", False)),
        task_interface_config=task_interface_config,
        router_config=router_config,
        evaluator=ExactMatchEvaluator(),
        evaluation_recorder=EvaluationRecorder(**evaluator_config),
        optimizer=Optimizer(**optimizer_config),
        memory_config=MemoryConfig(**memory_config),
        forgetting_probe_config=config.get("forgetting_probe"),
        prediction_cache_config=prediction_cache_config,
        checkpoint_path=_checkpoint_path(config, base_dir),
        resume_from_checkpoint=_checkpoint_resume_enabled(config),
        read_only_workers=int(execution_config.get("read_only_workers", 1)),
        run_train_final=bool(execution_config.get("run_train_final", True)),
        method=method,
    )


def _select_flow_steps(config: dict[str, Any], selected_steps: set[str]) -> list[dict[str, Any]]:
    raw_flow = config.get("task_flow")
    if isinstance(raw_flow, list) and raw_flow:
        flow = [_normalize_flow_step(step) for step in raw_flow if isinstance(step, dict)]
    else:
        flow = [
            {"name": "train", "phase": "train", "train_split": "train", "val_split": "validation"},
            {"name": "test", "phase": "test", "test_split": "test"},
        ]
    steps = [step for step in flow if step["name"] in selected_steps]
    if not steps:
        raise ValueError(f"No matching task_flow steps for --splits={','.join(sorted(selected_steps))}")
    return steps


def _normalize_flow_step(step: dict[str, Any]) -> dict[str, Any]:
    name = str(step.get("name") or step.get("phase") or "")
    phase = str(step.get("phase") or name)
    if phase not in {"train", "test"}:
        raise ValueError("task_flow step phase must be one of: train, test")
    normalized = {
        "name": name or phase,
        "phase": phase,
    }
    if phase == "train":
        normalized["train_split"] = str(step.get("train_split") or "train")
        normalized["val_split"] = str(step.get("val_split") or "validation")
        normalized["test_split"] = str(step.get("test_split") or "test")
        normalized["continual"] = bool(step.get("continual", False))
        if step.get("batchsize") is not None:
            normalized["batchsize"] = int(step["batchsize"])
        for key in ("train_limit_per_task", "validation_limit_per_task", "test_limit_per_task"):
            if step.get(key) is not None:
                normalized[key] = int(step[key])
    else:
        normalized["test_split"] = str(step.get("test_split") or "test")
        normalized["group_by_task"] = bool(step.get("group_by_task", False))
        if step.get("test_limit_per_task") is not None:
            normalized["test_limit_per_task"] = int(step["test_limit_per_task"])
        if step.get("reference_predictions"):
            normalized["reference_predictions"] = step["reference_predictions"]
    return normalized


def _run_step(
    config: dict[str, Any],
    *,
    step: dict[str, Any],
    pipeline: HCLPipeline,
    base_dir: Path,
    limit_per_task: int | None,
) -> Any:
    task_stream_path = _resolve_path(str(config["task_stream_path"]), base_dir)
    loader = DatasetLoader()
    train_limit = _split_limit(step, config, "train_limit_per_task", limit_per_task)
    validation_limit = _split_limit(step, config, "validation_limit_per_task", limit_per_task)
    test_limit = _split_limit(step, config, "test_limit_per_task", limit_per_task)
    progress(
        f"split limits train={train_limit} validation={validation_limit} test={test_limit}"
    )
    if step["phase"] == "train":
        if step.get("continual"):
            return pipeline.continual_train(
                task_stream_path,
                train_split=str(step["train_split"]),
                val_split=str(step["val_split"]),
                test_split=str(step["test_split"]),
                limit_per_task=limit_per_task,
                train_limit_per_task=train_limit,
                validation_limit_per_task=validation_limit,
                test_limit_per_task=test_limit,
                batchsize=int(step.get("batchsize", config.get("batchsize", 0)) or 0) or None,
            )
        train_examples = loader.load_task_stream(task_stream_path, split=str(step["train_split"]), limit_per_task=train_limit)
        val_examples = loader.load_task_stream(task_stream_path, split=str(step["val_split"]), limit_per_task=validation_limit)
        return pipeline.train(
            train_examples,
            val_data=val_examples,
            split=str(step["train_split"]),
            batchsize=int(step.get("batchsize", config.get("batchsize", 0)) or 0) or None,
        )
    test_examples = loader.load_task_stream(task_stream_path, split=str(step["test_split"]), limit_per_task=test_limit)
    reference_predictions = _load_optional_json_list(step.get("reference_predictions"), base_dir)
    if step.get("group_by_task"):
        return _run_grouped_test_step(
            pipeline,
            test_examples,
            split=str(step["test_split"]),
            reference_predictions=reference_predictions,
        )
    return pipeline.test(
        test_examples,
        split=str(step["test_split"]),
        reference_predictions=reference_predictions,
    )


def _run_grouped_test_step(
    pipeline: HCLPipeline,
    test_examples: list[dict[str, Any]],
    *,
    split: str,
    reference_predictions: list[dict[str, object]] | None = None,
) -> PipelineResult:
    grouped_examples: dict[str, list[dict[str, Any]]] = {}
    for example in test_examples:
        task_name = str(example.get("task_name") or "unknown_task")
        grouped_examples.setdefault(task_name, []).append(example)

    grouped_results: dict[str, dict[str, object]] = {}
    combined_predictions: list[dict[str, object]] = []
    reference_by_id = {
        str(prediction.get("task_id", "")): prediction
        for prediction in (reference_predictions or [])
        if isinstance(prediction, dict)
    }
    for task_name, task_examples in grouped_examples.items():
        progress(f"grouped test start task={task_name} examples={len(task_examples)}")
        task_reference_predictions = None
        if reference_predictions is not None:
            task_ids = {str(example.get("task_id", "")) for example in task_examples}
            task_reference_predictions = [
                prediction
                for task_id, prediction in reference_by_id.items()
                if task_id in task_ids
            ]
        task_result = pipeline.test(
            task_examples,
            split=split,
            reference_predictions=task_reference_predictions,
            phase="test",
            current_task=task_name,
            tested_task=task_name,
        )
        grouped_results[task_name] = dict(task_result.evaluation)
        combined_predictions.extend(
            dict(prediction)
            for prediction in task_result.evaluation.get("predictions", [])
            if isinstance(prediction, dict)
        )
        metrics = task_result.evaluation.get("metrics", {})
        progress(
            f"grouped test done task={task_name} "
            f"primary_metric={metrics.get('primary_metric_name')} "
            f"primary_score={float(metrics.get('primary_score') or 0.0):.4f} "
            f"correct={metrics.get('correct', 0)} total={metrics.get('total', 0)}"
        )

    combined_result = pipeline.evaluator.evaluate(
        test_examples,
        combined_predictions,
        split=split,
        reference_predictions=reference_predictions,
    )
    combined_result["by_task"] = {
        task_name: result.get("metrics", {})
        for task_name, result in grouped_results.items()
    }
    return PipelineResult(
        evaluation=combined_result,
        task_interface_config=pipeline.task_interface_config,
        router_config=pipeline.router_config,
        optimizer_history=list(pipeline.optimizer.history),
        memory_summary=pipeline.memory.summary(),
    )


def _split_limit(
    step: dict[str, Any],
    config: dict[str, Any],
    key: str,
    fallback: int | None,
) -> int | None:
    value = step.get(key, config.get(key, fallback))
    if value is None:
        return None
    return max(int(value), 0)


def _load_model(model_config: object, *, base_dir: Path | None = None, storage_dir: object = None) -> Any:
    if not isinstance(model_config, dict):
        return RuleBasedSmokeModel()
    model_config = _normalize_model_config_paths(model_config, base_dir=base_dir, storage_dir=storage_dir)
    class_path = str(model_config.get("class_path") or "")
    if class_path:
        kwargs = {key: value for key, value in model_config.items() if key not in {"class_path", "kwargs"}}
        extra_kwargs = model_config.get("kwargs") or {}
        if not isinstance(extra_kwargs, dict):
            raise ValueError("model.kwargs must be a JSON object.")
        kwargs.update(extra_kwargs)
        return _load_object(class_path)(**kwargs)
    backend = str(model_config.get("backend") or model_config.get("provider") or model_config.get("name") or "").lower()
    if backend == "smoke":
        return RuleBasedSmokeModel()
    if backend in {
        "local",
        "local_transformers",
        "local-transformers",
        "transformers",
        "hf",
        "huggingface",
        "qwen",
        "qwen_vl",
        "qwen-vl",
    }:
        return LocalTransformersModel(**model_config)
    if backend in {
        "api",
        "chat_api",
        "chat-api",
        "chat_completions",
        "chat-completions",
        "openai_compatible",
        "openai-compatible",
        "deepseek",
        "ds",
        "deepseek-api",
        "deepseek_api",
    }:
        return ChatCompletionsAPIModel(**model_config)
    raise ValueError(
        "model.backend must be smoke/local_transformers/chat_completions, "
        "or model.class_path must be provided."
    )



def _normalize_model_config_paths(model_config: dict[str, Any], *, base_dir: Path | None, storage_dir: object) -> dict[str, Any]:
    normalized = dict(model_config)
    if base_dir is not None:
        if isinstance(normalized.get("log_path"), str):
            normalized["log_path"] = str(_resolve_path(str(normalized["log_path"]), base_dir))
        elif "log_path" not in normalized and storage_dir:
            normalized["log_path"] = str(_resolve_path(str(storage_dir), base_dir) / "llm_calls.txt")
    return normalized
def _load_object(class_path: str) -> Any:
    if ":" in class_path:
        module_name, object_name = class_path.split(":", 1)
    else:
        module_name, object_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def _summary(result: Any, pipeline: HCLPipeline) -> dict[str, Any]:
    return {
        "harness_version": pipeline.harness_version,
        "evaluation_metrics": result.evaluation.get("metrics", {}),
        "current_validation_metrics": result.current_validation.get("metrics", {}) if result.current_validation else None,
        "candidate_metrics": {
            candidate_id: evaluation.get("metrics", {})
            for candidate_id, evaluation in result.candidate_evaluations.items()
        },
        "accepted_candidate": result.accepted_candidate,
        "memory": result.memory_summary or pipeline.memory.summary(),
        "continual_matrix": result.continual_matrix,
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    if "task_stream_path" not in data:
        raise ValueError("Config must include task_stream_path.")
    return data


def _load_optional_json_list(path_value: object, base_dir: Path) -> list[dict[str, object]] | None:
    if not path_value:
        return None
    path = _resolve_path(str(path_value), base_dir)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Reference predictions must be a JSON list: {path}")
    return [dict(item) for item in data if isinstance(item, dict)]


def _component_config_with_storage_default(config: dict[str, Any], component: str, base_dir: Path) -> dict[str, Any]:
    component_config = _resolve_paths_in_dict(config.get(component, {}), base_dir)
    if "record_dir" not in component_config and config.get("storage_dir"):
        component_config["record_dir"] = str(_resolve_path(str(config["storage_dir"]), base_dir) / component)
    return component_config


def _prediction_cache_config(config: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    raw = config.get("prediction_cache")
    values = _resolve_paths_in_dict(raw, base_dir) if isinstance(raw, dict) else {}
    if "enabled" not in values:
        values["enabled"] = bool(config.get("storage_dir"))
    if "path" not in values and config.get("storage_dir"):
        values["path"] = str(_resolve_path(str(config["storage_dir"]), base_dir) / "prediction_cache" / "read_only_predictions.jsonl")
    return values


def _resolve_paths_in_dict(raw: object, base_dir: Path) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Component config must be a JSON object.")
    path_keys = {
        "record_dir",
        "structuring_template_path",
        "workflow_template_path",
        "context_template_path",
        "memory_selector_template_path",
        "skill_selector_template_path",
        "tool_selector_template_path",
        "tool_argument_template_path",
        "capability_workflow_template_path",
        "capability_context_template_path",
        "memory_selector_cache_path",
        "workflow_selector_cache_path",
        "skill_selector_cache_path",
        "tool_selector_cache_path",
        "tool_argument_cache_path",
        "final_generation_cache_path",
        "candidate_generation_template_path",
        "cache_path",
        "path",
        "skill_dir",
        "tool_js_dir",
    }
    resolved: dict[str, Any] = {}
    for key, value in raw.items():
        if key in path_keys and isinstance(value, str):
            resolved[key] = str(_resolve_path(value, base_dir))
        else:
            resolved[key] = value
    return resolved


def _config_base_dir(config_path: Path) -> Path:
    return config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _prepare_run_storage(config: dict[str, Any], base_dir: Path) -> None:
    if not config.get("reset_run_on_start", False) or not config.get("storage_dir"):
        return
    checkpoint_path = _checkpoint_path(config, base_dir)
    if checkpoint_path is not None and _checkpoint_resume_enabled(config) and checkpoint_path.exists():
        progress(f"checkpoint resume detected; preserving run storage path={checkpoint_path}")
        return
    storage_root = (base_dir / "storage").resolve()
    run_dir = _resolve_path(str(config["storage_dir"]), base_dir)
    if run_dir == storage_root or storage_root not in run_dir.parents:
        raise ValueError("reset_run_on_start only permits a named run directory under this project's storage/ directory")
    if run_dir.exists():
        progress(f"reset run storage path={run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def _prepare_checkpoint_resume(config: dict[str, Any], base_dir: Path) -> None:
    checkpoint_path = _checkpoint_path(config, base_dir)
    if checkpoint_path is None or not _checkpoint_resume_enabled(config) or not checkpoint_path.exists():
        return
    memory_config = config.get("memory")
    if isinstance(memory_config, dict):
        memory_config["reset_on_start"] = False
    progress(f"checkpoint resume enabled path={checkpoint_path}")


def _checkpoint_path(config: dict[str, Any], base_dir: Path) -> Path | None:
    checkpoint_config = config.get("checkpoint")
    if not isinstance(checkpoint_config, dict) or not checkpoint_config.get("enabled", False):
        return None
    checkpoint_path = checkpoint_config.get("path")
    if checkpoint_path:
        return _resolve_path(str(checkpoint_path), base_dir)
    storage_dir = config.get("storage_dir")
    if not storage_dir:
        return None
    return _resolve_path(str(storage_dir), base_dir) / "checkpoint.json"


def _checkpoint_resume_enabled(config: dict[str, Any]) -> bool:
    checkpoint_config = config.get("checkpoint")
    if not isinstance(checkpoint_config, dict) or not checkpoint_config.get("enabled", False):
        return False
    return bool(checkpoint_config.get("resume", True))


if __name__ == "__main__":
    raise SystemExit(main())



