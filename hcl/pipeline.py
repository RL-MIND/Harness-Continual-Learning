from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
from threading import RLock
from time import perf_counter
from typing import Iterable, Any

from .dataset import DatasetLoader, load_task_stream_specs
from .evaluator import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationRecorder,
    ExactMatchEvaluator,
    answer_matches,
    compact_metrics,
)
from .json_utils import atomic_write_json
from .json_utils import json_clone as _json_clone
from .json_utils import stable_json_hash as _stable_json_hash
from .memory import ExperienceMemory, MemoryConfig
from .optimizer import Optimizer
from .progress import progress, report_item
from .router import Model, Router, RouterConfig
from .task_interface import TaskInterface, TaskInterfaceConfig


@dataclass
class PipelineResult:
    evaluation: dict[str, object]
    task_interface_config: TaskInterfaceConfig
    router_config: RouterConfig
    generated_candidates: list[dict[str, object]] = field(default_factory=list)
    candidate_evaluations: dict[str, dict[str, object]] = field(default_factory=dict)
    current_validation: dict[str, object] | None = None
    accepted_candidate: dict[str, object] | None = None
    optimizer_history: list[dict[str, object]] = field(default_factory=list)
    continual_matrix: dict[str, object] | None = None
    memory_summary: dict[str, int] = field(default_factory=dict)


@dataclass
class ForgettingProbeConfig:
    enabled: bool = True
    interval_batches: int = 2
    samples_per_task: int = 100
    target_correct_ratio: float = 0.8
    seed: int = 20260713
    split: str = "test"
    historical_tasks_only: bool = True
    reuse_probe_set: bool = True
    record_predictions: bool = False


@dataclass
class PredictionCacheConfig:
    enabled: bool = False
    path: str | Path | None = None


PROJECT_ROOT = Path(__file__).resolve().parent

OPTIMIZABLE_COMPONENTS = (
    "task_interface_structuring",
    "skill_selector",
    "router_workflow",
    "memory_selector",
    "tool_selector",
    "router_context",
)


def _relocate_component_version_paths(
    component_version: dict[str, object],
    checkpoint_path: Path | None,
) -> dict[str, object]:
    artifact_path = component_version.get("artifact_path")
    if isinstance(artifact_path, str) and artifact_path:
        component_version["artifact_path"] = _relocate_checkpoint_path(artifact_path, checkpoint_path)
    return component_version


def _relocate_checkpoint_path(raw_path: str, checkpoint_path: Path | None) -> str:
    path = Path(raw_path)
    checkpoint_dir = checkpoint_path.parent if checkpoint_path is not None else None
    if checkpoint_dir is not None:
        storage_root = checkpoint_dir.parent
        parts = path.parts
        if checkpoint_dir.name in parts:
            index = parts.index(checkpoint_dir.name)
            candidate = checkpoint_dir.joinpath(*parts[index + 1 :])
            if candidate.exists():
                return str(candidate)
        if storage_root.name in parts:
            index = parts.index(storage_root.name)
            candidate = storage_root.joinpath(*parts[index + 1 :])
            if candidate.exists():
                return str(candidate)

    known_suffixes = (
        ("task_interface", "structuring_prompt_default.json"),
        ("router", "workflow_prompt_default.json"),
        ("router", "context_prompt_default.json"),
        ("router", "memory_selector_prompt_default.json"),
        ("router", "skill_selector_prompt_default.json"),
        ("router", "tool_selector_prompt_default.json"),
        ("optimizer", "candidate_generation_prompt_default.json"),
    )
    parts_tail = path.parts[-2:]
    for suffix in known_suffixes:
        if tuple(parts_tail) == suffix:
            candidate = PROJECT_ROOT.joinpath(*suffix)
            if candidate.exists():
                return str(candidate)
    if path.exists():
        return str(path)
    return raw_path


class PipelineCheckpoint:
    """Small JSON checkpoint for resumable continual training.

    The checkpoint deliberately records only safe restart boundaries. If a run
    stops inside one batch/test pass, that unit is repeated on resume; completed
    batches and stages are skipped.
    """

    def __init__(self, path: str | Path | None, *, resume: bool = True) -> None:
        self.path = Path(path) if path else None
        self.resume = bool(resume)
        self.state: dict[str, Any] = {}
        if self.path is not None and self.resume and self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                loaded = {}
            if isinstance(loaded, dict):
                self.state = loaded

    @property
    def enabled(self) -> bool:
        return self.path is not None

    @property
    def has_state(self) -> bool:
        return bool(self.state)

    def restore_pipeline(self, pipeline: "HCLPipeline") -> None:
        memory_snapshot = self.state.get("memory_snapshot")
        if isinstance(memory_snapshot, dict):
            pipeline.memory.restore(memory_snapshot)
            progress(
                f"checkpoint memory restored path={self.path} summary={pipeline.memory.summary()}"
            )
        artifacts = self.state.get("artifacts")
        if not isinstance(artifacts, dict):
            return
        try:
            pipeline.harness_version = int(artifacts.get("harness_version", pipeline.harness_version))
        except (TypeError, ValueError):
            pass
        try:
            pipeline.global_batch_index = int(self.state.get("global_batch_index", pipeline.global_batch_index))
        except (TypeError, ValueError):
            pass
        component_versions = self.state.get("component_versions", artifacts.get("component_versions"))
        if isinstance(component_versions, dict):
            pipeline.component_versions.update(
                {
                    str(component): _relocate_component_version_paths(dict(value), self.path)
                    for component, value in component_versions.items()
                    if isinstance(value, dict)
                }
            )
        forgetting_probe = self.state.get("forgetting_probe")
        if isinstance(forgetting_probe, dict):
            probe_sets = forgetting_probe.get("probe_sets")
            if isinstance(probe_sets, dict):
                pipeline.historical_task_probe_sets.update(
                    {
                        str(task_name): dict(value)
                        for task_name, value in probe_sets.items()
                        if isinstance(value, dict)
                    }
                )
            reference_predictions = forgetting_probe.get("reference_predictions")
            if isinstance(reference_predictions, dict):
                pipeline.historical_task_reference_predictions.update(
                    {
                        str(task_name): list(value)
                        for task_name, value in reference_predictions.items()
                        if isinstance(value, list)
                    }
                )
        paths = artifacts.get("paths")
        if not isinstance(paths, dict):
            return
        if paths.get("task_interface_structuring"):
            pipeline.task_interface_config.structuring_template_path = _relocate_checkpoint_path(
                str(paths["task_interface_structuring"]),
                self.path,
            )
        if paths.get("router_workflow"):
            pipeline.router_config.workflow_template_path = _relocate_checkpoint_path(
                str(paths["router_workflow"]),
                self.path,
            )
        if paths.get("router_context"):
            pipeline.router_config.context_template_path = _relocate_checkpoint_path(
                str(paths["router_context"]),
                self.path,
            )
        if paths.get("memory_selector"):
            pipeline.router_config.memory_selector_template_path = _relocate_checkpoint_path(
                str(paths["memory_selector"]),
                self.path,
            )
        if paths.get("skill_selector"):
            pipeline.router_config.skill_selector_template_path = _relocate_checkpoint_path(
                str(paths["skill_selector"]),
                self.path,
            )
        if paths.get("tool_selector"):
            pipeline.router_config.tool_selector_template_path = _relocate_checkpoint_path(
                str(paths["tool_selector"]),
                self.path,
            )
        if paths.get("capability_workflow"):
            pipeline.router_config.capability_workflow_template_path = _relocate_checkpoint_path(
                str(paths["capability_workflow"]),
                self.path,
            )
        if paths.get("capability_context"):
            pipeline.router_config.capability_context_template_path = _relocate_checkpoint_path(
                str(paths["capability_context"]),
                self.path,
            )
        progress(
            f"checkpoint restored path={self.path} harness_version={pipeline.harness_version}"
        )

    def save(self, pipeline: "HCLPipeline", update: dict[str, Any]) -> None:
        if self.path is None:
            return
        state = _json_clone(self.state)
        _merge_dict(state, update)
        state["version"] = 1
        state["artifacts"] = pipeline._checkpoint_artifacts()
        state["global_batch_index"] = pipeline.global_batch_index
        state["component_versions"] = pipeline._component_version_snapshot()
        state["forgetting_probe"] = {
            "probe_sets": pipeline.historical_task_probe_sets,
            "reference_predictions": pipeline.historical_task_reference_predictions,
        }
        state["memory_snapshot"] = pipeline.memory.snapshot()
        atomic_write_json(self.path, state)
        self.state = state


class PredictionCache:
    """Append-only cache for read-only predictions.

    Only the model answer is persisted. Runtime identifiers are materialized
    from the current example on cache hit, so the cache cannot replay a stale
    internal task_id into evaluation or traces.
    """

    def __init__(self, config: PredictionCacheConfig) -> None:
        self.enabled = bool(config.enabled and config.path)
        self.path = Path(config.path) if config.path else None
        self.entries: dict[str, dict[str, Any]] = {}
        self._lock = RLock()
        if not self.enabled or self.path is None or not self.path.exists():
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
                answer = row.get("output", row.get("answer"))
                if cache_key and answer is not None:
                    self.entries[cache_key] = {"answer": str(answer)}

    def get(self, key: str, *, task_id: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        with self._lock:
            value = self.entries.get(key)
            if value is None:
                return None
            answer = str(value.get("answer", ""))
        return {
            "task_id": task_id,
            "answer": answer,
            "trace": {
                "task_id": task_id,
                "cache": "prediction_cache_hit",
                "cache_key": key[:12],
                "raw_output": answer,
                "final_answer": answer,
                "selected_memory": [],
                "selected_skills": [],
                "selected_tools": [],
            },
        }

    def put(self, key: str, prediction: dict[str, Any], *, phase: str) -> None:
        with self._lock:
            if not self.enabled or self.path is None or key in self.entries:
                return
            answer = prediction.get("answer")
            if answer is None:
                return
            stored = {"answer": str(answer)}
            self.entries[key] = stored
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "cache_key": key,
                            "phase_hash": _stable_json_hash(phase)[:12],
                            "output": stored["answer"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


class HCLPipeline:
    def __init__(
        self,
        model: Model,
        *,
        judge_model: Model | None = None,
        selection_model: Model | None = None,
        memory_model: Model | None = None,
        judge_memory: bool = False,
        task_interface_config: TaskInterfaceConfig | None = None,
        router_config: RouterConfig | None = None,
        evaluator: ExactMatchEvaluator | None = None,
        evaluation_recorder: EvaluationRecorder | None = None,
        optimizer: Optimizer | None = None,
        dataset_loader: DatasetLoader | None = None,
        memory: ExperienceMemory | None = None,
        memory_config: MemoryConfig | None = None,
        forgetting_probe_config: ForgettingProbeConfig | dict[str, Any] | None = None,
        prediction_cache_config: PredictionCacheConfig | dict[str, Any] | None = None,
        checkpoint_path: str | Path | None = None,
        resume_from_checkpoint: bool = True,
        read_only_workers: int = 1,
        run_train_final: bool = True,
        method: str = "hcl",
    ) -> None:
        self.model = model
        self.judge_model = judge_model or model
        self.selection_model = selection_model or model
        self.judge_memory = bool(judge_memory)
        self.task_interface_config = task_interface_config or TaskInterfaceConfig()
        self.router_config = router_config or RouterConfig()
        self.evaluator = evaluator or ExactMatchEvaluator()
        self.evaluation_recorder = evaluation_recorder or EvaluationRecorder()
        self.optimizer = optimizer or Optimizer()
        self.dataset_loader = dataset_loader or DatasetLoader()
        self.method = str(method or "hcl").lower()
        resolved_memory_model = memory_model or (self.judge_model if self.judge_memory else model)
        self.memory_model = resolved_memory_model
        self.memory = memory or ExperienceMemory(memory_config, model=resolved_memory_model)
        self.harness_version = 0
        self.global_batch_index = 0
        self.read_only_workers = max(int(read_only_workers), 1)
        self.run_train_final = bool(run_train_final)
        if isinstance(forgetting_probe_config, ForgettingProbeConfig):
            self.forgetting_probe = forgetting_probe_config
        elif isinstance(forgetting_probe_config, dict):
            self.forgetting_probe = ForgettingProbeConfig(**forgetting_probe_config)
        else:
            self.forgetting_probe = ForgettingProbeConfig()
        if isinstance(prediction_cache_config, PredictionCacheConfig):
            prediction_cache_values = prediction_cache_config
        elif isinstance(prediction_cache_config, dict):
            prediction_cache_values = PredictionCacheConfig(**prediction_cache_config)
        else:
            prediction_cache_values = PredictionCacheConfig()
        self.prediction_cache = PredictionCache(prediction_cache_values)
        self.component_versions: dict[str, dict[str, object]] = {
            component: {"version": 0}
            for component in OPTIMIZABLE_COMPONENTS
        }
        self.historical_task_probe_sets: dict[str, dict[str, object]] = {}
        self.historical_task_reference_predictions: dict[str, list[dict[str, object]]] = {}
        self.historical_task_test_sets: dict[str, list[dict[str, Any]]] = {}
        self._active_continual_context: dict[str, Any] | None = None
        self.checkpoint = PipelineCheckpoint(checkpoint_path, resume=resume_from_checkpoint)
        if self.checkpoint.has_state:
            self.checkpoint.restore_pipeline(self)

    def _checkpoint_artifacts(self) -> dict[str, object]:
        paths = {
            "task_interface_structuring": str(self.task_interface_config.structuring_template_path),
            "router_workflow": str(self.router_config.workflow_template_path),
            "router_context": str(self.router_config.context_template_path),
            "memory_selector": str(self.router_config.memory_selector_template_path),
            "skill_selector": str(self.router_config.skill_selector_template_path),
            "tool_selector": str(self.router_config.tool_selector_template_path),
        }
        if self.router_config.capability_enabled:
            paths.update({
                "capability_workflow": str(self.router_config.capability_workflow_template_path),
                "capability_context": str(self.router_config.capability_context_template_path),
            })
        return {
            "harness_version": self.harness_version,
            "component_versions": self._component_version_snapshot(),
            "paths": paths,
        }

    def _matching_train_checkpoint(
        self,
        split: str,
        signature: dict[str, object],
    ) -> dict[str, Any]:
        if not self.checkpoint.enabled:
            return {}
        train_state = self.checkpoint.state.get("train")
        if not isinstance(train_state, dict):
            return {}
        split_state = train_state.get(split)
        if not isinstance(split_state, dict):
            return {}
        if split_state.get("signature") != signature:
            progress(f"checkpoint train ignored split={split} reason=signature_mismatch")
            return {}
        return split_state

    def _save_train_checkpoint(
        self,
        split: str,
        *,
        signature: dict[str, object],
        payload: dict[str, Any],
    ) -> None:
        if not self.checkpoint.enabled:
            return
        self.checkpoint.save(
            self,
            {
                "train": {
                    split: {
                        "signature": signature,
                        **payload,
                    }
                }
            },
        )

    def _matching_continual_checkpoint(self, signature: dict[str, object]) -> dict[str, Any]:
        if not self.checkpoint.enabled:
            return {}
        continual_state = self.checkpoint.state.get("continual")
        if not isinstance(continual_state, dict):
            return {}
        if continual_state.get("signature") != signature:
            progress("checkpoint continual ignored reason=signature_mismatch")
            return {}
        return continual_state

    def _save_continual_checkpoint(
        self,
        *,
        signature: dict[str, object],
        payload: dict[str, Any],
    ) -> None:
        if not self.checkpoint.enabled:
            return
        self.checkpoint.save(
            self,
            {
                "continual": {
                    "signature": signature,
                    **payload,
                }
            },
        )

    def _matching_test_checkpoint(self, split: str, signature: dict[str, object]) -> dict[str, Any]:
        if not self.checkpoint.enabled:
            return {}
        test_state = self.checkpoint.state.get("test")
        if not isinstance(test_state, dict):
            return {}
        split_state = test_state.get(split)
        if not isinstance(split_state, dict):
            return {}
        if split_state.get("signature") != signature:
            progress(f"checkpoint test ignored split={split} reason=signature_mismatch")
            return {}
        return split_state

    def _save_test_checkpoint(
        self,
        split: str,
        *,
        signature: dict[str, object],
        result: dict[str, object],
    ) -> None:
        if not self.checkpoint.enabled:
            return
        self.checkpoint.save(
            self,
            {
                "test": {
                    split: {
                        "signature": signature,
                        "completed": True,
                        "result": result,
                    }
                }
            },
        )

    def run(
        self,
        data: str | Path | Iterable[dict[str, Any]],
        *,
        val_data: str | Path | Iterable[dict[str, Any]] | None = None,
        split: str = "train",
        learn: bool = True,
        batchsize: int | None = None,
        reference_predictions: list[dict[str, object]] | None = None,
    ) -> PipelineResult:
        if learn:
            return self.train(data, val_data=val_data, split=split, batchsize=batchsize)
        return self.test(data, split=split, reference_predictions=reference_predictions)

    def train(
        self,
        train_data: str | Path | Iterable[dict[str, Any]],
        *,
        val_data: str | Path | Iterable[dict[str, Any]] | None,
        split: str = "train",
        batchsize: int | None = None,
    ) -> PipelineResult:
        examples = self.dataset_loader.load(train_data)
        val_examples = self.dataset_loader.load(val_data) if val_data is not None else None
        progress(
            f"train start split={split} train_examples={len(examples)} "
            f"validation_examples={len(val_examples) if val_examples is not None else 0}"
        )
        if val_examples is None:
            train_result = self._evaluate_train_batch(examples, split=split)
            self.evaluation_recorder.record_update(
                {
                    "accepted": False,
                    "reason": "missing_validation_data",
                    "harness_version": self.harness_version,
                }
            )
            return PipelineResult(
                evaluation=train_result,
                task_interface_config=self.task_interface_config,
                router_config=self.router_config,
                optimizer_history=list(self.optimizer.history),
            )

        current_val_examples = val_examples
        historical_anchor_examples = _historical_anchors_for_train(
            self.memory.historical_anchors(),
            examples,
        )
        anchor_context = self._historical_anchor_context(historical_anchor_examples)
        gate_examples = _unique_examples(current_val_examples + historical_anchor_examples)
        batches = list(_batch_examples(examples, batchsize))
        signature = _train_checkpoint_signature(
            split=split,
            examples=examples,
            val_examples=val_examples,
            batch_count=len(batches),
            batchsize=batchsize,
        )
        train_checkpoint = self._matching_train_checkpoint(split, signature)
        if train_checkpoint.get("completed") and isinstance(train_checkpoint.get("final_result"), dict):
            progress(f"checkpoint train hit split={split} completed_batches={train_checkpoint.get('completed_batches', 0)}")
            return PipelineResult(
                evaluation=dict(train_checkpoint["final_result"]),
                task_interface_config=self.task_interface_config,
                router_config=self.router_config,
                generated_candidates=list(train_checkpoint.get("generated_candidates", [])),
                candidate_evaluations=dict(train_checkpoint.get("candidate_evaluations", {})),
                current_validation=train_checkpoint.get("current_validation") if isinstance(train_checkpoint.get("current_validation"), dict) else None,
                accepted_candidate=train_checkpoint.get("accepted_candidate") if isinstance(train_checkpoint.get("accepted_candidate"), dict) else None,
                optimizer_history=list(self.optimizer.history),
            )
        completed_batches = min(int(train_checkpoint.get("completed_batches", 0) or 0), len(batches))
        generated_candidates: list[dict[str, object]] = list(train_checkpoint.get("generated_candidates", []))
        candidate_evaluations: dict[str, dict[str, object]] = dict(train_checkpoint.get("candidate_evaluations", {}))
        current_validation: dict[str, object] | None = (
            train_checkpoint.get("current_validation")
            if isinstance(train_checkpoint.get("current_validation"), dict)
            else None
        )
        accepted_candidate: dict[str, object] | None = (
            train_checkpoint.get("accepted_candidate")
            if isinstance(train_checkpoint.get("accepted_candidate"), dict)
            else None
        )
        last_train_result: dict[str, object] | None = (
            train_checkpoint.get("last_train_result")
            if isinstance(train_checkpoint.get("last_train_result"), dict)
            else None
        )
        for batch_index, batch_examples in enumerate(batches):
            if batch_index < completed_batches:
                progress(
                    f"checkpoint skip train batch split={split} "
                    f"batch={batch_index + 1}/{len(batches)}"
                )
                continue
            batch_split = split if len(batches) == 1 else f"{split}_batch_{batch_index + 1}"
            batch_result = self._train_batch_and_optimize(
                batch_examples,
                val_examples=gate_examples,
                current_val_examples=current_val_examples,
                historical_anchor_examples=historical_anchor_examples,
                anchor_context=anchor_context,
                split=batch_split,
                batch_index=batch_index,
                batch_count=len(batches),
            )
            last_train_result = batch_result.evaluation
            generated_candidates.extend(batch_result.generated_candidates)
            for candidate_id, candidate_evaluation in batch_result.candidate_evaluations.items():
                result_key = candidate_id
                if result_key in candidate_evaluations:
                    result_key = f"{candidate_id}#batch_{batch_index + 1}"
                candidate_evaluations[result_key] = candidate_evaluation
            current_validation = batch_result.current_validation
            if batch_result.accepted_candidate is not None:
                accepted_candidate = batch_result.accepted_candidate
            self.global_batch_index += 1
            self._maybe_run_forgetting_probe_after_batch()
            self._save_train_checkpoint(
                split,
                signature=signature,
                payload={
                    "completed": False,
                    "completed_batches": batch_index + 1,
                    "batch_count": len(batches),
                    "generated_candidates": generated_candidates,
                    "candidate_evaluations": candidate_evaluations,
                    "current_validation": current_validation,
                    "accepted_candidate": accepted_candidate,
                    "last_train_result": last_train_result,
                },
            )
            progress(
                f"checkpoint train saved split={split} "
                f"completed_batches={batch_index + 1}/{len(batches)}"
            )

        if len(batches) > 1 and self.run_train_final:
            final_predictions = self._predict(examples, progress_label="train_final")
            final_train_result = self.evaluator.evaluate(examples, final_predictions, split=split)
            self.evaluation_recorder.record_metrics(
                phase="train_final",
                split=split,
                result=final_train_result,
                harness_version=self.harness_version,
            )
        elif len(batches) > 1:
            final_train_result = last_train_result or self.evaluator.evaluate([], [], split=split)
            progress(f"train final skipped split={split} reason=execution.run_train_final_false")
        else:
            final_train_result = last_train_result or self.evaluator.evaluate([], [], split=split)
        self._save_train_checkpoint(
            split,
            signature=signature,
            payload={
                "completed": True,
                "completed_batches": len(batches),
                "batch_count": len(batches),
                "generated_candidates": generated_candidates,
                "candidate_evaluations": candidate_evaluations,
                "current_validation": current_validation,
                "accepted_candidate": accepted_candidate,
                "last_train_result": last_train_result,
                "final_result": final_train_result,
            },
        )
        return PipelineResult(
            evaluation=final_train_result,
            task_interface_config=self.task_interface_config,
            router_config=self.router_config,
            generated_candidates=generated_candidates,
            candidate_evaluations=candidate_evaluations,
            current_validation=current_validation,
            accepted_candidate=accepted_candidate,
            optimizer_history=list(self.optimizer.history),
        )

    def _evaluate_train_batch(self, examples: list[dict[str, Any]], *, split: str) -> dict[str, object]:
        progress(f"train batch inference start split={split} examples={len(examples)} memory_mode=read_write")
        predictions = self._predict(examples, update_memory=True, progress_label=split)
        examples_by_task: dict[str, list[dict[str, Any]]] = {}
        for example in examples:
            task_name = str(example.get("task_name", ""))
            examples_by_task.setdefault(task_name, []).append(example)
        for task_name, task_examples in examples_by_task.items():
            self.memory.consolidate_task(task_name, task_examples)
            progress(f"memory abstract updated task={task_name} summary={self.memory.summary()}")
        train_result = self.evaluator.evaluate(examples, predictions, split=split)
        progress(
            f"train batch inference done split={split} "
            f"primary_metric={train_result['metrics'].get('primary_metric_name')} "
            f"primary_score={_primary_metric_score(train_result['metrics']):.4f}"
        )
        self.evaluation_recorder.record_metrics(
            phase="train",
            split=split,
            result=train_result,
            harness_version=self.harness_version,
        )
        return train_result

    def _train_batch_and_optimize(
        self,
        examples: list[dict[str, Any]],
        *,
        val_examples: list[dict[str, Any]],
        current_val_examples: list[dict[str, Any]],
        historical_anchor_examples: list[dict[str, Any]],
        anchor_context: dict[str, object],
        split: str,
        batch_index: int,
        batch_count: int,
    ) -> PipelineResult:
        memory_snapshot = self.memory.snapshot()
        progress(f"memory candidate baseline start gate_examples={len(val_examples)}")
        memory_reference_predictions = self._predict(
            val_examples,
            progress_label="memory_gate:stable",
        )
        memory_reference = self._evaluate_gate(
            val_examples,
            memory_reference_predictions,
            current_val_examples=current_val_examples,
            historical_anchor_examples=historical_anchor_examples,
            split="val:memory_stable",
        )
        train_result = self._evaluate_train_batch(examples, split=split)
        memory_candidate_predictions = self._predict(
            val_examples,
            progress_label="memory_gate:candidate",
        )
        memory_candidate = self._evaluate_gate(
            val_examples,
            memory_candidate_predictions,
            current_val_examples=current_val_examples,
            historical_anchor_examples=historical_anchor_examples,
            split="val:memory_candidate",
            reference_predictions=memory_reference_predictions,
        )
        memory_accepted, memory_reasons = self._memory_candidate_decision(
            memory_reference,
            memory_candidate,
            anchor_context=anchor_context,
        )
        if memory_accepted:
            stable_val_predictions = memory_candidate_predictions
            progress(f"memory candidate committed summary={self.memory.summary()}")
        else:
            self.memory.restore(memory_snapshot)
            stable_val_predictions = memory_reference_predictions
            progress(f"memory candidate rejected reasons={memory_reasons} restored={self.memory.summary()}")
        anchor_update = self._select_batch_anchors(
            examples,
            train_result,
            split=split,
            batch_index=batch_index,
            batch_count=batch_count,
        )
        stable_validation = self._evaluate_gate(
            val_examples,
            stable_val_predictions,
            current_val_examples=current_val_examples,
            historical_anchor_examples=historical_anchor_examples,
            split="val",
        )
        self.evaluation_recorder.record_memory_update(
            {
                "accepted": memory_accepted,
                "reason": "memory_candidate_passed_commit_gate" if memory_accepted else "memory_candidate_rejected",
                "rejection_reasons": memory_reasons,
                "selection_objective": self.optimizer.selection_objective,
                "batch_index": batch_index,
                "batch_count": batch_count,
                "stable_current_task_metrics": memory_reference.get("current_task_metrics", {}),
                "candidate_current_task_metrics": memory_candidate.get("current_task_metrics", {}),
                "candidate_historical_anchor_metrics": memory_candidate.get("historical_anchor_metrics", {}),
                "memory_summary": self.memory.summary(),
                "anchor_update": anchor_update,
            }
        )
        generated_candidates: list[dict[str, object]] = []
        candidate_evaluations: dict[str, dict[str, object]] = {}
        current_validation: dict[str, object] | None = None
        accepted_candidate: dict[str, object] | None = None

        if bool(anchor_context.get("required")) and not bool(anchor_context.get("ready")):
            progress(
                "optimizer skipped reason=insufficient_historical_anchors "
                f"anchor_context={anchor_context}"
            )
            self.evaluation_recorder.record_update(
                {
                    "accepted": False,
                    "reason": "insufficient_historical_anchors",
                    "harness_version": self.harness_version,
                    "batch_index": batch_index,
                    "batch_count": batch_count,
                    "anchor_context": anchor_context,
                }
            )
            return PipelineResult(
                evaluation=train_result,
                task_interface_config=self.task_interface_config,
                router_config=self.router_config,
                generated_candidates=generated_candidates,
                candidate_evaluations=candidate_evaluations,
                current_validation=stable_validation,
                accepted_candidate=None,
                optimizer_history=list(self.optimizer.history),
            )

        components_to_optimize: list[str] = []
        applicability_interface = TaskInterface(self.task_interface_config, model=self.selection_model)
        for component in self.optimizer.components:
            if component != "task_interface_structuring":
                components_to_optimize.append(component)
                continue
            active_count = sum(
                applicability_interface.uses_llm_structuring(example)
                for example in current_val_examples
            )
            if active_count:
                components_to_optimize.append(component)
                continue
            progress(
                "optimizer component skipped component=task_interface_structuring "
                "reason=no_active_current_validation_examples "
                f"active_examples=0 total_current_validation_examples={len(current_val_examples)}"
            )
            self.evaluation_recorder.record_update(
                {
                    "accepted": False,
                    "reason": "no_active_current_validation_examples",
                    "harness_version": self.harness_version,
                    "batch_index": batch_index,
                    "batch_count": batch_count,
                    "component": component,
                    "active_current_validation_examples": 0,
                    "total_current_validation_examples": len(current_val_examples),
                }
            )

        for component_index, component in enumerate(components_to_optimize):
            progress(
                f"optimizer component start component={component} gate_examples={len(val_examples)} "
                f"harness_version={self.harness_version}"
            )
            if component_index == 0:
                current_val_predictions = stable_val_predictions
                current_validation = stable_validation
            else:
                current_val_predictions = self._predict(
                    val_examples,
                    progress_label=f"validation_current:{component}",
                )
                current_validation = self._evaluate_gate(
                    val_examples,
                    current_val_predictions,
                    current_val_examples=current_val_examples,
                    historical_anchor_examples=historical_anchor_examples,
                    split="val",
                )
            gate_metrics = current_validation.get("metrics", {})
            current_task_metrics = current_validation.get("current_task_metrics", {})
            anchor_metrics = current_validation.get("historical_anchor_metrics", {})
            progress(
                f"validation gate baseline component={component} "
                f"current_primary_metric={current_task_metrics.get('primary_metric_name')} "
                f"current_primary_score={_primary_metric_score(current_task_metrics):.4f} "
                f"anchor_task_primary_scores={anchor_metrics.get('task_primary_scores', {})} "
                f"format_compliance={current_task_metrics.get('format_compliance_rate', 0.0):.4f}"
            )
            self.evaluation_recorder.record_metrics(
                phase="val_current",
                split="val",
                result=current_validation,
                harness_version=self.harness_version,
                artifact_id=component,
            )
            try:
                current_artifact = self._component_artifact(component)
            except Exception as exc:
                self.evaluation_recorder.record_update(
                    {
                        "accepted": False,
                        "reason": "unsupported_or_invalid_component_artifact",
                        "harness_version": self.harness_version,
                        "batch_index": batch_index,
                        "batch_count": batch_count,
                        "component": component,
                        "error": str(exc),
                    }
                )
                continue

            component_candidates = self.optimizer.generate_artifact_candidates(
                component,
                train_result,
                model=self.judge_model,
                current_artifact=current_artifact,
                harness_version=self.harness_version,
                generation_context={
                    "selection_objective": self.optimizer.selection_objective,
                    "objective_guidance": self._optimizer_objective_guidance(),
                    "gate_metrics": current_validation.get("metrics", {}),
                    "current_task_metrics": current_validation.get("current_task_metrics", {}),
                    "historical_anchor_metrics": current_validation.get("historical_anchor_metrics", {}),
                    "commit_gate": {
                        "selection_objective": self.optimizer.selection_objective,
                        "commit_policy": self.optimizer.commit_policy,
                        "historical_loss_budget": self.optimizer.historical_loss_budget,
                        "min_primary_score_delta": self.optimizer.min_primary_score_delta,
                        "min_correct_gain": self.optimizer.min_correct_gain,
                        "max_forget_count": self.optimizer.max_forget_count,
                        "min_format_compliance_rate": self.optimizer.min_format_compliance_rate,
                        "recover_reward": self.optimizer.recover_reward,
                        "forget_penalty": self.optimizer.forget_penalty,
                        "min_anchor_score": self.optimizer.min_anchor_score,
                        "min_anchor_net_score": self.optimizer.min_anchor_net_score,
                        "max_anchor_forget_rate": self.optimizer.max_anchor_forget_rate,
                        "max_anchor_forget_count": self.optimizer.max_anchor_forget_count,
                        "plasticity_weight": self.optimizer.plasticity_weight,
                        "plasticity_scale": self.optimizer.plasticity_scale,
                        "stability_scale": self.optimizer.stability_scale,
                        "min_joint_score": self.optimizer.min_joint_score,
                    },
                    "anchor_context": anchor_context,
                    "screening": {
                        "ratio": self.optimizer.screening_ratio,
                        "min_current_count": self.optimizer.screening_min_current_count,
                        "min_anchors_per_task": self.optimizer.screening_min_anchors_per_task,
                    },
                    "batch_index": batch_index,
                    "batch_count": batch_count,
                    "batch_train_phase": f"train_batch_{batch_index + 1}",
                    "component": component,
                    "note": "Candidates are evaluated on validation data by temporarily replacing one harness artifact.",
                },
            )
            progress(
                f"optimizer candidates generated component={component} count={len(component_candidates)} "
                f"parse_status={self.optimizer.last_generation_diagnostics.get('parse_status', 'unknown')} "
                f"raw_candidates={self.optimizer.last_generation_diagnostics.get('raw_candidate_count', 0)}"
            )
            generated_candidates.extend(component_candidates)
            component_evaluations: dict[str, dict[str, object]] = {}
            promoted_candidates = component_candidates
            if self.optimizer.selection_objective == "balanced" and len(component_candidates) > 1:
                screening_current, screening_anchors, screening_examples = self._candidate_screening_examples(
                    current_val_examples,
                    historical_anchor_examples,
                    component=component,
                    batch_index=batch_index,
                )
                if len(screening_examples) < len(val_examples):
                    screening_ids = {
                        str(example.get("task_id", ""))
                        for example in screening_examples
                    }
                    screening_reference_predictions = [
                        prediction
                        for prediction in current_val_predictions
                        if str(prediction.get("task_id", "")) in screening_ids
                    ]
                    current_screening = self._evaluate_gate(
                        screening_examples,
                        screening_reference_predictions,
                        current_val_examples=screening_current,
                        historical_anchor_examples=screening_anchors,
                        split="val:screening_current",
                    )
                    screening_results: dict[str, dict[str, object]] = {}
                    for candidate in component_candidates:
                        candidate_id = str(candidate.get("candidate_id") or "")
                        progress(
                            f"candidate screening start id={candidate_id} "
                            f"gate_examples={len(screening_examples)}"
                        )
                        screening_result = self._evaluate_component_candidate(
                            candidate,
                            val_examples=screening_examples,
                            current_val_examples=screening_current,
                            historical_anchor_examples=screening_anchors,
                            reference_predictions=screening_reference_predictions,
                        )
                        screening_results[candidate_id] = screening_result
                        self.evaluation_recorder.record_metrics(
                            phase="candidate_screening",
                            split="val:screening",
                            result=screening_result,
                            harness_version=self.harness_version,
                            candidate_id=candidate_id,
                            artifact_id=str(candidate.get("component") or ""),
                        )
                    promoted_candidates, screening_diagnostics = self.optimizer.shortlist_candidates(
                        component_candidates,
                        screening_results,
                        current_screening,
                    )
                    promoted_ids = {
                        str(candidate.get("candidate_id") or "")
                        for candidate in promoted_candidates
                    }
                    for candidate in component_candidates:
                        candidate_id = str(candidate.get("candidate_id") or "")
                        if candidate_id in promoted_ids:
                            continue
                        component_evaluations[candidate_id] = _screening_only_result(
                            screening_results.get(candidate_id, {}),
                            screening_diagnostics,
                        )
                        candidate_evaluations[candidate_id] = component_evaluations[candidate_id]
                    progress(
                        f"candidate screening done component={component} generated={len(component_candidates)} "
                        f"promoted={len(promoted_candidates)} full_gate_examples={len(val_examples)}"
                    )

            for candidate in promoted_candidates:
                candidate_id = str(candidate.get("candidate_id") or "")
                progress(f"candidate evaluation start id={candidate_id} gate_examples={len(val_examples)}")
                candidate_result = self._evaluate_component_candidate(
                    candidate,
                    val_examples=val_examples,
                    current_val_examples=current_val_examples,
                    historical_anchor_examples=historical_anchor_examples,
                    reference_predictions=current_val_predictions,
                )
                candidate_metrics = candidate_result.get("metrics", {})
                candidate_task_metrics = candidate_result.get("current_task_metrics", {})
                candidate_anchor_metrics = candidate_result.get("historical_anchor_metrics", {})
                progress(
                    f"candidate evaluation done id={candidate_id} "
                    f"current_primary_metric={candidate_task_metrics.get('primary_metric_name')} "
                    f"current_primary_score={_primary_metric_score(candidate_task_metrics):.4f} "
                    f"anchor_task_primary_scores={candidate_anchor_metrics.get('task_primary_scores', {})} "
                    f"format_compliance={candidate_task_metrics.get('format_compliance_rate', 0.0):.4f} "
                    f"forget_count={candidate_metrics.get('forget_count', 0)}"
                )
                component_evaluations[candidate_id] = candidate_result
                candidate_evaluations[candidate_id] = candidate_result
                self.evaluation_recorder.record_metrics(
                    phase="candidate_val",
                    split="val",
                    result=candidate_result,
                    harness_version=self.harness_version,
                    candidate_id=candidate_id,
                    artifact_id=str(candidate.get("component") or ""),
                )

            component_accepted = self.optimizer.select_best_candidate(
                component_candidates,
                component_evaluations,
                current_validation,
                anchor_context=anchor_context,
            )
            if component_accepted is not None:
                accepted_candidate = component_accepted
                progress(f"candidate accepted id={component_accepted.get('candidate_id')}")
            else:
                decision_reason = str(self.optimizer.history[-1].get("reason", "commit_gate_rejected"))
                progress(f"candidate rejected component={component} reason={decision_reason}")
            self._record_component_update_decision(
                component=component,
                candidates=component_candidates,
                candidate_evaluations=component_evaluations,
                current_validation=current_validation,
                accepted_candidate=component_accepted,
                batch_index=batch_index,
                batch_count=batch_count,
            )
        return PipelineResult(
            evaluation=train_result,
            task_interface_config=self.task_interface_config,
            router_config=self.router_config,
            generated_candidates=generated_candidates,
            candidate_evaluations=candidate_evaluations,
            current_validation=current_validation,
            accepted_candidate=accepted_candidate,
            optimizer_history=list(self.optimizer.history),
        )

    def _select_batch_anchors(
        self,
        examples: list[dict[str, Any]],
        train_result: dict[str, object],
        *,
        split: str,
        batch_index: int,
        batch_count: int,
    ) -> dict[str, object]:
        if not self.memory.batch_anchor_selection_enabled():
            return {"selected": [], "reason": "batch_anchor_selection_disabled"}
        predictions = train_result.get("predictions", [])
        if not isinstance(predictions, list):
            return {"selected": [], "reason": "missing_train_predictions"}
        examples_by_task: dict[str, list[dict[str, Any]]] = {}
        for example in examples:
            examples_by_task.setdefault(str(example.get("task_name", "")), []).append(example)
        updates: list[dict[str, Any]] = []
        for task_name, task_examples in examples_by_task.items():
            task_ids = {str(item.get("task_id", "")) for item in task_examples}
            task_predictions = [
                item
                for item in predictions
                if isinstance(item, dict) and str(item.get("task_id", "")) in task_ids
            ]
            update = self.memory.select_batch_anchors(
                task_name,
                task_examples,
                task_predictions,
                split=split,
                batch_index=batch_index,
                batch_count=batch_count,
            )
            updates.append(update)
            progress(
                f"memory anchors selected task={task_name} split={split} "
                f"selected={len(update.get('selected', []))} "
                f"target={update.get('target_count', 0)} summary={self.memory.summary()}"
            )
        return {"updates": updates}

    def _memory_candidate_decision(
        self,
        stable: dict[str, object],
        candidate: dict[str, object],
        *,
        anchor_context: dict[str, object] | None = None,
    ) -> tuple[bool, list[str]]:
        stable_current = stable.get("current_task_metrics", {})
        candidate_current = candidate.get("current_task_metrics", {})
        candidate_anchors = candidate.get("historical_anchor_metrics", {})
        reasons: list[str] = []
        stable_primary_score = _primary_metric_score(stable_current)
        candidate_primary_score = _primary_metric_score(candidate_current)
        stable_correct = int(stable_current.get("correct", 0) or 0)
        candidate_correct = int(candidate_current.get("correct", 0) or 0)
        candidate_format = float(candidate_current.get("format_compliance_rate", 0.0) or 0.0)
        anchor_forget_count = int(candidate_anchors.get("forget_count", 0) or 0)
        if (
            self.optimizer.historical_loss_budget is not None
            and anchor_forget_count > self.optimizer.historical_loss_budget
        ):
            reasons.append("historical_loss_budget_exceeded")
        if self.optimizer.selection_objective == "stability":
            if int(candidate_current.get("forget_count", 0) or 0) > self.optimizer.max_forget_count:
                reasons.append("current_task_forgetting_exceeded")
            if int(candidate_anchors.get("forget_count", 0) or 0) > self.optimizer.max_forget_count:
                reasons.append("historical_anchor_forgetting_exceeded")
            if candidate_primary_score < stable_primary_score:
                reasons.append("current_task_primary_score_regressed")
            if candidate_format < float(stable_current.get("format_compliance_rate", 0.0) or 0.0):
                reasons.append("current_task_format_regressed")
        elif self.optimizer.selection_objective == "plasticity":
            if candidate_primary_score - stable_primary_score + 1e-12 < self.optimizer.min_primary_score_delta:
                reasons.append("current_task_primary_score_delta_below_minimum")
            if candidate_correct - stable_correct < self.optimizer.min_correct_gain:
                reasons.append("current_task_correct_gain_below_minimum")
            if candidate_format + 1e-12 < self.optimizer.min_format_compliance_rate:
                reasons.append("current_task_format_below_minimum")
        else:
            anchor_context = anchor_context or {}
            if bool(anchor_context.get("required")) and not bool(anchor_context.get("ready")):
                reasons.append("historical_anchor_requirement_not_met")
            if candidate_primary_score - stable_primary_score + 1e-12 < self.optimizer.min_primary_score_delta:
                reasons.append("current_task_primary_score_delta_below_minimum")
            if candidate_correct - stable_correct < self.optimizer.min_correct_gain:
                reasons.append("current_task_correct_gain_below_minimum")
            if candidate_format + 1e-12 < self.optimizer.min_format_compliance_rate:
                reasons.append("current_task_format_below_minimum")
            if bool(anchor_context.get("required")):
                anchor_forget_rate = float(candidate_anchors.get("forget_rate", 0.0) or 0.0)
                anchor_forget_count = int(candidate_anchors.get("forget_count", 0) or 0)
                if (
                    self.optimizer.max_anchor_forget_rate is not None
                    and anchor_forget_rate > float(self.optimizer.max_anchor_forget_rate) + 1e-12
                ):
                    reasons.append("historical_anchor_forget_rate_exceeded")
                if (
                    self.optimizer.max_anchor_forget_count is not None
                    and anchor_forget_count > int(self.optimizer.max_anchor_forget_count)
                ):
                    reasons.append("historical_anchor_forget_count_exceeded")
        return not reasons, reasons

    def _historical_anchor_context(
        self,
        historical_anchor_examples: list[dict[str, Any]],
    ) -> dict[str, object]:
        configured_historical = self._active_context_value("historical_tasks")
        if isinstance(configured_historical, list):
            expected_tasks = [str(task) for task in configured_historical if str(task)]
        else:
            expected_tasks = sorted(
                {
                    str(example.get("task_name") or "").strip()
                    for example in historical_anchor_examples
                    if str(example.get("task_name") or "").strip()
                }
            )
        task_counts = {
            task: sum(
                str(example.get("task_name") or "").strip() == task
                for example in historical_anchor_examples
            )
            for task in expected_tasks
        }
        required = bool(expected_tasks)
        minimum_total = max(
            int(self.optimizer.min_historical_anchor_count),
            int(self.optimizer.min_historical_anchors_per_task) * len(expected_tasks),
        ) if required else 0
        missing_tasks = [task for task in expected_tasks if task_counts.get(task, 0) == 0]
        underfilled_tasks = [
            task
            for task in expected_tasks
            if task_counts.get(task, 0) < int(self.optimizer.min_historical_anchors_per_task)
        ]
        coverage_ready = not missing_tasks if self.optimizer.require_all_historical_tasks else True
        ready = (
            not required
            or (
                len(historical_anchor_examples) >= minimum_total
                and not underfilled_tasks
                and coverage_ready
            )
        )
        return {
            "required": required,
            "ready": ready,
            "total": len(historical_anchor_examples),
            "minimum_total": minimum_total,
            "minimum_per_task": int(self.optimizer.min_historical_anchors_per_task),
            "expected_tasks": expected_tasks,
            "task_counts": task_counts,
            "missing_tasks": missing_tasks,
            "underfilled_tasks": underfilled_tasks,
            "require_all_historical_tasks": bool(self.optimizer.require_all_historical_tasks),
        }

    def _optimizer_objective_guidance(self) -> str:
        if self.optimizer.selection_objective == "stability":
            return "Preserve every currently correct gate behavior; minimize forgetting."
        if self.optimizer.selection_objective == "plasticity":
            return "Maximize the named primary metric and correct-answer gain on the current new task."
        return (
            "Balance current-task primary-metric gain against historical-anchor retention. "
            "Propose distinct conservative, balanced, and adaptive prompt-artifact changes."
        )

    def _candidate_screening_examples(
        self,
        current_val_examples: list[dict[str, Any]],
        historical_anchor_examples: list[dict[str, Any]],
        *,
        component: str,
        batch_index: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        ratio = self.optimizer.screening_ratio
        current_count = min(
            len(current_val_examples),
            max(
                int(self.optimizer.screening_min_current_count),
                math.ceil(len(current_val_examples) * ratio),
            ),
        )
        seed_prefix = (
            f"{self.optimizer.selection_seed}:{self.harness_version}:"
            f"{batch_index}:{component}"
        )
        current_screening = _deterministic_screening_sample(
            current_val_examples,
            current_count,
            seed=f"{seed_prefix}:current",
        )
        anchors_by_task: dict[str, list[dict[str, Any]]] = {}
        for example in historical_anchor_examples:
            task_name = str(example.get("task_name") or "")
            anchors_by_task.setdefault(task_name, []).append(example)
        anchor_screening: list[dict[str, Any]] = []
        for task_name, task_examples in sorted(anchors_by_task.items()):
            task_count = min(
                len(task_examples),
                max(
                    int(self.optimizer.screening_min_anchors_per_task),
                    math.ceil(len(task_examples) * ratio),
                ),
            )
            anchor_screening.extend(
                _deterministic_screening_sample(
                    task_examples,
                    task_count,
                    seed=f"{seed_prefix}:anchor:{task_name}",
                )
            )
        return current_screening, anchor_screening, _unique_examples(current_screening + anchor_screening)
    def _component_artifact(self, component: str) -> dict[str, object]:
        path = self._component_artifact_path(component)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Component artifact must be a JSON object: {path}")
        return {
            "component": component,
            "path": str(path),
            "content": data,
        }

    def _component_artifact_path(self, component: str) -> Path:
        if component == "task_interface_structuring":
            return Path(self.task_interface_config.structuring_template_path)
        if component == "router_workflow":
            return Path(
                self.router_config.capability_workflow_template_path
                if self.router_config.capability_enabled
                else self.router_config.workflow_template_path
            )
        if component == "router_context":
            return Path(
                self.router_config.capability_context_template_path
                if self.router_config.capability_enabled
                else self.router_config.context_template_path
            )
        if component == "memory_selector":
            return Path(self.router_config.memory_selector_template_path)
        if component == "skill_selector":
            return Path(self.router_config.skill_selector_template_path)
        if component == "tool_selector":
            return Path(self.router_config.tool_selector_template_path)
        raise ValueError(f"Unsupported optimizable component: {component}")

    def _evaluate_component_candidate(
        self,
        candidate: dict[str, object],
        *,
        val_examples: list[dict[str, Any]],
        current_val_examples: list[dict[str, Any]],
        historical_anchor_examples: list[dict[str, Any]],
        reference_predictions: list[dict[str, Any]],
    ) -> dict[str, object]:
        component = str(candidate.get("component") or "")
        artifact_content = candidate.get("artifact_content")
        if not isinstance(artifact_content, dict):
            return self._invalid_candidate_result("candidate artifact_content is not a JSON object")
        validation_errors = candidate.get("validation_errors")
        if isinstance(validation_errors, list) and validation_errors:
            return self._invalid_candidate_result("; ".join(str(item) for item in validation_errors))
        try:
            if component == "task_interface_structuring":
                task_interface = TaskInterface(
                    self.task_interface_config,
                    model=self.selection_model,
                    structuring_template_content=artifact_content,
                )
                router = Router(
                    self.model,
                    self.router_config,
                    memory=self.memory,
                    selection_model=self.selection_model,
                )
                predictions = self._predict_with_router(
                    val_examples,
                    router,
                    task_interface=task_interface,
                    progress_label=f"candidate:{candidate.get('candidate_id')}",
                )
            elif component == "router_workflow":
                router = Router(
                    self.model,
                    self.router_config,
                    memory=self.memory,
                    selection_model=self.selection_model,
                    workflow_template_content=artifact_content,
                )
                predictions = self._predict_with_router(
                    val_examples,
                    router,
                    progress_label=f"candidate:{candidate.get('candidate_id')}",
                )
            elif component == "router_context":
                router = Router(
                    self.model,
                    self.router_config,
                    memory=self.memory,
                    selection_model=self.selection_model,
                    context_template_content=artifact_content,
                )
                predictions = self._predict_with_router(
                    val_examples,
                    router,
                    progress_label=f"candidate:{candidate.get('candidate_id')}",
                )
            elif component == "memory_selector":
                router = Router(
                    self.model,
                    self.router_config,
                    memory=self.memory,
                    selection_model=self.selection_model,
                    memory_selector_template_content=artifact_content,
                )
                predictions = self._predict_with_router(
                    val_examples,
                    router,
                    progress_label=f"candidate:{candidate.get('candidate_id')}",
                )
            elif component == "skill_selector":
                router = Router(
                    self.model,
                    self.router_config,
                    memory=self.memory,
                    selection_model=self.selection_model,
                    skill_selector_template_content=artifact_content,
                )
                predictions = self._predict_with_router(
                    val_examples,
                    router,
                    progress_label=f"candidate:{candidate.get('candidate_id')}",
                )
            elif component == "tool_selector":
                router = Router(
                    self.model,
                    self.router_config,
                    memory=self.memory,
                    selection_model=self.selection_model,
                    tool_selector_template_content=artifact_content,
                )
                predictions = self._predict_with_router(
                    val_examples,
                    router,
                    progress_label=f"candidate:{candidate.get('candidate_id')}",
                )
            else:
                return self._invalid_candidate_result(f"unsupported component: {component}")
            return self._evaluate_gate(
                val_examples,
                predictions,
                current_val_examples=current_val_examples,
                historical_anchor_examples=historical_anchor_examples,
                split="val",
                reference_predictions=reference_predictions,
            )
        except Exception as exc:
            return self._invalid_candidate_result(str(exc))

    def _record_component_update_decision(
        self,
        *,
        component: str,
        candidates: list[dict[str, object]],
        candidate_evaluations: dict[str, dict[str, object]],
        current_validation: dict[str, object],
        accepted_candidate: dict[str, object] | None,
        batch_index: int,
        batch_count: int,
    ) -> None:
        accepted_candidate_id = str(accepted_candidate.get("candidate_id")) if accepted_candidate else None
        rejected_candidates = [
            candidate
            for candidate in candidates
            if str(candidate.get("candidate_id") or "") != accepted_candidate_id
        ]
        optimizer_decision = self.optimizer.history[-1] if self.optimizer.history else {}
        rejection_by_id = {
            str(item.get("candidate_id", "")): item
            for item in optimizer_decision.get("candidate_rejections", [])
            if isinstance(item, dict)
        }
        harness_version_before = self.harness_version
        baseline = {
            "artifact_id": component,
            "metrics": compact_metrics(dict(current_validation.get("metrics", {}))),
            "current_task_metrics": compact_metrics(dict(current_validation.get("current_task_metrics", {}))),
            "historical_anchor_metrics": compact_metrics(dict(current_validation.get("historical_anchor_metrics", {}))),
        }
        gate_candidates: list[dict[str, object]] = []
        for candidate in candidates:
            candidate_id = str(candidate.get("candidate_id") or "")
            candidate_result = candidate_evaluations.get(candidate_id, {})
            rejection = rejection_by_id.get(candidate_id, {})
            decision = "accepted" if candidate_id and candidate_id == accepted_candidate_id else "rejected"
            gate_candidates.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "candidate_name": candidate.get("candidate_name"),
                    "metrics": compact_metrics(dict(candidate_result.get("metrics", {}))),
                    "current_task_metrics": compact_metrics(dict(candidate_result.get("current_task_metrics", {}))),
                    "historical_anchor_metrics": compact_metrics(dict(candidate_result.get("historical_anchor_metrics", {}))),
                    "decision": decision,
                    "decision_reasons": [] if decision == "accepted" else rejection.get("reasons", []),
                    "validation_errors": candidate.get("validation_errors", []),
                }
            )
        if accepted_candidate is None:
            self.evaluation_recorder.record_update(
                {
                    "accepted": False,
                    "reason": optimizer_decision.get("reason", "no_candidate_passed_commit_gate"),
                    "harness_version": self.harness_version,
                    "batch_index": batch_index,
                    "batch_count": batch_count,
                    "component": component,
                    "current_validation_metrics": current_validation.get("metrics", {}),
                    "current_task_metrics": current_validation.get("current_task_metrics", {}),
                    "historical_anchor_metrics": current_validation.get("historical_anchor_metrics", {}),
                    "candidate_ids": [candidate.get("candidate_id") for candidate in candidates],
                    "commit_gate": optimizer_decision.get("commit_gate", {}),
                    "candidate_rejections": optimizer_decision.get("candidate_rejections", []),
                    "generation_diagnostics": optimizer_decision.get(
                        "generation_diagnostics",
                        self.optimizer.last_generation_diagnostics,
                    ),
                }
            )
        else:
            previous_harness_version = self.harness_version
            next_harness_version = self.harness_version + 1
            accepted_artifact_path = self.evaluation_recorder.persist_accepted_artifact(
                harness_version=next_harness_version,
                candidate=accepted_candidate,
            )
            accepted_result = candidate_evaluations.get(accepted_candidate_id or "", {})
            self._apply_accepted_candidate(component, accepted_artifact_path)
            self.harness_version = next_harness_version
            self.component_versions[component] = {
                "version": int(self.component_versions.get(component, {}).get("version", 0) or 0) + 1,
                "artifact_path": accepted_artifact_path,
                "artifact_hash": _file_sha256(Path(accepted_artifact_path)),
            }
            self.evaluation_recorder.record_accepted_component_artifact(
                {
                    "accepted": True,
                    "reason": "candidate_passed_commit_gate",
                    "previous_harness_version": previous_harness_version,
                    "new_harness_version": next_harness_version,
                    "batch_index": batch_index,
                    "batch_count": batch_count,
                    "candidate_id": accepted_candidate.get("candidate_id"),
                    "candidate_version": accepted_candidate.get("candidate_version"),
                    "component": component,
                    "artifact_path": accepted_artifact_path,
                    "current_metrics": current_validation.get("metrics", {}),
                    "candidate_metrics": accepted_result.get("metrics", {}),
                    "current_task_metrics": current_validation.get("current_task_metrics", {}),
                    "candidate_current_task_metrics": accepted_result.get("current_task_metrics", {}),
                    "historical_anchor_metrics": accepted_result.get("historical_anchor_metrics", {}),
                    "artifact_diff": accepted_candidate.get("artifact_diff"),
                    "rationale": accepted_candidate.get("rationale"),
                }
            )
            self.evaluation_recorder.record_update(
                {
                    "accepted": True,
                    "reason": "candidate_passed_commit_gate",
                    "previous_harness_version": previous_harness_version,
                    "new_harness_version": next_harness_version,
                    "batch_index": batch_index,
                    "batch_count": batch_count,
                    "candidate_id": accepted_candidate.get("candidate_id"),
                    "component": component,
                    "artifact_path": accepted_artifact_path,
                    "current_validation_metrics": current_validation.get("metrics", {}),
                    "candidate_validation_metrics": accepted_result.get("metrics", {}),
                    "current_task_metrics": current_validation.get("current_task_metrics", {}),
                    "candidate_current_task_metrics": accepted_result.get("current_task_metrics", {}),
                    "historical_anchor_metrics": accepted_result.get("historical_anchor_metrics", {}),
                }
            )
        self.evaluation_recorder.record_gate_validation(
            {
                "global_batch_index": self.global_batch_index + 1,
                "task_stage": self._active_context_value("task_stage"),
                "current_task": self._active_context_value("current_task"),
                "component": component,
                "harness_version_before": harness_version_before,
                "baseline": baseline,
                "candidates": gate_candidates,
                "accepted_candidate_id": accepted_candidate_id,
                "harness_version_after": self.harness_version,
                "component_versions_after": self._component_version_snapshot(),
                "decision": "accepted" if accepted_candidate is not None else "rejected",
                "decision_reason": optimizer_decision.get("reason"),
            }
        )
        for candidate in rejected_candidates:
            candidate_id = str(candidate.get("candidate_id") or "")
            rejection = rejection_by_id.get(candidate_id, {})
            self.evaluation_recorder.record_rejected_component_artifact(
                {
                    "accepted": False,
                    "reason": "accepted_other_candidate" if accepted_candidate else "candidate_failed_commit_gate",
                    "rejection_reasons": rejection.get("reasons", []),
                    "harness_version": self.harness_version,
                    "batch_index": batch_index,
                    "batch_count": batch_count,
                    "candidate_id": candidate.get("candidate_id"),
                    "candidate_version": candidate.get("candidate_version"),
                    "component": component,
                    "candidate_metrics": candidate_evaluations.get(candidate_id, {}).get("metrics", {}),
                    "candidate_current_task_metrics": candidate_evaluations.get(candidate_id, {}).get("current_task_metrics", {}),
                    "candidate_historical_anchor_metrics": candidate_evaluations.get(candidate_id, {}).get("historical_anchor_metrics", {}),
                    "current_metrics": current_validation.get("metrics", {}),
                    "current_task_metrics": current_validation.get("current_task_metrics", {}),
                    "validation_errors": candidate.get("validation_errors", []),
                    "artifact_diff": candidate.get("artifact_diff"),
                    "rationale": candidate.get("rationale"),
                }
            )

    def _apply_accepted_candidate(self, component: str, artifact_path: str) -> None:
        if component == "task_interface_structuring":
            self.task_interface_config.structuring_template_path = artifact_path
        elif component == "router_workflow":
            if self.router_config.capability_enabled:
                self.router_config.capability_workflow_template_path = artifact_path
            else:
                self.router_config.workflow_template_path = artifact_path
        elif component == "router_context":
            if self.router_config.capability_enabled:
                self.router_config.capability_context_template_path = artifact_path
            else:
                self.router_config.context_template_path = artifact_path
        elif component == "memory_selector":
            self.router_config.memory_selector_template_path = artifact_path
        elif component == "skill_selector":
            self.router_config.skill_selector_template_path = artifact_path
        elif component == "tool_selector":
            self.router_config.tool_selector_template_path = artifact_path
        else:
            raise ValueError(f"Unsupported accepted component: {component}")

    @staticmethod
    def _invalid_candidate_result(error: str) -> dict[str, object]:
        return {
            "split": "val",
            "metrics": {
                "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
                "primary_metric_name": "invalid_candidate",
                "primary_score": 0.0,
                "mean_example_score": 0.0,
                "correct": 0,
                "total": 0,
                "forget_count": 0,
                "forget_rate": 0.0,
                "reference_correct": 0,
                "recovered_count": 0,
                "reference_mean_example_score": None,
                "mean_example_score_delta": None,
                "has_reference": True,
                "format_compliant_count": 0,
                "format_compliance_rate": 0.0,
                "average_output_chars": 0.0,
                "max_output_chars": 0,
                "invalid_candidate": True,
            },
            "current_task_metrics": {
                "primary_metric_name": "invalid_candidate",
                "primary_score": 0.0,
                "correct": 0,
                "total": 0,
                "forget_count": 0,
                "forget_rate": 0.0,
                "format_compliance_rate": 0.0,
                "invalid_candidate": True,
            },
            "historical_anchor_metrics": {
                "primary_metric_name": "invalid_candidate",
                "primary_score": 0.0,
                "correct": 0,
                "total": 0,
                "forget_count": 0,
                "forget_rate": 0.0,
                "format_compliance_rate": 0.0,
                "invalid_candidate": True,
            },
            "predictions": [],
            "error": error,
        }

    def _evaluate_gate(
        self,
        gate_examples: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
        *,
        current_val_examples: list[dict[str, Any]],
        historical_anchor_examples: list[dict[str, Any]],
        split: str,
        reference_predictions: list[dict[str, Any]] | None = None,
    ) -> dict[str, object]:
        result = self.evaluator.evaluate(
            gate_examples,
            predictions,
            split=split,
            reference_predictions=reference_predictions,
        )
        current_result = self.evaluator.evaluate(
            current_val_examples,
            predictions,
            split=f"{split}:current_task",
            reference_predictions=reference_predictions,
        )
        anchor_result = self.evaluator.evaluate(
            historical_anchor_examples,
            predictions,
            split=f"{split}:historical_anchors",
            reference_predictions=reference_predictions,
        )
        anchor_examples_by_task: dict[str, list[dict[str, Any]]] = {}
        for example in historical_anchor_examples:
            task_name = str(example.get("task_name") or example.get("task_type") or "unknown")
            anchor_examples_by_task.setdefault(task_name, []).append(example)
        anchor_task_results = {
            task_name: self.evaluator.evaluate(
                task_examples,
                predictions,
                split=f"{split}:historical_anchors:{task_name}",
                reference_predictions=reference_predictions,
            )
            for task_name, task_examples in anchor_examples_by_task.items()
        }
        combined_metrics = result["metrics"]
        # A gate may contain current-task validation examples plus historical
        # anchors from tasks with fundamentally different metrics.  Never
        # expose their arithmetic mixture as the gate's top-level score.
        result["metrics"] = current_result["metrics"]
        result["combined_gate_metrics"] = combined_metrics
        result["metric_scope"] = "current_task"
        result["current_task_metrics"] = current_result["metrics"]
        result["historical_anchor_task_metrics"] = {
            task_name: task_result["metrics"]
            for task_name, task_result in anchor_task_results.items()
        }
        result["historical_anchor_metrics"] = _historical_anchor_metric_summary(
            result["historical_anchor_task_metrics"],
            fallback=anchor_result["metrics"],
        )
        return result

    def continual_train(
        self,
        task_stream_path: str | Path,
        *,
        train_split: str = "train",
        val_split: str = "validation",
        test_split: str = "test",
        limit_per_task: int | None = None,
        train_limit_per_task: int | None = None,
        validation_limit_per_task: int | None = None,
        test_limit_per_task: int | None = None,
        batchsize: int | None = None,
    ) -> PipelineResult:
        """Learn tasks in order and evaluate every seen task after each update."""
        train_limit_per_task = limit_per_task if train_limit_per_task is None else train_limit_per_task
        validation_limit_per_task = limit_per_task if validation_limit_per_task is None else validation_limit_per_task
        test_limit_per_task = limit_per_task if test_limit_per_task is None else test_limit_per_task
        specs = load_task_stream_specs(task_stream_path)
        task_order = [str(spec.get("task_name", "")) for spec in specs]
        progress(f"continual train start tasks={len(specs)} order={task_order}")
        signature = _continual_checkpoint_signature(
            task_stream_path=task_stream_path,
            task_order=task_order,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            train_limit_per_task=train_limit_per_task,
            validation_limit_per_task=validation_limit_per_task,
            test_limit_per_task=test_limit_per_task,
            batchsize=batchsize,
        )
        continual_checkpoint = self._matching_continual_checkpoint(signature)
        raw_primary_score_rows = continual_checkpoint.get("primary_score_rows", [])
        raw_forgetting_rows = continual_checkpoint.get("forgetting_rows", [])
        raw_stage_summaries = continual_checkpoint.get("stage_summaries", [])
        primary_score_rows: list[list[float | None]] = (
            list(raw_primary_score_rows) if isinstance(raw_primary_score_rows, list) else []
        )
        forgetting_rows: list[list[float | None]] = list(raw_forgetting_rows) if isinstance(raw_forgetting_rows, list) else []
        stage_summaries: list[dict[str, object]] = list(raw_stage_summaries) if isinstance(raw_stage_summaries, list) else []
        raw_previous_predictions = continual_checkpoint.get("previous_predictions", {})
        previous_predictions: dict[str, list[dict[str, object]]] = {
            str(key): list(value)
            for key, value in raw_previous_predictions.items()
            if isinstance(value, list)
        } if isinstance(raw_previous_predictions, dict) else {}
        seen_test_sets: dict[str, list[dict[str, Any]]] = {}
        raw_candidates = continual_checkpoint.get("generated_candidates", [])
        raw_candidate_evaluations = continual_checkpoint.get("candidate_evaluations", {})
        all_candidates: list[dict[str, object]] = list(raw_candidates) if isinstance(raw_candidates, list) else []
        all_candidate_evaluations: dict[str, dict[str, object]] = (
            dict(raw_candidate_evaluations) if isinstance(raw_candidate_evaluations, dict) else {}
        )
        accepted_candidate: dict[str, object] | None = (
            continual_checkpoint.get("accepted_candidate")
            if isinstance(continual_checkpoint.get("accepted_candidate"), dict)
            else None
        )
        last_train_result: PipelineResult | None = None
        completed_stages = min(int(continual_checkpoint.get("completed_stages", 0) or 0), len(specs))

        for stage_index, spec in enumerate(specs):
            task_name = task_order[stage_index]
            train_examples = self.dataset_loader.load_task_spec(
                spec,
                split=train_split,
                dataset_index=stage_index,
                limit=train_limit_per_task,
            )
            val_examples = self.dataset_loader.load_task_spec(
                spec,
                split=val_split,
                dataset_index=stage_index,
                limit=validation_limit_per_task,
            )
            if self.memory.batch_anchor_selection_enabled():
                anchor_examples = []
            else:
                anchor_examples = self.dataset_loader.load_task_spec(
                    spec,
                    split="anchor",
                    dataset_index=stage_index,
                    limit=self.memory.config.anchor_capacity_per_task,
                )
                if not anchor_examples:
                    anchor_examples = val_examples[: self.memory.config.anchor_capacity_per_task]
            seen_test_sets[task_name] = self.dataset_loader.load_task_spec(
                spec,
                split=test_split,
                dataset_index=stage_index,
                limit=test_limit_per_task,
            )
            self.historical_task_test_sets.update(seen_test_sets)
            if stage_index < completed_stages:
                progress(
                    f"checkpoint skip stage {stage_index + 1}/{len(specs)} "
                    f"task={task_name}"
                )
                continue
            progress(
                f"stage start {stage_index + 1}/{len(specs)} task={task_name} "
                f"train={len(train_examples)} val={len(val_examples)} "
                f"anchor={len(anchor_examples)} test={len(seen_test_sets[task_name])} "
                f"memory_before={self.memory.summary()}"
            )
            self._active_continual_context = {
                "task_stage": stage_index + 1,
                "current_task": task_name,
                "historical_tasks": task_order[:stage_index],
                "test_split": test_split,
            }
            try:
                last_train_result = self.train(
                    train_examples,
                    val_data=val_examples,
                    split=f"{train_split}:{task_name}",
                    batchsize=batchsize,
                )
            finally:
                self._active_continual_context = None
            if self.memory.batch_anchor_selection_enabled():
                progress(f"memory batch anchors retained task={task_name} summary={self.memory.summary()}")
            else:
                self.memory.finalize_task(task_name, anchor_examples)
                progress(f"memory anchors finalized task={task_name} summary={self.memory.summary()}")
            all_candidates.extend(last_train_result.generated_candidates)
            all_candidate_evaluations.update(last_train_result.candidate_evaluations)
            if last_train_result.accepted_candidate is not None:
                accepted_candidate = last_train_result.accepted_candidate

            primary_score_row: list[float | None] = [None] * len(specs)
            forgetting_row: list[float | None] = [None] * len(specs)
            cell_metrics: dict[str, dict[str, object]] = {}
            for seen_index in range(stage_index + 1):
                seen_task = task_order[seen_index]
                progress(
                    f"continual test stage={stage_index + 1}/{len(specs)} "
                    f"task={seen_task} examples={len(seen_test_sets[seen_task])}"
                )
                test_result = self.test(
                    seen_test_sets[seen_task],
                    split=f"{test_split}:after_{task_name}:{seen_task}",
                    reference_predictions=previous_predictions.get(seen_task),
                    phase="task_end_test",
                    task_stage=stage_index + 1,
                    current_task=task_name,
                    tested_task=seen_task,
                )
                metrics = dict(test_result.evaluation.get("metrics", {}))
                primary_score = _primary_metric_score(metrics)
                primary_score_row[seen_index] = primary_score
                previous_scores = [
                    float(row[seen_index])
                    for row in primary_score_rows
                    if row[seen_index] is not None
                ]
                forgetting_row[seen_index] = max(
                    max(previous_scores, default=primary_score) - primary_score,
                    0.0,
                )
                previous_predictions[seen_task] = list(test_result.evaluation.get("predictions", []))
                cell_metrics[seen_task] = metrics
                if seen_task == task_name:
                    self._build_probe_set(
                        task_name,
                        seen_test_sets[seen_task],
                        list(test_result.evaluation.get("predictions", [])),
                    )
            primary_score_rows.append(primary_score_row)
            forgetting_rows.append(forgetting_row)
            old_forgetting = [
                float(value)
                for value in forgetting_row[:stage_index]
                if value is not None
            ]
            stage_summaries.append(
                {
                    "stage": stage_index + 1,
                    "learned_task": task_name,
                    "harness_version": self.harness_version,
                    "average_old_task_forgetting": sum(old_forgetting) / len(old_forgetting) if old_forgetting else 0.0,
                    "memory": self.memory.summary(),
                    "test_metrics": cell_metrics,
                }
            )
            progress(
                f"stage done {stage_index + 1}/{len(specs)} task={task_name} "
                f"primary_score_row={primary_score_row} forgetting_row={forgetting_row}"
            )
            self._save_continual_checkpoint(
                signature=signature,
                payload={
                    "completed_stages": stage_index + 1,
                    "primary_score_rows": primary_score_rows,
                    "forgetting_rows": forgetting_rows,
                    "stage_summaries": stage_summaries,
                    "previous_predictions": previous_predictions,
                    "generated_candidates": all_candidates,
                    "candidate_evaluations": all_candidate_evaluations,
                    "accepted_candidate": accepted_candidate,
                    "last_evaluation": last_train_result.evaluation if last_train_result else None,
                },
            )
            progress(
                f"checkpoint continual saved completed_stages={stage_index + 1}/{len(specs)}"
            )

        continual_matrix = {
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "task_order": task_order,
            "primary_scores": primary_score_rows,
            "primary_metric_names": {
                task_name: next(
                    (
                        str(stage["test_metrics"][task_name].get("primary_metric_name"))
                        for stage in reversed(stage_summaries)
                        if isinstance(stage.get("test_metrics"), dict)
                        and isinstance(stage["test_metrics"].get(task_name), dict)
                    ),
                    "unknown",
                )
                for task_name in task_order
            },
            "forgetting": forgetting_rows,
            "stages": stage_summaries,
        }
        self.evaluation_recorder.record_continual_matrix(continual_matrix)
        progress(f"continual train done matrix_path={self.evaluation_recorder.record_dir / 'continual_matrix.json'}")
        empty_evaluation = self.evaluator.evaluate([], [], split=train_split)
        if last_train_result is None and isinstance(continual_checkpoint.get("last_evaluation"), dict):
            last_evaluation = dict(continual_checkpoint["last_evaluation"])
        else:
            last_evaluation = last_train_result.evaluation if last_train_result else empty_evaluation
        return PipelineResult(
            evaluation=last_evaluation,
            task_interface_config=self.task_interface_config,
            router_config=self.router_config,
            generated_candidates=all_candidates,
            candidate_evaluations=all_candidate_evaluations,
            current_validation=last_train_result.current_validation if last_train_result else None,
            accepted_candidate=accepted_candidate,
            optimizer_history=list(self.optimizer.history),
            continual_matrix=continual_matrix,
            memory_summary=self.memory.summary(),
        )

    def test(
        self,
        test_data: str | Path | Iterable[dict[str, Any]],
        *,
        split: str = "test",
        reference_predictions: list[dict[str, object]] | None = None,
        phase: str = "test",
        task_stage: int | None = None,
        current_task: str | None = None,
        tested_task: str | None = None,
        record_task_test: bool = True,
    ) -> PipelineResult:
        examples = self.dataset_loader.load(test_data)
        signature = _test_checkpoint_signature(split=split, examples=examples)
        test_checkpoint = self._matching_test_checkpoint(split, signature)
        if test_checkpoint.get("completed") and isinstance(test_checkpoint.get("result"), dict):
            progress(f"checkpoint test hit split={split} examples={len(examples)}")
            return PipelineResult(
                evaluation=dict(test_checkpoint["result"]),
                task_interface_config=self.task_interface_config,
                router_config=self.router_config,
                optimizer_history=list(self.optimizer.history),
            )
        progress(f"test start split={split} examples={len(examples)} memory_mode=read_only")
        predictions = self._predict(examples, progress_label=split)
        result = self.evaluator.evaluate(
            examples,
            predictions,
            split=split,
            reference_predictions=reference_predictions,
        )
        if record_task_test:
            self._record_task_test_result(
                phase=phase,
                split=split,
                result=result,
                task_stage=task_stage,
                current_task=current_task,
                tested_task=tested_task,
            )
        progress(
            f"test done split={split} primary_metric={result['metrics'].get('primary_metric_name')} "
            f"primary_score={_primary_metric_score(result['metrics']):.4f}"
        )
        self._save_test_checkpoint(split, signature=signature, result=result)
        return PipelineResult(
            evaluation=result,
            task_interface_config=self.task_interface_config,
            router_config=self.router_config,
            optimizer_history=list(self.optimizer.history),
        )

    def _maybe_run_forgetting_probe_after_batch(self) -> None:
        if not self.forgetting_probe.enabled:
            return
        interval = max(int(self.forgetting_probe.interval_batches), 1)
        if self.global_batch_index % interval != 0:
            return
        context = self._active_continual_context or {}
        historical_tasks = list(context.get("historical_tasks", []))
        if not historical_tasks:
            progress(
                f"forgetting probe skipped global_batch_index={self.global_batch_index} "
                "reason=no_historical_tasks"
            )
            return
        self._run_forgetting_probe(
            current_task=str(context.get("current_task") or ""),
            task_stage=int(context.get("task_stage", 0) or 0),
            historical_tasks=[str(task) for task in historical_tasks],
        )

    def _run_forgetting_probe(
        self,
        *,
        current_task: str,
        task_stage: int,
        historical_tasks: list[str],
    ) -> None:
        progress(
            f"forgetting probe start global_batch_index={self.global_batch_index} "
            f"historical_tasks={historical_tasks}"
        )
        probe_results: dict[str, dict[str, object]] = {}
        total_forget_count = 0
        total_recovered_count = 0
        for task_name in historical_tasks:
            probe_set = self.historical_task_probe_sets.get(task_name, {})
            sample_ids = [str(item) for item in probe_set.get("sample_ids", [])] if isinstance(probe_set, dict) else []
            test_examples = self.historical_task_test_sets.get(task_name, [])
            if not sample_ids or not test_examples:
                progress(f"forgetting probe skip task={task_name} reason=missing_probe_set_or_test_examples")
                continue
            sample_id_set = set(sample_ids)
            probe_examples = [
                example
                for example in test_examples
                if str(example.get("task_id", "")) in sample_id_set
            ]
            probe_examples.sort(key=lambda item: sample_ids.index(str(item.get("task_id", ""))))
            reference_predictions = self.historical_task_reference_predictions.get(task_name, [])
            predictions = self._predict(
                probe_examples,
                progress_label=f"forgetting_probe:{task_name}:batch_{self.global_batch_index}",
            )
            result = self.evaluator.evaluate(
                probe_examples,
                predictions,
                split=f"{self.forgetting_probe.split}:forgetting_probe:{task_name}",
                reference_predictions=reference_predictions,
            )
            metrics = dict(result.get("metrics", {}))
            total_forget_count += int(metrics.get("forget_count", 0) or 0)
            total_recovered_count += int(metrics.get("recovered_count", 0) or 0)
            reference_correct = int(metrics.get("reference_correct", 0) or 0)
            sample_composition = dict(probe_set.get("sample_composition", {})) if isinstance(probe_set, dict) else {}
            task_payload: dict[str, object] = {
                "sample_total": int(metrics.get("total", len(probe_examples)) or 0),
                "sample_ids": sample_ids,
                "sample_composition": sample_composition,
                "reference_correct": reference_correct,
                "reference_wrong": max(int(metrics.get("total", len(probe_examples)) or 0) - reference_correct, 0),
                "primary_metric_name": metrics.get("primary_metric_name"),
                "reference_primary_score": metrics.get("reference_primary_score"),
                "current_primary_score": metrics.get("primary_score"),
                "primary_score_delta": metrics.get("primary_score_delta"),
                "forget_count": metrics.get("forget_count", 0),
                "forget_rate": metrics.get("forget_rate", 0.0),
                "recovered_count": metrics.get("recovered_count", 0),
                "format_compliance_rate": metrics.get("format_compliance_rate", 0.0),
            }
            if self.forgetting_probe.record_predictions:
                task_payload["predictions"] = predictions
            probe_results[task_name] = task_payload
            progress(
                f"forgetting probe task done task={task_name} "
                f"primary_score={_primary_metric_score(metrics):.4f} "
                f"forget_rate={float(metrics.get('forget_rate', 0.0) or 0.0):.4f}"
            )
        if not probe_results:
            return
        worst_task = max(
            probe_results,
            key=lambda task_name: float(probe_results[task_name].get("forget_rate", 0.0) or 0.0),
        )
        self.evaluation_recorder.record_forgetting_probe(
            {
                "global_batch_index": self.global_batch_index,
                "task_stage": task_stage,
                "current_task": current_task,
                "harness_version": self.harness_version,
                "component_versions": self._component_version_snapshot(),
                "memory_summary": self.memory.summary(),
                "probe_config": self._probe_config_payload(),
                "probe_results": probe_results,
                "summary": {
                    "old_task_primary_scores": {
                        task_name: result.get("current_primary_score")
                        for task_name, result in probe_results.items()
                    },
                    "average_old_task_forget_rate": sum(
                        float(result.get("forget_rate", 0.0) or 0.0)
                        for result in probe_results.values()
                    ) / len(probe_results),
                    "total_forget_count": total_forget_count,
                    "total_recovered_count": total_recovered_count,
                    "worst_task": worst_task,
                },
            }
        )
        progress(f"forgetting probe recorded tasks={len(probe_results)}")

    def _build_probe_set(
        self,
        task_name: str,
        test_examples: list[dict[str, Any]],
        reference_predictions: list[dict[str, object]],
    ) -> None:
        if (
            self.forgetting_probe.reuse_probe_set
            and task_name in self.historical_task_probe_sets
            and task_name in self.historical_task_reference_predictions
        ):
            return
        by_id = {str(prediction.get("task_id", "")): prediction for prediction in reference_predictions}
        correct_ids: list[str] = []
        wrong_ids: list[str] = []
        for example in test_examples:
            task_id = str(example.get("task_id", ""))
            prediction = by_id.get(task_id)
            if prediction is None:
                continue
            if answer_matches(prediction.get("answer"), example.get("answer"), example=example):
                correct_ids.append(task_id)
            else:
                wrong_ids.append(task_id)
        sample_total = max(int(self.forgetting_probe.samples_per_task), 0)
        target_correct = int(round(sample_total * float(self.forgetting_probe.target_correct_ratio)))
        target_wrong = max(sample_total - target_correct, 0)
        rng = random.Random(int(self.forgetting_probe.seed) + _stable_task_seed(task_name))
        rng.shuffle(correct_ids)
        rng.shuffle(wrong_ids)
        selected_correct = correct_ids[: min(target_correct, len(correct_ids))]
        selected_wrong = wrong_ids[: min(target_wrong, len(wrong_ids))]
        remaining = max(sample_total - len(selected_correct) - len(selected_wrong), 0)
        if remaining:
            selected_correct_ids = set(selected_correct)
            selected_wrong_ids = set(selected_wrong)
            fill_pool = [
                task_id
                for task_id in correct_ids + wrong_ids
                if task_id not in selected_correct_ids and task_id not in selected_wrong_ids
            ]
            fill = fill_pool[:remaining]
            selected_correct.extend([task_id for task_id in fill if task_id in set(correct_ids)])
            selected_wrong.extend([task_id for task_id in fill if task_id in set(wrong_ids)])
        selected_ids = selected_correct + selected_wrong
        rng.shuffle(selected_ids)
        selected_id_set = set(selected_ids)
        self.historical_task_probe_sets[task_name] = {
            "sample_ids": selected_ids,
            "sample_composition": {
                "target_correct": target_correct,
                "target_wrong": target_wrong,
                "actual_correct": len(selected_correct),
                "actual_wrong": len(selected_wrong),
                "total": len(selected_ids),
            },
            "split": self.forgetting_probe.split,
        }
        self.historical_task_reference_predictions[task_name] = [
            dict(prediction)
            for prediction in reference_predictions
            if str(prediction.get("task_id", "")) in selected_id_set
        ]
        progress(
            f"forgetting probe set built task={task_name} total={len(selected_ids)} "
            f"correct={len(selected_correct)} wrong={len(selected_wrong)}"
        )

    def _record_task_test_result(
        self,
        *,
        phase: str,
        split: str,
        result: dict[str, object],
        task_stage: int | None = None,
        current_task: str | None = None,
        tested_task: str | None = None,
    ) -> None:
        self.evaluation_recorder.record_task_test(
            {
                "phase": phase,
                "global_batch_index": self.global_batch_index,
                "task_stage": task_stage,
                "current_task": current_task,
                "tested_task": tested_task,
                "split": split,
                "harness_version": self.harness_version,
                "component_versions": self._component_version_snapshot(),
                "memory_summary": self.memory.summary(),
                "metrics": compact_metrics(dict(result.get("metrics", {}))),
            }
        )

    def _component_version_snapshot(self) -> dict[str, dict[str, object]]:
        snapshot: dict[str, dict[str, object]] = {}
        for component in OPTIMIZABLE_COMPONENTS:
            existing = dict(self.component_versions.get(component, {}))
            version = int(existing.get("version", 0) or 0)
            try:
                path = self._component_artifact_path(component)
            except ValueError:
                path = None
            snapshot[component] = {
                "version": version,
                "artifact_path": str(path) if path is not None else existing.get("artifact_path", ""),
                "artifact_hash": _file_sha256(path) if path is not None else existing.get("artifact_hash", ""),
            }
        return snapshot

    def _probe_config_payload(self) -> dict[str, object]:
        return {
            "interval_batches": self.forgetting_probe.interval_batches,
            "samples_per_task": self.forgetting_probe.samples_per_task,
            "target_correct_ratio": self.forgetting_probe.target_correct_ratio,
            "seed": self.forgetting_probe.seed,
            "split": self.forgetting_probe.split,
            "historical_tasks_only": self.forgetting_probe.historical_tasks_only,
            "reuse_probe_set": self.forgetting_probe.reuse_probe_set,
            "record_predictions": self.forgetting_probe.record_predictions,
        }

    def _active_context_value(self, key: str) -> object | None:
        if not self._active_continual_context:
            return None
        return self._active_continual_context.get(key)

    def _prediction_cache_context(self, router: Router, task_interface: TaskInterface) -> dict[str, Any]:
        return {
            "schema": "hcl_read_only_prediction_cache_v1",
            "model": _model_cache_fingerprint(self.model),
            "selection_model": _model_cache_fingerprint(self.selection_model),
            "harness_version": self.harness_version,
            "component_versions": self._component_version_snapshot(),
            "task_interface_config": self.task_interface_config.to_dict(),
            "task_interface_artifact": _template_fingerprint(task_interface.structuring_template),
            "router_config": router.config.to_dict(),
            "router_artifacts": {
                "workflow": _template_fingerprint(router.workflow_template),
                "context": _template_fingerprint(router.context_template),
                "memory_selector": _template_fingerprint(router.memory_selector_template),
                "skill_selector": _template_fingerprint(router.skill_selector_template),
                "tool_selector": _template_fingerprint(router.tool_selector_template),
            },
            "memory_snapshot_hash": _stable_json_hash(self.memory.snapshot()),
        }

    def _prediction_cache_key(
        self,
        taskinterfacechunk: dict[str, Any],
        *,
        cache_context: dict[str, Any],
    ) -> str:
        payload = {
            **cache_context,
            "model_visible_input": TaskInterface.to_model_visible(taskinterfacechunk),
            "file_fingerprint": _file_context_fingerprint(taskinterfacechunk),
        }
        return _stable_json_hash(payload)

    def _predict(
        self,
        examples: list[dict[str, Any]],
        *,
        update_memory: bool = False,
        progress_label: str = "predict",
    ) -> list[dict[str, Any]]:
        task_interface = TaskInterface(self.task_interface_config, model=self.selection_model)
        router = Router(
            self.model,
            self.router_config,
            memory=self.memory,
            selection_model=self.selection_model,
        )
        return self._predict_with_router(
            examples,
            router,
            update_memory=update_memory,
            progress_label=progress_label,
        )

    def _predict_with_router(
        self,
        examples: list[dict[str, Any]],
        router: Router,
        *,
        task_interface: TaskInterface | None = None,
        update_memory: bool = False,
        progress_label: str = "predict",
    ) -> list[dict[str, Any]]:
        task_interface = task_interface or TaskInterface(
            self.task_interface_config,
            model=self.selection_model,
        )
        use_prediction_cache = bool(self.prediction_cache.enabled and not update_memory)
        cache_context = (
            self._prediction_cache_context(router, task_interface)
            if use_prediction_cache
            else None
        )
        total = len(examples)
        workers = 1 if update_memory else min(self.read_only_workers, max(total, 1))
        if workers > 1:
            progress(
                f"parallel prediction start phase={progress_label} examples={total} "
                f"workers={workers} memory_mode=read_only"
            )
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hcl-read-only") as executor:
                predictions = list(
                    executor.map(
                        lambda item: self._predict_one(
                            item[0],
                            item[1],
                            total=total,
                            router=router,
                            task_interface=task_interface,
                            cache_context=cache_context,
                            update_memory=False,
                            progress_label=progress_label,
                        ),
                        enumerate(examples),
                    )
                )
            progress(f"parallel prediction done phase={progress_label} examples={total} workers={workers}")
            return predictions

        return [
            self._predict_one(
                stream_position,
                example,
                total=total,
                router=router,
                task_interface=task_interface,
                cache_context=cache_context,
                update_memory=update_memory,
                progress_label=progress_label,
            )
            for stream_position, example in enumerate(examples)
        ]

    def _predict_one(
        self,
        stream_position: int,
        example: dict[str, Any],
        *,
        total: int,
        router: Router,
        task_interface: TaskInterface,
        cache_context: dict[str, Any] | None,
        update_memory: bool,
        progress_label: str,
    ) -> dict[str, Any]:
        item_index = stream_position + 1
        show_item = report_item(item_index, total)
        if show_item:
            progress(
                f"sample start phase={progress_label} item={item_index}/{total} "
                f"task={example.get('task_name', '')} task_id={example.get('task_id', '')} "
                f"memory_mode={'read_write' if update_memory else 'read_only'}"
            )
        item_start = perf_counter()
        taskinterfacechunk: dict[str, Any] | None = None
        cache_key = ""
        try:
            taskinterfacechunk = task_interface.build_taskinterfacechunk(example, stream_position)
            cache_key = (
                self._prediction_cache_key(taskinterfacechunk, cache_context=cache_context)
                if cache_context is not None
                else ""
            )
            prediction = (
                self.prediction_cache.get(cache_key, task_id=str(taskinterfacechunk.get("task_id", "")))
                if cache_key
                else None
            )
            if prediction is not None:
                progress(
                    f"prediction cache hit phase={progress_label} "
                    f"task_id={taskinterfacechunk.get('task_id', '')} key={cache_key[:12]}"
                )
            else:
                prediction = router.run(
                    taskinterfacechunk,
                    self.task_interface_config,
                    allow_final_generation_cache=not update_memory,
                )
                if cache_key:
                    self.prediction_cache.put(cache_key, prediction, phase=progress_label)
                    progress(
                        f"prediction cache store phase={progress_label} "
                        f"task_id={taskinterfacechunk.get('task_id', '')} key={cache_key[:12]}"
                    )
        except RuntimeError as exc:
            if not _is_content_risk_rejection(exc):
                raise
            task_id = str(
                taskinterfacechunk.get("task_id", "")
                if taskinterfacechunk is not None
                else example.get("task_id", "")
            )
            task_interface_mode = (
                taskinterfacechunk.get("metadata", {}).get("task_interface_mode", "content_risk_rejected")
                if taskinterfacechunk is not None
                else "content_risk_rejected"
            )
            prediction = {
                "task_id": task_id,
                "answer": "",
                "trace": {
                    "task_id": task_id,
                    "task_interface_mode": task_interface_mode,
                    "selected_memory": [],
                    "api_rejection": {
                        "type": "content_risk_rejection",
                        "sample_skipped": True,
                        "memory_update_skipped": True,
                    },
                },
            }
            if cache_key:
                self.prediction_cache.put(cache_key, prediction, phase=progress_label)
            progress(
                f"sample skipped phase={progress_label} task_id={task_id} "
                "reason=content_risk_rejection memory_update=false"
            )
        rejected_for_content_risk = bool(
            prediction.get("trace", {}).get("api_rejection", {}).get("type")
            == "content_risk_rejection"
        )
        if update_memory and not rejected_for_content_risk:
            self.memory.record_training_interaction(
                str(example.get("task_name", "")),
                example,
                prediction,
            )
            if show_item:
                progress(
                    f"memory raw updated task={example.get('task_name', '')} "
                    f"task_id={example.get('task_id', '')} summary={self.memory.summary()}"
                )
        if show_item:
            selected_count = len(prediction.get("trace", {}).get("selected_memory", []))
            progress(
                f"sample done phase={progress_label} item={item_index}/{total} "
                f"interface_mode={prediction.get('trace', {}).get('task_interface_mode', 'unknown')} "
                f"selected_memory={selected_count} seconds={perf_counter() - item_start:.1f}"
            )
        return prediction

def _template_fingerprint(template: Any) -> dict[str, object]:
    return {
        "template_id": getattr(template, "template_id", ""),
        "description": getattr(template, "description", ""),
        "sections": getattr(template, "sections", []),
    }


def _is_content_risk_rejection(exc: BaseException) -> bool:
    return "content exists risk" in str(exc).lower()


def _model_cache_fingerprint(model: Any) -> dict[str, object]:
    config = getattr(model, "config", None)
    return {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "name": str(getattr(config, "name", "")),
        "backend": str(getattr(config, "backend", "")),
        "path": str(getattr(config, "path", "")),
        "base_url": str(getattr(config, "base_url", "")),
        "max_new_tokens": str(getattr(config, "max_new_tokens", "")),
        "answer_max_new_tokens": str(getattr(config, "answer_max_new_tokens", "")),
        "temperature": str(getattr(config, "temperature", "")),
        "thinking": getattr(config, "thinking", None),
        "reasoning_effort": str(getattr(config, "reasoning_effort", "")),
        "enable_thinking": str(getattr(config, "enable_thinking", "")),
        "cleanup_output": str(getattr(config, "cleanup_output", "")),
        "system_prompt_hash": _stable_json_hash(str(getattr(config, "system_prompt", ""))),
    }


def _file_context_fingerprint(taskinterfacechunk: dict[str, Any]) -> list[dict[str, object]]:
    task_context = taskinterfacechunk.get("task_context")
    files = task_context.get("files", []) if isinstance(task_context, dict) else []
    fingerprints: list[dict[str, object]] = []
    if not isinstance(files, list):
        return fingerprints
    for item in files:
        if not isinstance(item, dict):
            continue
        path = Path(str(item.get("path", ""))) if item.get("path") else None
        stat_payload: dict[str, object] = {}
        if path is not None and path.exists():
            try:
                stat = path.stat()
                stat_payload = {
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            except OSError:
                stat_payload = {"stat_error": True}
        fingerprints.append(
            {
                "type": str(item.get("type", "")),
                "path_hash": _stable_json_hash(str(path) if path is not None else ""),
                "stat": stat_payload,
            }
        )
    return fingerprints


def _file_sha256(path: Path | None) -> str:
    if path is None or not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_task_seed(task_name: str) -> int:
    digest = hashlib.sha256(task_name.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _merge_dict(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_dict(target[key], value)
        else:
            target[key] = _json_clone(value)


def _task_id_sequence(examples: list[dict[str, Any]]) -> list[str]:
    return [str(example.get("task_id") or index) for index, example in enumerate(examples)]


def _train_checkpoint_signature(
    *,
    split: str,
    examples: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    batch_count: int,
    batchsize: int | None,
) -> dict[str, object]:
    return {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "split": split,
        "task_ids": _task_id_sequence(examples),
        "validation_task_ids": _task_id_sequence(val_examples),
        "batch_count": int(batch_count),
        "batchsize": int(batchsize) if batchsize is not None else None,
    }


def _continual_checkpoint_signature(
    *,
    task_stream_path: str | Path,
    task_order: list[str],
    train_split: str,
    val_split: str,
    test_split: str,
    train_limit_per_task: int | None,
    validation_limit_per_task: int | None,
    test_limit_per_task: int | None,
    batchsize: int | None,
) -> dict[str, object]:
    return {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "task_stream_path": str(Path(task_stream_path).resolve()),
        "task_order": list(task_order),
        "train_split": train_split,
        "val_split": val_split,
        "test_split": test_split,
        "train_limit_per_task": train_limit_per_task,
        "validation_limit_per_task": validation_limit_per_task,
        "test_limit_per_task": test_limit_per_task,
        "batchsize": batchsize,
    }


def _test_checkpoint_signature(
    *,
    split: str,
    examples: list[dict[str, Any]],
) -> dict[str, object]:
    return {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "split": split,
        "task_ids": _task_id_sequence(examples),
    }


def _batch_examples(examples: list[dict[str, Any]], batchsize: int | None) -> Iterable[list[dict[str, Any]]]:
    if batchsize is None or batchsize <= 0:
        yield examples
        return
    for start in range(0, len(examples), batchsize):
        yield examples[start : start + batchsize]


def _historical_anchors_for_train(
    anchor_examples: list[dict[str, Any]],
    train_examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep anchors from earlier tasks, never the task currently being trained.

    On checkpoint resume, memory already contains anchors selected by completed
    batches of the current task.  Treating those as historical changes the gate
    from the uninterrupted-run semantics and evaluates duplicate current-task
    data.  Task-less inputs retain all anchors because no safe exclusion key is
    available.
    """
    current_task_names = {
        str(example.get("task_name") or "").strip()
        for example in train_examples
        if str(example.get("task_name") or "").strip()
    }
    if not current_task_names:
        return list(anchor_examples)
    return [
        example
        for example in anchor_examples
        if str(example.get("task_name") or "").strip() not in current_task_names
    ]


def _historical_anchor_metric_summary(
    per_task: dict[str, dict[str, object]],
    *,
    fallback: dict[str, object],
) -> dict[str, object]:
    """Summarize retention counters while keeping task-native scores separate."""
    if not per_task:
        return dict(fallback)
    task_scores: dict[str, float] = {}
    for task_name, metrics in per_task.items():
        value = metrics.get("primary_score")
        if value is not None:
            task_scores[task_name] = float(value)
    total = sum(int(metrics.get("total", 0) or 0) for metrics in per_task.values())
    format_count = sum(
        int(metrics.get("format_compliant_count", 0) or 0)
        for metrics in per_task.values()
    )
    summary: dict[str, object] = {
        "primary_metric_name": "historical_tasks_reported_separately",
        "primary_score": None,
        "task_primary_scores": task_scores,
        "task_count": len(per_task),
        "correct": sum(int(metrics.get("correct", 0) or 0) for metrics in per_task.values()),
        "total": total,
        "forget_count": sum(int(metrics.get("forget_count", 0) or 0) for metrics in per_task.values()),
        "reference_correct": sum(
            int(metrics.get("reference_correct", 0) or 0)
            for metrics in per_task.values()
        ),
        "recovered_count": sum(
            int(metrics.get("recovered_count", 0) or 0)
            for metrics in per_task.values()
        ),
        "format_compliant_count": format_count,
        "format_compliance_rate": format_count / total if total else 0.0,
    }
    reference_correct = int(summary["reference_correct"])
    summary["forget_rate"] = (
        int(summary["forget_count"]) / reference_correct
        if reference_correct
        else 0.0
    )
    return summary


def _primary_metric_score(metrics: object) -> float:
    if not isinstance(metrics, dict):
        return 0.0
    value = metrics.get("primary_score")
    return float(value) if value is not None else 0.0


def _unique_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for index, example in enumerate(examples):
        key = str(example.get("task_id") or index)
        unique[key] = example
    return list(unique.values())


def _deterministic_screening_sample(
    examples: list[dict[str, Any]],
    count: int,
    *,
    seed: str,
) -> list[dict[str, Any]]:
    if count >= len(examples):
        return list(examples)
    ranked = sorted(
        enumerate(examples),
        key=lambda item: hashlib.sha256(
            f"{seed}:{item[1].get('task_id', item[0])}".encode("utf-8")
        ).hexdigest(),
    )
    selected_indexes = {index for index, _ in ranked[: max(count, 0)]}
    return [example for index, example in enumerate(examples) if index in selected_indexes]


def _screening_only_result(
    result: dict[str, object],
    diagnostics: list[dict[str, object]],
) -> dict[str, object]:
    copied = _json_clone(result)
    metrics = copied.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
        copied["metrics"] = metrics
    metrics["screening_only"] = True
    copied["evaluation_stage"] = "screening_only"
    copied["screening_diagnostics"] = diagnostics
    return copied

