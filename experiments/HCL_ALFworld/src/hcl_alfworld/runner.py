from __future__ import annotations

import csv
import copy
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .config import resolve_project_path
from .embedding import make_embedder
from .environment import ALFWorldEpisodeEnv, list_split_games, read_task_type
from .harness import make_continual_harness
from .llm import make_llm
from .metrics import continual_metrics
from .policies import Policy, make_policy
from .run_artifacts import RunArtifacts
from .schemas import EpisodeRecord, EpisodeResult, HarnessState, StepRecord
from .sequence import load_sequence
from .task_interface import TaskInterface


@dataclass
class _ReadOnlyRuntime:
    llm: Any
    interface: TaskInterface
    policy: Policy
    harness: Any
    harness_version: int


@dataclass
class _ReadOnlyOutcome:
    result: EpisodeResult
    state_before: Dict[str, Any]
    llm_usage: Dict[str, Any]
    requires_runtime_record: bool = True
    requires_usage_merge: bool = True


class ExperimentRunner:
    def __init__(
        self,
        config: Dict[str, Any],
        resume_dir: Optional[str | Path] = None,
        raw_memory_retrieval_override: Optional[bool] = None,
    ):
        if raw_memory_retrieval_override is not None and resume_dir is None:
            raise ValueError("Raw-memory retrieval overrides are only supported when resuming.")
        configured_method = str(config.get("continual_method", "hcl")).lower()
        if raw_memory_retrieval_override is not None and configured_method != "hcl":
            raise ValueError("Raw-memory retrieval overrides apply only to the HCL method.")
        self.config = copy.deepcopy(config)
        if raw_memory_retrieval_override is not None:
            self.config["harness"]["raw_memory_retrieval_enabled"] = bool(
                raw_memory_retrieval_override
            )
        self.raw_memory_retrieval_override = raw_memory_retrieval_override
        self.seed = int(config.get("seed", 42))
        self.data_dir = resolve_project_path(config, "data_dir")
        self.base_output_dir = resolve_project_path(config, "output_dir")
        self.artifacts = RunArtifacts(self.base_output_dir, config, resume_dir=resume_dir)
        self.output_dir = self.artifacts.run_dir
        self.agent_config = config["agent"]
        self.continual_method = configured_method
        self.llm = make_llm(self.agent_config)
        self.interface = TaskInterface(
            self.llm,
            episode_progress_enabled=bool(
                self.agent_config.get("episode_progress_enabled", True)
            ),
        )
        self.eval_policy = make_policy(
            self.agent_config["policy"], self.agent_config, self.seed, self.llm
        )
        learning_name = self.agent_config.get("learning_policy", self.agent_config["policy"])
        self.learning_policy = make_policy(learning_name, self.agent_config, self.seed, self.llm)
        self.embedding_config = dict(self.config.get("embedding", {}))
        self.embedder = (
            make_embedder(self.embedding_config)
            if self.continual_method in {"rag"}
            else None
        )
        method_config = (
            self.config[self.continual_method]
            if self.continual_method in {"rag"}
            else self.config["harness"]
        )
        self.harness = make_continual_harness(
            self.continual_method,
            self.output_dir / "harness",
            method_config,
            self.llm,
            embedder=self.embedder,
            seed=self.seed,
        )
        self.runtime_stats: Dict[str, Any] = {
            "episodes": 0,
            "steps": 0,
            "learning_episodes": 0,
            "evaluation_episodes": 0,
            "decisions": {"commit": 0, "partial_merge": 0, "reject": 0, "rollback": 0},
            "anchor_retention_sum": 0.0,
            "anchor_evaluations": 0,
        }
        self.prior_llm_usage: Dict[str, Any] = {}
        self.parallel_llm_usage: Dict[str, Any] = {}
        self._worker_local = threading.local()
        self.elapsed_before = 0.0
        self.resume_requested = resume_dir is not None
        self.artifacts.event(
            "run_initialized",
            "Run directory initialized.",
            run_dir=str(self.output_dir),
            resume=self.resume_requested,
        )
        if raw_memory_retrieval_override is not None:
            self.artifacts.event(
                "resume_protocol_override",
                "Resume intentionally changed online raw-memory retrieval.",
                raw_memory_retrieval_enabled=bool(raw_memory_retrieval_override),
            )

    def run_episode(
        self,
        game_file: str,
        split: str,
        policy: Policy,
        learn: bool,
        training_episode: bool = False,
        runtime: Optional[_ReadOnlyRuntime] = None,
        record_runtime: bool = True,
    ) -> EpisodeResult:
        active_interface = runtime.interface if runtime is not None else self.interface
        active_harness = runtime.harness if runtime is not None else self.harness
        game_path = Path(game_file)
        if not game_path.is_absolute():
            game_path = self.data_dir / game_path
        game_file = str(game_path.resolve())
        episode_id = f"episode-{uuid.uuid4().hex[:12]}"
        env_config = self.config["environment"]
        max_steps = int(env_config.get("max_steps", 50))
        task_type_hint = read_task_type(game_file)
        self.artifacts.event(
            "episode_started",
            "Episode environment initialization started.",
            episode_id=episode_id,
            split=split,
            game_file=game_file,
            training_episode=training_episode,
            learn=learn,
            harness_version=active_harness.state.version,
        )
        env = ALFWorldEpisodeEnv(
            game_file,
            self.data_dir,
            max_steps=max_steps,
            domain_randomization=bool(env_config.get("domain_randomization", False)),
        )
        steps: List[StepRecord] = []
        routing_traces: List[Dict[str, Any]] = []
        success = False
        score = 0.0
        final_context = None
        episode_context = None
        last_semantic_route_step = -2
        goal = ""
        task_type = ""
        try:
            observations, info = env.reset()
            initial_observation = observations[0]
            self.artifacts.event(
                "environment_ready",
                "Episode environment reset completed.",
                episode_id=episode_id,
                task_type_hint=task_type_hint,
                max_steps=max_steps,
            )
            for step_index in range(max_steps):
                admissible = list(info.get("admissible_commands", [[]])[0])
                history_limit = int(
                    self.agent_config.get("episode_history_max_steps", max_steps)
                )
                visible_steps = steps[-history_limit:] if history_limit > 0 else []
                visible_start = len(steps) - len(visible_steps)
                episode_history = [
                    {
                        "step": visible_start + history_index,
                        "observation": history_step.observation,
                        "action": history_step.action,
                        "next_observation": history_step.next_observation,
                        "reward": history_step.reward,
                        "state_assessment": str(
                            routing_traces[visible_start + history_index]
                            .get("action_policy", {})
                            .get("state_assessment", "")
                        ),
                        "next_step_plan": str(
                            routing_traces[visible_start + history_index]
                            .get("action_policy", {})
                            .get("next_step_plan", "")
                        ),
                        "action_rationale": str(
                            routing_traces[visible_start + history_index]
                            .get("action_policy", {})
                            .get("rationale", "")
                        ),
                    }
                    for history_index, history_step in enumerate(visible_steps)
                ]
                self.artifacts.event(
                    "step_started",
                    "LLM step processing started.",
                    episode_id=episode_id,
                    step=step_index,
                    admissible_command_count=len(admissible),
                )
                task = active_interface.structure(
                    interaction_id=episode_id,
                    observation=observations[0],
                    initial_observation=initial_observation,
                    admissible_commands=admissible,
                    game_file=game_file,
                    step=step_index,
                    max_steps=max_steps,
                    known_goal=goal or None,
                    known_task_type=task_type or None,
                    known_progress=(
                        episode_context.task.progress
                        if episode_context is not None
                        else None
                    ),
                    task_type_hint=task_type_hint,
                    episode_history=episode_history,
                )
                goal = task.goal
                task_type = task.task_type
                if task.interface_trace.get("fallback"):
                    self.artifacts.event(
                        "component_fallback",
                        "Task Interface exhausted JSON retries; deterministic episode "
                        "state fallback was used.",
                        episode_id=episode_id,
                        step=step_index,
                        component="task_interface",
                        error=task.interface_trace.get("fallback_error", ""),
                        goal_source=task.interface_trace.get("goal_source"),
                    )
                self.artifacts.event(
                    "component_completed",
                    "Task Interface completed.",
                    episode_id=episode_id,
                    step=step_index,
                    component="task_interface",
                    task_type=task_type,
                )
                reroute_reasons = active_harness.router.reroute_reasons(
                    task,
                    episode_context.task if episode_context is not None else None,
                    steps,
                )
                reroute_allowed = (
                    episode_context is None
                    or active_harness.router.should_refresh_semantic_context(
                        reroute_reasons,
                        step_index,
                        last_semantic_route_step,
                    )
                )
                if reroute_allowed:
                    context = active_harness.route(task, reroute_reasons)
                    episode_context = context
                    if context.trace.get("llm_called"):
                        last_semantic_route_step = step_index
                else:
                    context = active_harness.reuse_route(task, episode_context)
                    episode_context = context
                self.artifacts.event(
                    "component_completed",
                    "Router completed.",
                    episode_id=episode_id,
                    step=step_index,
                    component="router",
                    selected_memory_count=len(context.memories),
                    selected_strategy_count=len(context.strategies),
                    selected_capability_count=len(context.capabilities),
                    routing_mode=context.trace.get("mode"),
                    llm_called=bool(context.trace.get("llm_called")),
                    trigger_reasons=context.trace.get("trigger_reasons", []),
                )
                action = policy.select_action(context, info)
                self.artifacts.event(
                    "action_selected",
                    "Action Policy selected an admissible action.",
                    episode_id=episode_id,
                    step=step_index,
                    action=action,
                )
                routing_traces.append(
                    {
                        "step": step_index,
                        "task_interface": task.interface_trace,
                        "router": context.trace,
                        "action_policy": dict(getattr(policy, "last_trace", {})),
                    }
                )
                next_observations, rewards, dones, next_info = env.step(action)
                reward = float(rewards[0])
                steps.append(
                    StepRecord(
                        observation=observations[0],
                        action=action,
                        next_observation=next_observations[0],
                        admissible_commands=admissible,
                        reward=reward,
                    )
                )
                observations, info = next_observations, next_info
                score = max(score, reward)
                success = bool(info.get("won", [False])[0])
                final_context = context
                if bool(dones[0]) or success:
                    break
        finally:
            env.close()

        record = EpisodeRecord(
            episode_id=episode_id,
            task_type=task_type,
            goal=goal,
            game_file=game_file,
            split=split,
            steps=steps,
            success=success,
            score=1.0 if success else score,
            failure_reason="" if success else f"No success within {len(steps)} steps.",
            harness_traces=routing_traces,
        )
        result = EpisodeResult(record=record, routing_traces=routing_traces)
        if learn and final_context is not None:
            result.candidate, result.evaluation = active_harness.learn(record, final_context)
            optimizer_fallback = next(
                (
                    trace["optimizer"]
                    for trace in reversed(record.harness_traces)
                    if isinstance(trace.get("optimizer"), dict)
                    and trace["optimizer"].get("fallback") is True
                ),
                None,
            )
            if optimizer_fallback is not None:
                self.artifacts.event(
                    "component_fallback",
                    "Optimizer exhausted JSON retries; Harness update was safely skipped.",
                    episode_id=episode_id,
                    component="optimizer",
                    error=optimizer_fallback.get("error", ""),
                    fallback_mode=optimizer_fallback.get("mode"),
                    harness_version=active_harness.state.version,
                )
            decision = result.evaluation.decision
            self.runtime_stats["decisions"][decision] += 1
            self.runtime_stats["anchor_retention_sum"] += result.evaluation.anchor_retention
            self.runtime_stats["anchor_evaluations"] += 1
        if record_runtime:
            self._record_episode_runtime(result, training_episode)
        return result

    def _record_episode_runtime(
        self, result: EpisodeResult, training_episode: bool = False
    ) -> None:
        self.runtime_stats["episodes"] += 1
        self.runtime_stats["steps"] += len(result.record.steps)
        category = "learning_episodes" if training_episode else "evaluation_episodes"
        self.runtime_stats[category] += 1

    def _evaluation_workers(self, remaining: int) -> int:
        configured = int(self.config.get("execution", {}).get("evaluation_workers", 1))
        return min(max(configured, 1), max(remaining, 1))

    def _read_only_runtime(self, state_snapshot: Dict[str, Any]) -> _ReadOnlyRuntime:
        version = int(state_snapshot.get("version", 0))
        runtime = getattr(self._worker_local, "runtime", None)
        if runtime is None:
            llm = make_llm(self.agent_config)
            embedder = (
                make_embedder(self.embedding_config)
                if self.continual_method in {"rag"}
                else None
            )
            method_config = (
                self.config[self.continual_method]
                if self.continual_method in {"rag"}
                else self.config["harness"]
            )
            runtime = _ReadOnlyRuntime(
                llm=llm,
                interface=TaskInterface(
                    llm,
                    episode_progress_enabled=bool(
                        self.agent_config.get("episode_progress_enabled", True)
                    ),
                ),
                policy=make_policy(
                    self.agent_config["policy"], self.agent_config, self.seed, llm
                ),
                harness=make_continual_harness(
                    self.continual_method,
                    self.output_dir / "harness",
                    method_config,
                    llm,
                    embedder=embedder,
                    seed=self.seed,
                ),
                harness_version=-1,
            )
            self._worker_local.runtime = runtime
        if runtime.harness_version != version:
            runtime.harness.state = HarnessState.from_dict(state_snapshot)
            runtime.harness_version = version
        return runtime

    def _run_read_only_episode(
        self, game_file: str, split: str, state_snapshot: Dict[str, Any]
    ) -> _ReadOnlyOutcome:
        print(f"[eval_worker start] {game_file}", flush=True)
        runtime = self._read_only_runtime(state_snapshot)
        usage_before = runtime.llm.usage_snapshot()
        result = self.run_episode(
            game_file,
            split,
            runtime.policy,
            learn=False,
            training_episode=False,
            runtime=runtime,
            record_runtime=False,
        )
        print(
            f"[eval_worker done] success={result.record.success} "
            f"steps={len(result.record.steps)} {game_file}",
            flush=True,
        )
        return _ReadOnlyOutcome(
            result=result,
            state_before=state_snapshot,
            llm_usage=self._usage_delta(usage_before, runtime.llm.usage_snapshot()),
        )

    def _ordered_read_only_outcomes(
        self, game_files: List[str], split: str
    ) -> Iterator[_ReadOnlyOutcome]:
        state_snapshot = self.harness.state.to_dict()
        workers = self._evaluation_workers(len(game_files))
        if workers == 1:
            for game_file in game_files:
                usage_before = self.llm.usage_snapshot()
                result = self.run_episode(
                    game_file,
                    split,
                    self.eval_policy,
                    learn=False,
                    training_episode=False,
                )
                yield _ReadOnlyOutcome(
                    result=result,
                    state_before=state_snapshot,
                    llm_usage=self._usage_delta(
                        usage_before, self.llm.usage_snapshot()
                    ),
                    requires_runtime_record=False,
                    requires_usage_merge=False,
                )
            return
        self.artifacts.event(
            "parallel_evaluation_started",
            "Read-only evaluation episodes started in parallel.",
            workers=workers,
            episodes=len(game_files),
            harness_version=state_snapshot.get("version", 0),
        )
        print(
            f"[parallel_eval] workers={workers} episodes={len(game_files)} "
            f"harness_version={state_snapshot.get('version', 0)}",
            flush=True,
        )
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="hcl-eval"
        ) as executor:
            yield from executor.map(
                lambda game_file: self._run_read_only_episode(
                    game_file, split, state_snapshot
                ),
                game_files,
            )
        self.artifacts.event(
            "parallel_evaluation_completed",
            "Read-only parallel evaluation batch completed.",
            workers=workers,
            episodes=len(game_files),
            harness_version=state_snapshot.get("version", 0),
        )

    @staticmethod
    def _state_summary(state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "version": int(state.get("version", 0)),
            "raw_memory_count": len(state.get("raw_memory", [])),
            "abstract_memory_count": len(state.get("abstract_memory", [])),
            "capability_count": len(state.get("capabilities", [])),
            "anchor_count": len(state.get("anchors", [])),
        }

    @staticmethod
    def _entity_changes(
        before: List[Dict[str, Any]], after: List[Dict[str, Any]], identifier: str
    ) -> Dict[str, Any]:
        before_by_id = {str(item[identifier]): item for item in before}
        after_by_id = {str(item[identifier]): item for item in after}
        added_ids = sorted(set(after_by_id) - set(before_by_id))
        removed_ids = sorted(set(before_by_id) - set(after_by_id))
        changed_ids = sorted(
            item_id
            for item_id in set(before_by_id) & set(after_by_id)
            if before_by_id[item_id] != after_by_id[item_id]
        )
        return {
            "added": [after_by_id[item_id] for item_id in added_ids],
            "changed": [
                {
                    "id": item_id,
                    "before": before_by_id[item_id],
                    "after": after_by_id[item_id],
                }
                for item_id in changed_ids
            ],
            "removed": [before_by_id[item_id] for item_id in removed_ids],
        }

    @classmethod
    def _state_diff(cls, before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        before_raw = {
            str(item.get("episode_id"))
            for item in before.get("raw_memory", [])
            if item.get("episode_id")
        }
        after_raw = {
            str(item.get("episode_id"))
            for item in after.get("raw_memory", [])
            if item.get("episode_id")
        }
        return {
            "before": cls._state_summary(before),
            "after": cls._state_summary(after),
            "raw_memory_added_episode_ids": sorted(after_raw - before_raw),
            "raw_memory_removed_episode_ids": sorted(before_raw - after_raw),
            "strategies": cls._entity_changes(
                before.get("abstract_memory", []),
                after.get("abstract_memory", []),
                "artifact_id",
            ),
            "capabilities": cls._entity_changes(
                before.get("capabilities", []),
                after.get("capabilities", []),
                "capability_id",
            ),
            "anchors": cls._entity_changes(
                before.get("anchors", []), after.get("anchors", []), "anchor_id"
            ),
        }

    def _write_episode_log(
        self,
        phase: int,
        mode: str,
        result: EpisodeResult,
        state_before: Dict[str, Any],
        usage_before: Dict[str, Any],
        usage_delta: Optional[Dict[str, Any]] = None,
    ) -> None:
        state_after = self.harness.state.to_dict()
        candidate_summary = None
        if result.candidate is not None:
            candidate_summary = {
                "candidate_id": result.candidate.candidate_id,
                "base_version": result.candidate.base_version,
                "proposed_version": result.candidate.proposed_state.version,
                "operations": result.candidate.operations,
                "rationale": result.candidate.rationale,
                "candidate_file": (
                    f"harness/candidates/{result.candidate.candidate_id}.json"
                ),
            }
        payload = {
            "phase": phase,
            "mode": mode,
            "harness_version": self.harness.state.version,
            "record": result.record.to_dict(),
            "candidate": candidate_summary,
            "evaluation": result.evaluation.to_dict() if result.evaluation else None,
            "state_change": self._state_diff(state_before, state_after),
            "llm_usage": usage_delta
            if usage_delta is not None
            else self._usage_delta(usage_before, self.llm.usage_snapshot()),
        }
        episode_path = self.artifacts.write_episode(payload)
        if result.candidate is not None:
            update_path = self.artifacts.write_update(
                {
                    "phase": phase,
                    "mode": mode,
                    "episode_id": result.record.episode_id,
                    "task_type": result.record.task_type,
                    "success": result.record.success,
                    "candidate": candidate_summary,
                    "evaluation": payload["evaluation"],
                    "applied_state_change": payload["state_change"],
                }
            )
        else:
            update_path = None
        self.artifacts.event(
            "episode_completed",
            "Episode completed and logged.",
            phase=phase,
            mode=mode,
            episode_id=result.record.episode_id,
            success=result.record.success,
            steps=len(result.record.steps),
            harness_version=self.harness.state.version,
            episode_file=str(episode_path.relative_to(self.output_dir)),
            update_file=(
                str(update_path.relative_to(self.output_dir)) if update_path else None
            ),
        )

    @staticmethod
    def _merge_usage(previous: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        merged = {
            "requests": int(previous.get("requests", 0)) + int(current.get("requests", 0)),
            "prompt_tokens": int(previous.get("prompt_tokens", 0))
            + int(current.get("prompt_tokens", 0)),
            "completion_tokens": int(previous.get("completion_tokens", 0))
            + int(current.get("completion_tokens", 0)),
            "by_component": {},
        }
        component_names = set(previous.get("by_component", {})) | set(
            current.get("by_component", {})
        )
        for component in sorted(component_names):
            old = previous.get("by_component", {}).get(component, {})
            new = current.get("by_component", {}).get(component, {})
            merged["by_component"][component] = {
                "requests": int(old.get("requests", 0)) + int(new.get("requests", 0)),
                "prompt_tokens": int(old.get("prompt_tokens", 0))
                + int(new.get("prompt_tokens", 0)),
                "completion_tokens": int(old.get("completion_tokens", 0))
                + int(new.get("completion_tokens", 0)),
            }
        return merged

    @staticmethod
    def _usage_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        delta = {
            "requests": int(after.get("requests", 0)) - int(before.get("requests", 0)),
            "prompt_tokens": int(after.get("prompt_tokens", 0))
            - int(before.get("prompt_tokens", 0)),
            "completion_tokens": int(after.get("completion_tokens", 0))
            - int(before.get("completion_tokens", 0)),
            "by_component": {},
        }
        component_names = set(before.get("by_component", {})) | set(
            after.get("by_component", {})
        )
        for component in sorted(component_names):
            old = before.get("by_component", {}).get(component, {})
            new = after.get("by_component", {}).get(component, {})
            component_delta = {
                "requests": int(new.get("requests", 0)) - int(old.get("requests", 0)),
                "prompt_tokens": int(new.get("prompt_tokens", 0))
                - int(old.get("prompt_tokens", 0)),
                "completion_tokens": int(new.get("completion_tokens", 0))
                - int(old.get("completion_tokens", 0)),
            }
            if any(component_delta.values()):
                delta["by_component"][component] = component_delta
        return delta

    def _add_parallel_usage(self, usage: Dict[str, Any]) -> None:
        self.parallel_llm_usage = self._merge_usage(self.parallel_llm_usage, usage)

    def _live_llm_usage(self) -> Dict[str, Any]:
        return self._merge_usage(self.llm.usage_snapshot(), self.parallel_llm_usage)

    def _total_llm_usage(self) -> Dict[str, Any]:
        return self._merge_usage(self.prior_llm_usage, self._live_llm_usage())

    def _checkpoint(
        self,
        cursor: Dict[str, Any],
        matrix: List[List[float]],
        phase_summaries: List[Dict[str, Any]],
        elapsed_seconds: float,
    ) -> None:
        self.artifacts.save_checkpoint(
            {
                # next_phase remains for compatibility with earlier phase checkpoints.
                "next_phase": int(cursor["phase_index"]),
                "cursor": cursor,
                "matrix": matrix,
                "phase_summaries": phase_summaries,
                "runtime_stats": self.runtime_stats,
                "llm_usage": self._total_llm_usage(),
                "elapsed_seconds": elapsed_seconds,
                "harness_state": self.harness.state.to_dict(),
                "sequence_manifest": str(
                    resolve_project_path(self.config, "sequence_manifest")
                ),
            }
        )

    @staticmethod
    def _initial_cursor(phase_index: int, task_count: int) -> Dict[str, Any]:
        return {
            "phase_index": phase_index,
            "stage": "train",
            "train_game_index": 0,
            "train_successes": 0,
            "eval_task_index": 0,
            "eval_game_index": 0,
            "eval_successes": [0 for _ in range(task_count)],
            "eval_success_rates": [0.0 for _ in range(task_count)],
        }

    def run(self) -> Dict[str, Any]:
        if self.config.get("run_mode") == "standard_evaluation":
            return self.run_standard_evaluation()
        manifest_path = resolve_project_path(self.config, "sequence_manifest")
        manifest = load_sequence(manifest_path)
        phases = manifest["phases"]
        matrix: List[List[float]] = []
        phase_summaries: List[Dict[str, Any]] = []
        cursor = self._initial_cursor(0, len(phases))
        started = time.time()
        if self.resume_requested:
            checkpoint = self.artifacts.load_checkpoint()
            if checkpoint.get("sequence_manifest") != str(manifest_path):
                raise ValueError("Resume checkpoint uses a different sequence manifest.")
            matrix = list(checkpoint.get("matrix", []))
            phase_summaries = list(checkpoint.get("phase_summaries", []))
            cursor = dict(
                checkpoint.get(
                    "cursor",
                    self._initial_cursor(
                        int(checkpoint.get("next_phase", len(matrix))), len(phases)
                    ),
                )
            )
            self.runtime_stats = dict(checkpoint.get("runtime_stats", self.runtime_stats))
            self.prior_llm_usage = dict(checkpoint.get("llm_usage", {}))
            self.elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))
            self.harness.state = HarnessState.from_dict(checkpoint["harness_state"])
            if self.raw_memory_retrieval_override is not None:
                self.harness.state.routing_policy["raw_memory_retrieval_enabled"] = bool(
                    self.raw_memory_retrieval_override
                )
            self.harness.store.save_stable(self.harness.state)
            self.artifacts.event(
                "checkpoint_restored",
                "Restored the last completed episode checkpoint.",
                cursor=cursor,
                harness_version=self.harness.state.version,
            )
        else:
            self._checkpoint(cursor, matrix, phase_summaries, 0.0)

        self.artifacts.set_status("running", cursor=cursor)
        # Ensure every run has a result snapshot even before its first phase completes.
        self._save_results(matrix, phase_summaries, started)
        try:
            while int(cursor["phase_index"]) < len(phases):
                phase_index = int(cursor["phase_index"])
                phase = phases[phase_index]
                evaluation_scope = str(
                    self.config.get("sequence", {}).get(
                        "evaluation_scope", "all_tasks"
                    )
                )
                if evaluation_scope == "learned_tasks":
                    evaluation_task_count = phase_index + 1
                elif evaluation_scope == "all_tasks":
                    evaluation_task_count = len(phases)
                else:
                    raise ValueError(
                        f"Unsupported sequence.evaluation_scope: {evaluation_scope}"
                    )
                self.artifacts.event(
                    "phase_started",
                    "Phase execution started or resumed.",
                    phase=phase_index,
                    task_type=phase["task_type"],
                    cursor=cursor,
                )
                if cursor["stage"] == "train":
                    train_games = phase["train_games"]
                    for train_index in range(
                        int(cursor["train_game_index"]), len(train_games)
                    ):
                        game_file = train_games[train_index]
                        state_before = self.harness.state.to_dict()
                        usage_before = self.llm.usage_snapshot()
                        result = self.run_episode(
                            game_file,
                            manifest["train_split"],
                            self.learning_policy,
                            learn=bool(
                                (
                                    self.config[self.continual_method]
                                    if self.continual_method in {"rag"}
                                    else self.config["harness"]
                                ).get("updates_enabled", True)
                            ),
                            training_episode=True,
                        )
                        self._write_episode_log(
                            phase_index,
                            "learn",
                            result,
                            state_before,
                            usage_before,
                        )
                        cursor["train_successes"] = int(cursor["train_successes"]) + int(
                            result.record.success
                        )
                        cursor["train_game_index"] = train_index + 1
                        elapsed = self.elapsed_before + time.time() - started
                        self._checkpoint(cursor, matrix, phase_summaries, elapsed)
                    cursor["stage"] = "eval"
                    cursor["eval_task_index"] = 0
                    cursor["eval_game_index"] = 0
                    elapsed = self.elapsed_before + time.time() - started
                    self._checkpoint(cursor, matrix, phase_summaries, elapsed)

                for task_index in range(
                    int(cursor["eval_task_index"]), evaluation_task_count
                ):
                    eval_phase = phases[task_index]
                    eval_games = eval_phase["eval_games"]
                    start_game = (
                        int(cursor["eval_game_index"])
                        if task_index == int(cursor["eval_task_index"])
                        else 0
                    )
                    remaining_games = list(eval_games[start_game:])
                    outcomes = self._ordered_read_only_outcomes(
                        remaining_games, manifest["eval_split"]
                    )
                    for offset, outcome in enumerate(outcomes):
                        eval_game_index = start_game + offset
                        result = outcome.result
                        if outcome.requires_runtime_record:
                            self._record_episode_runtime(result)
                        if outcome.requires_usage_merge:
                            self._add_parallel_usage(outcome.llm_usage)
                        self._write_episode_log(
                            phase_index,
                            f"eval_task_{task_index}",
                            result,
                            outcome.state_before,
                            {},
                            usage_delta=outcome.llm_usage,
                        )
                        cursor["eval_successes"][task_index] = int(
                            cursor["eval_successes"][task_index]
                        ) + int(result.record.success)
                        cursor["eval_task_index"] = task_index
                        cursor["eval_game_index"] = eval_game_index + 1
                        elapsed = self.elapsed_before + time.time() - started
                        self._checkpoint(cursor, matrix, phase_summaries, elapsed)
                    cursor["eval_success_rates"][task_index] = int(
                        cursor["eval_successes"][task_index]
                    ) / max(1, len(eval_games))
                    cursor["eval_task_index"] = task_index + 1
                    cursor["eval_game_index"] = 0
                    elapsed = self.elapsed_before + time.time() - started
                    self._checkpoint(cursor, matrix, phase_summaries, elapsed)

                row = list(cursor["eval_success_rates"][:evaluation_task_count])
                matrix.append(row)
                phase_summaries.append(
                    {
                        "phase": phase_index,
                        "task_type": phase["task_type"],
                        "train_success_rate": int(cursor["train_successes"])
                        / max(1, len(phase["train_games"])),
                        "eval_success_rates": row,
                        "eval_success_counts": list(
                            cursor["eval_successes"][:evaluation_task_count]
                        ),
                        "eval_game_counts": [
                            len(phases[index]["eval_games"])
                            for index in range(evaluation_task_count)
                        ],
                        "harness_version": self.harness.state.version,
                    }
                )
                cursor = self._initial_cursor(phase_index + 1, len(phases))
                elapsed = self.elapsed_before + time.time() - started
                self._checkpoint(cursor, matrix, phase_summaries, elapsed)
                self._save_results(matrix, phase_summaries, started)
                self.artifacts.event(
                    "phase_completed",
                    "Phase checkpoint and results saved.",
                    phase=phase_index,
                    task_type=phase["task_type"],
                    train_success_rate=phase_summaries[-1]["train_success_rate"],
                    eval_success_rates=row,
                    harness_version=self.harness.state.version,
                )
            results = self._save_results(matrix, phase_summaries, started)
            final_elapsed = float(results["elapsed_seconds"])
            self._checkpoint(cursor, matrix, phase_summaries, final_elapsed)
            self.artifacts.set_status(
                "completed", next_phase=len(phases), harness_version=self.harness.state.version
            )
            self.artifacts.event(
                "run_completed",
                "All phases completed.",
                metrics=results["metrics"],
                elapsed_seconds=final_elapsed,
            )
            return results
        except BaseException as exc:
            self.artifacts.set_status(
                "interrupted",
                cursor=cursor,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            self.artifacts.event(
                "run_interrupted",
                "Run stopped; resume will restart only the incomplete episode.",
                cursor=cursor,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise

    def _standard_checkpoint(
        self,
        games: List[Dict[str, str]],
        split: str,
        next_game_index: int,
        successes: int,
        per_task_type: Dict[str, Dict[str, int]],
        elapsed_seconds: float,
    ) -> None:
        self.artifacts.save_checkpoint(
            {
                "run_mode": "standard_evaluation",
                "split": split,
                "games": games,
                "next_game_index": next_game_index,
                "successes": successes,
                "per_task_type": per_task_type,
                "runtime_stats": self.runtime_stats,
                "llm_usage": self._total_llm_usage(),
                "elapsed_seconds": elapsed_seconds,
                "harness_state": self.harness.state.to_dict(),
            }
        )

    def run_standard_evaluation(self) -> Dict[str, Any]:
        evaluation = self.config.get("standard_evaluation", {})
        split = str(evaluation.get("split", "valid_unseen"))
        game_source = str(evaluation.get("game_source", "full_split"))
        if game_source == "full_split":
            games = list_split_games(self.data_dir, split)
        elif game_source == "sequence_manifest":
            manifest = load_sequence(
                resolve_project_path(self.config, "sequence_manifest")
            )
            if manifest.get("eval_split") != split:
                raise ValueError(
                    "Sequence manifest eval_split does not match standard_evaluation.split."
                )
            games = [
                {"game_file": game_file, "task_type": str(phase["task_type"])}
                for phase in manifest["phases"]
                for game_file in phase["eval_games"]
            ]
            game_paths = [item["game_file"] for item in games]
            if len(game_paths) != len(set(game_paths)):
                raise ValueError("Sequence manifest contains duplicate evaluation games.")
        else:
            raise ValueError(f"Unsupported standard_evaluation.game_source: {game_source}")
        expected_count = int(evaluation.get("expected_game_count", 134))
        if len(games) != expected_count:
            raise ValueError(
                f"Expected {expected_count} games in {split}, found {len(games)}."
            )
        if bool(self.config["harness"].get("updates_enabled", True)):
            raise ValueError("standard_evaluation requires harness.updates_enabled: false")

        started = time.time()
        next_game_index = 0
        successes = 0
        per_task_type = {
            task_type: {"successes": 0, "total": 0}
            for task_type in sorted({item["task_type"] for item in games})
        }
        if self.resume_requested:
            checkpoint = self.artifacts.load_checkpoint()
            if checkpoint.get("run_mode") != "standard_evaluation":
                raise ValueError("Resume checkpoint is not a standard evaluation run.")
            if checkpoint.get("split") != split or checkpoint.get("games") != games:
                raise ValueError("The standard evaluation game list changed since checkpoint.")
            next_game_index = int(checkpoint.get("next_game_index", 0))
            successes = int(checkpoint.get("successes", 0))
            per_task_type = {
                key: {"successes": int(value["successes"]), "total": int(value["total"])}
                for key, value in checkpoint.get("per_task_type", {}).items()
            }
            self.runtime_stats = dict(checkpoint.get("runtime_stats", self.runtime_stats))
            self.prior_llm_usage = dict(checkpoint.get("llm_usage", {}))
            self.elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))
            self.harness.state = HarnessState.from_dict(checkpoint["harness_state"])
            self.harness.store.save_stable(self.harness.state)
            self.artifacts.event(
                "checkpoint_restored",
                "Restored the last completed standard-evaluation episode.",
                next_game_index=next_game_index,
                completed_games=next_game_index,
                total_games=len(games),
            )
        else:
            self._standard_checkpoint(
                games, split, next_game_index, successes, per_task_type, 0.0
            )

        self.artifacts.set_status(
            "running",
            run_mode="standard_evaluation",
            next_game_index=next_game_index,
            total_games=len(games),
        )
        results = self._save_standard_results(
            games, split, next_game_index, successes, per_task_type, started
        )
        try:
            batch_start_index = next_game_index
            remaining_games = games[batch_start_index:]
            outcomes = self._ordered_read_only_outcomes(
                [game["game_file"] for game in remaining_games], split
            )
            for offset, outcome in enumerate(outcomes):
                game_index = batch_start_index + offset
                game = games[game_index]
                result = outcome.result
                if outcome.requires_runtime_record:
                    self._record_episode_runtime(result)
                if outcome.requires_usage_merge:
                    self._add_parallel_usage(outcome.llm_usage)
                self._write_episode_log(
                    -1,
                    "standard_eval",
                    result,
                    outcome.state_before,
                    {},
                    usage_delta=outcome.llm_usage,
                )
                succeeded = int(result.record.success)
                successes += succeeded
                task_counts = per_task_type[game["task_type"]]
                task_counts["successes"] += succeeded
                task_counts["total"] += 1
                completed_index = game_index + 1
                elapsed = self.elapsed_before + time.time() - started
                self._standard_checkpoint(
                    games, split, completed_index, successes, per_task_type, elapsed
                )
                results = self._save_standard_results(
                    games, split, completed_index, successes, per_task_type, started
                )
                print(
                    f"[standard_eval {completed_index}/{len(games)}] "
                    f"success={bool(succeeded)} cumulative_sr="
                    f"{results['metrics']['success_rate']:.4f}",
                    flush=True,
                )
                next_game_index = completed_index

            self.artifacts.set_status(
                "completed",
                run_mode="standard_evaluation",
                completed_games=next_game_index,
                total_games=len(games),
            )
            self.artifacts.event(
                "run_completed",
                "Standard ALFWorld evaluation completed.",
                success_rate=results["metrics"]["success_rate"],
                successes=successes,
                total_games=len(games),
            )
            return results
        except BaseException as exc:
            self.artifacts.set_status(
                "interrupted",
                run_mode="standard_evaluation",
                next_game_index=next_game_index,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            self.artifacts.event(
                "run_interrupted",
                "Standard evaluation stopped; resume will retry the incomplete episode.",
                next_game_index=next_game_index,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise

    def _save_standard_results(
        self,
        games: List[Dict[str, str]],
        split: str,
        completed_games: int,
        successes: int,
        per_task_type: Dict[str, Dict[str, int]],
        started: float,
    ) -> Dict[str, Any]:
        elapsed_seconds = self.elapsed_before + time.time() - started
        game_source = str(
            self.config.get("standard_evaluation", {}).get(
                "game_source", "full_split"
            )
        )
        per_task_results = {
            task_type: {
                **counts,
                "success_rate": counts["successes"] / max(1, counts["total"]),
            }
            for task_type, counts in sorted(per_task_type.items())
        }
        results = {
            "config": self.config,
            "run_dir": str(self.output_dir),
            "run_mode": "standard_evaluation",
            "game_source": game_source,
            "split": split,
            "expected_games": len(games),
            "completed_games": completed_games,
            "successes": successes,
            "metrics": {
                "success_rate": successes / max(1, completed_games),
            },
            "per_task_type": per_task_results,
            "harness_version": self.harness.state.version,
            "elapsed_seconds": elapsed_seconds,
            "runtime": {
                **self.runtime_stats,
                "average_steps": self.runtime_stats["steps"]
                / max(1, self.runtime_stats["episodes"]),
                "llm_usage": self._total_llm_usage(),
            },
        }
        RunArtifacts._write_json_atomic(self.output_dir / "results.json", results)
        csv_path = self.output_dir / "task_type_results.csv"
        temporary_csv = csv_path.with_suffix(".csv.tmp")
        with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["task_type", "successes", "total", "success_rate"])
            for task_type, counts in per_task_results.items():
                writer.writerow(
                    [
                        task_type,
                        counts["successes"],
                        counts["total"],
                        counts["success_rate"],
                    ]
                )
        temporary_csv.replace(csv_path)
        status = "complete" if completed_games == len(games) else "in progress"
        lines = [
            "# Standard ALFWorld evaluation",
            "",
            f"- Status: {status}",
            f"- Split: {split}",
            f"- Completed games: {completed_games}/{len(games)}",
            f"- Successes: {successes}",
            f"- Success rate: {results['metrics']['success_rate']:.6f}",
            f"- Harness version: {self.harness.state.version}",
            f"- LLM requests: {results['runtime']['llm_usage']['requests']}",
            "",
            "## Per task type",
            "",
        ]
        for task_type, counts in per_task_results.items():
            lines.append(
                f"- {task_type}: {counts['successes']}/{counts['total']} "
                f"({counts['success_rate']:.6f})"
            )
        summary_path = self.output_dir / "results_summary.md"
        temporary_summary = summary_path.with_suffix(".md.tmp")
        with temporary_summary.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        temporary_summary.replace(summary_path)
        return results

    def _save_results(
        self,
        matrix: List[List[float]],
        phase_summaries: List[Dict[str, Any]],
        started: float,
    ) -> Dict[str, Any]:
        elapsed_seconds = self.elapsed_before + time.time() - started
        llm_usage = self._total_llm_usage()
        metrics = continual_metrics(matrix)
        if phase_summaries:
            final_summary = phase_summaries[-1]
            final_successes = sum(final_summary.get("eval_success_counts", []))
            final_games = sum(final_summary.get("eval_game_counts", []))
            metrics["final_micro_success_rate"] = final_successes / max(1, final_games)
        else:
            metrics["final_micro_success_rate"] = 0.0
        results = {
            "config": self.config,
            "run_dir": str(self.output_dir),
            "matrix": matrix,
            "metrics": metrics,
            "phases": phase_summaries,
            "harness_version": self.harness.state.version,
            "elapsed_seconds": elapsed_seconds,
            "runtime": {
                **self.runtime_stats,
                "average_steps": self.runtime_stats["steps"]
                / max(1, self.runtime_stats["episodes"]),
                "average_anchor_retention": self.runtime_stats["anchor_retention_sum"]
                / max(1, self.runtime_stats["anchor_evaluations"]),
                "llm_usage": llm_usage,
            },
        }
        RunArtifacts._write_json_atomic(self.output_dir / "results.json", results)
        matrix_path = self.output_dir / "success_matrix.csv"
        temporary_matrix = matrix_path.with_suffix(".csv.tmp")
        with temporary_matrix.open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            width = max((len(row) for row in matrix), default=0)
            writer.writerow(["after_phase"] + [f"task_{index}" for index in range(width)])
            for index, row in enumerate(matrix):
                writer.writerow([index] + row)
        temporary_matrix.replace(matrix_path)
        self._write_result_summary(results)
        return results

    def _write_result_summary(self, results: Dict[str, Any]) -> None:
        metrics = results["metrics"]
        forward_transfer = metrics["forward_transfer"]
        forward_transfer_text = (
            f"{forward_transfer:.6f}" if forward_transfer is not None else "N/A"
        )
        expected_phases = len(self.config.get("sequence", {}).get("task_order", []))
        is_complete = expected_phases > 0 and len(results["matrix"]) == expected_phases
        lines = [
            "# Run summary",
            "",
            f"- Status: {'complete' if is_complete else 'in progress'}",
            f"- Completed phases: {len(results['matrix'])}",
            f"- Harness version: {results['harness_version']}",
            f"- Episodes: {results['runtime']['episodes']}",
            f"- LLM requests: {results['runtime']['llm_usage']['requests']}",
            f"- Elapsed seconds: {results['elapsed_seconds']:.3f}",
            "",
            "## Continual-learning metrics",
            "",
            f"- Average accuracy: {metrics['average_accuracy']:.6f}",
            f"- Backward transfer: {metrics['backward_transfer']:.6f}",
            f"- Forgetting: {metrics['forgetting']:.6f}",
            f"- Forward transfer: {forward_transfer_text}",
            f"- Final micro success rate: {metrics['final_micro_success_rate']:.6f}",
            "",
        ]
        path = self.output_dir / "results_summary.md"
        temporary = path.with_suffix(".md.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        temporary.replace(path)
