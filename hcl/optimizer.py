from __future__ import annotations

import json
import math
import random
import re
from hashlib import sha256
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .json_utils import atomic_write_json
from .prompt_templates import PromptTemplate


DEFAULT_OPTIMIZER_DIR = Path(__file__).resolve().parent / "optimizer"
DEFAULT_CANDIDATE_GENERATION_TEMPLATE = DEFAULT_OPTIMIZER_DIR / "candidate_generation_prompt_default.json"


@dataclass
class Optimizer:
    record_dir: str | Path = DEFAULT_OPTIMIZER_DIR
    write_pretty_history: bool = True
    write_candidate_files: bool = True
    candidate_generation_template_path: str | Path = DEFAULT_CANDIDATE_GENERATION_TEMPLATE
    candidate_count: int = 3
    selection_objective: str = "stability"
    max_forget_count: int = 0
    # Paper-facing historical-loss budget B_n. None represents infinity. This
    # is intentionally separate from candidate ranking so Stability-HCL and
    # Plasticity-HCL can differ only in the retention constraint.
    historical_loss_budget: int | None = None
    # Deprecated configuration alias retained so existing experiment JSON can
    # migrate without changing behavior.
    min_accuracy_delta: float = 0.10
    min_correct_gain: int = 2
    min_format_compliance_rate: float = 0.90
    commit_policy: str = "strict_no_forgetting"
    recover_reward: float = 1.0
    forget_penalty: float = 2.0
    min_anchor_score: float = -1.0
    min_anchor_net_score: float | None = None
    max_anchor_forget_rate: float | None = None
    max_anchor_forget_count: int | None = None
    plasticity_weight: float = 0.5
    plasticity_scale: float = 0.05
    stability_scale: float = 0.05
    min_joint_score: float = 0.0
    selection_mode: str = "gibbs"
    selection_temperature: float = 0.10
    selection_seed: int = 20260723
    min_historical_anchor_count: int = 20
    min_historical_anchors_per_task: int = 15
    require_all_historical_tasks: bool = True
    min_anchor_evaluation_coverage: float = 0.90
    screening_ratio: float = 0.25
    screening_min_current_count: int = 12
    screening_min_anchors_per_task: int = 5
    screening_keep_top_k: int = 1
    screening_close_score_margin: float = 0.10
    screening_close_keep_top_k: int = 2
    components: list[str] = field(
        default_factory=lambda: ["task_interface_structuring", "router_workflow", "router_context"]
    )
    min_primary_score_delta: float | None = None
    history: list[dict[str, object]] = field(default_factory=list)
    last_generation_diagnostics: dict[str, object] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.record_dir = Path(self.record_dir)
        self.components = [str(component) for component in self.components]
        self.selection_objective = str(self.selection_objective or "stability").strip().lower()
        if self.selection_objective not in {"stability", "plasticity", "balanced"}:
            raise ValueError("optimizer.selection_objective must be one of: stability, plasticity, balanced")
        if self.historical_loss_budget is not None:
            self.historical_loss_budget = int(self.historical_loss_budget)
            if self.historical_loss_budget < 0:
                raise ValueError("optimizer.historical_loss_budget must be non-negative or null")
        self.plasticity_weight = min(max(float(self.plasticity_weight), 0.0), 1.0)
        self.min_primary_score_delta = float(
            self.min_accuracy_delta
            if self.min_primary_score_delta is None
            else self.min_primary_score_delta
        )
        self.plasticity_scale = max(float(self.plasticity_scale), 1e-12)
        self.stability_scale = max(float(self.stability_scale), 1e-12)
        self.selection_mode = str(self.selection_mode or "gibbs").strip().lower()
        if self.selection_mode not in {"gibbs", "argmax"}:
            raise ValueError("optimizer.selection_mode must be one of: gibbs, argmax")
        self.selection_temperature = max(float(self.selection_temperature), 1e-12)
        self.screening_ratio = min(max(float(self.screening_ratio), 0.0), 1.0)
        self.min_anchor_evaluation_coverage = min(max(float(self.min_anchor_evaluation_coverage), 0.0), 1.0)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.candidate_history_path = self.record_dir / "candidate_history.jsonl"
        self.generation_history_path = self.record_dir / "candidate_generation_history.jsonl"
        self.generation_failure_path = self.record_dir / "candidate_generation_failure_history.jsonl"
        self.candidate_dir = self.record_dir / "candidates"
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        self.candidate_generation_template = PromptTemplate.from_path(self.candidate_generation_template_path)

    def generate_artifact_candidates(
        self,
        component: str,
        train_result: dict[str, object],
        *,
        model: Any,
        current_artifact: dict[str, object] | None = None,
        harness_version: int = 0,
        generation_context: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        count = max(int(self.candidate_count), 0)
        if count == 0:
            self.last_generation_diagnostics = {
                "parse_status": "skipped",
                "parse_error": None,
                "raw_output_chars": 0,
                "raw_output_preview": "",
                "raw_candidate_count": 0,
                "normalized_candidate_count": 0,
                "normalization_rejections": [],
            }
            return []
        metrics = dict(train_result.get("metrics", {}))
        candidate_version = self._next_candidate_version()
        current_artifact = current_artifact or {"template_id": f"{component}.current", "content": {}}
        generation_context = generation_context or {}
        prompt = self.candidate_generation_template.render(
            {
                "component": component,
                "candidate_count": count,
                "harness_version": harness_version,
                "train_metrics": json.dumps(metrics, ensure_ascii=False, indent=2),
                "current_artifact": json.dumps(current_artifact, ensure_ascii=False, indent=2),
                "generation_context": json.dumps(generation_context, ensure_ascii=False, indent=2),
            }
        )
        raw_output = model.generate(
            prompt,
            state={
                "optimizer_phase": "generate_artifact_candidates",
                "optimizer_component": component,
                "optimizer_candidate_count": count,
            },
        )
        decoded, parse_status, parse_error = _extract_json_with_diagnostics(raw_output)
        candidates, normalization_rejections = self._normalize_candidates(
            decoded.get("candidates"),
            component=component,
            source_version=harness_version,
            candidate_version=candidate_version,
            count=count,
            current_artifact=current_artifact,
        )
        raw_candidate_count = len(decoded.get("candidates", [])) if isinstance(decoded.get("candidates"), list) else 0
        self.last_generation_diagnostics = {
            "parse_status": parse_status,
            "parse_error": parse_error,
            "raw_output_chars": len(raw_output),
            "raw_output_preview": raw_output[:2000],
            "raw_candidate_count": raw_candidate_count,
            "normalized_candidate_count": len(candidates),
            "normalization_rejections": normalization_rejections,
        }
        self.record_candidate_generation(
            candidate_version=candidate_version,
            component=component,
            source_version=harness_version,
            source_metrics={"train": metrics},
            current_artifact=current_artifact,
            generation_context=generation_context,
            candidates=candidates,
            diagnostics=self.last_generation_diagnostics,
        )
        return candidates

    def _normalize_candidates(
        self,
        raw_candidates: Any,
        *,
        component: str,
        source_version: int,
        candidate_version: int,
        count: int,
        current_artifact: dict[str, object],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        if not isinstance(raw_candidates, list):
            return [], [{"index": None, "reason": "top_level_candidates_not_list"}]
        candidates: list[dict[str, object]] = []
        rejections: list[dict[str, object]] = []
        for index, raw in enumerate(raw_candidates[:count]):
            if not isinstance(raw, dict):
                rejections.append({"index": index, "reason": "candidate_not_object"})
                continue
            candidate_name = str(raw.get("candidate_name") or f"candidate_{index + 1}")
            candidate_id = f"{component}:h{source_version}:c{candidate_version}:{candidate_name}"
            artifact_value = raw.get("artifact_content")
            if not isinstance(artifact_value, dict) and isinstance(raw.get("sections"), list):
                current_content = current_artifact.get("content")
                current_content = current_content if isinstance(current_content, dict) else {}
                artifact_value = {
                    "template_id": str(
                        raw.get("template_id")
                        or current_content.get("template_id")
                        or f"{component}.candidate"
                    ),
                    "description": str(
                        raw.get("description")
                        or current_content.get("description")
                        or f"Candidate prompt for {component}."
                    ),
                    "sections": raw.get("sections"),
                }
            artifact_content = _normalize_artifact_content(artifact_value)
            if not artifact_content:
                rejections.append({"index": index, "candidate_name": candidate_name, "reason": "missing_prompt_sections"})
                continue
            artifact_diff = raw.get("artifact_diff") if isinstance(raw.get("artifact_diff"), dict) else {
                "summary": str(raw.get("artifact_diff_summary") or "")
            }
            validation_errors = _artifact_validation_errors(component, artifact_content)
            candidates.append(
                {
                    "candidate_version": candidate_version,
                    "candidate_id": candidate_id,
                    "component": component,
                    "source_version": source_version,
                    "candidate_name": candidate_name,
                    "rationale": str(raw.get("rationale") or ""),
                    "artifact_content": artifact_content,
                    "artifact_diff": artifact_diff,
                    "validation_errors": validation_errors,
                }
            )
        return candidates, rejections

    def record_candidate_generation(
        self,
        *,
        candidate_version: int,
        component: str,
        source_version: int,
        source_metrics: dict[str, object],
        current_artifact: dict[str, object],
        generation_context: dict[str, object],
        candidates: list[dict[str, object]],
        diagnostics: dict[str, object],
    ) -> None:
        generation_record = {
            "candidate_version": candidate_version,
            "component": component,
            "source_version": source_version,
            "source_metrics": source_metrics,
            "current_artifact": current_artifact,
            "generation_context": generation_context,
            "candidate_ids": [candidate.get("candidate_id") for candidate in candidates],
            "generation_diagnostics": diagnostics,
        }
        if candidates:
            # A generation record must reference artifacts that already exist.
            self.record_candidates(candidates)
        else:
            _append_jsonl_and_pretty(
                self.generation_failure_path,
                {
                    "candidate_version": candidate_version,
                    "component": component,
                    "source_version": source_version,
                    **diagnostics,
                },
                write_pretty=self.write_pretty_history,
            )
        _append_jsonl_and_pretty(
            self.generation_history_path,
            generation_record,
            write_pretty=self.write_pretty_history,
        )

    def record_candidates(self, candidates: list[dict[str, object]]) -> None:
        if not candidates:
            return
        if self.write_candidate_files:
            for candidate in candidates:
                self._write_candidate_file(candidate)
        for candidate in candidates:
            _append_jsonl_and_pretty(
                self.candidate_history_path,
                candidate,
                write_pretty=self.write_pretty_history,
            )

    def _next_candidate_version(self) -> int:
        if not self.generation_history_path.exists():
            return 1
        latest = 0
        with self.generation_history_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    latest = max(latest, int(row.get("candidate_version", 0)))
                except (TypeError, ValueError):
                    continue
        return latest + 1

    def _write_candidate_file(self, candidate: dict[str, object]) -> None:
        raw_candidate_id = str(candidate.get("candidate_id", "candidate"))
        candidate_id = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_candidate_id).strip("._") or "candidate"
        candidate_id = re.sub(r"\.{2,}", "_", candidate_id)
        if len(candidate_id) > 180:
            digest = sha256(raw_candidate_id.encode("utf-8")).hexdigest()[:12]
            candidate_id = f"{candidate_id[:167]}_{digest}"
        path = self.candidate_dir / f"{candidate_id}.json"
        # Recreate a directory removed after construction and never expose a
        # partially written candidate JSON file.
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, candidate)

    def select_best_candidate(
        self,
        candidates: list[dict[str, object]],
        candidate_results: dict[str, dict[str, object]],
        current_val_result: dict[str, object],
        *,
        anchor_context: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        if not candidates:
            parse_status = str(self.last_generation_diagnostics.get("parse_status", "unknown"))
            reason = "candidate_generation_parse_failed" if parse_status == "failed" else "no_valid_candidate_generated"
            self.history.append(
                {
                    "accepted": False,
                    "reason": reason,
                    "generation_diagnostics": self.last_generation_diagnostics,
                    "candidate_rejections": [],
                }
            )
            return None
        if self.selection_objective == "balanced":
            return self._select_balanced_candidate(
                candidates,
                candidate_results,
                current_val_result,
                anchor_context=anchor_context or {},
            )
        current_task_result = _scoped_result(current_val_result, "current_task_metrics")
        current_primary_score = _primary_score(current_task_result)
        current_correct = _metric_int(current_task_result, "correct")
        best: dict[str, object] | None = None
        best_rank_key: tuple[float, ...] | None = None
        rejection_details: list[dict[str, object]] = []
        commit_policy = str(self.commit_policy or "strict_no_forgetting")
        for candidate in candidates:
            candidate_id = str(candidate.get("candidate_id", ""))
            candidate_result = candidate_results.get(candidate_id, {})
            candidate_task_result = _scoped_result(candidate_result, "current_task_metrics")
            candidate_anchor_result = _scoped_result(candidate_result, "historical_anchor_metrics")
            candidate_primary_score = _primary_score(candidate_task_result)
            forget_count = _metric_int(candidate_result, "forget_count")
            anchor_forget_count = _metric_int(candidate_anchor_result, "forget_count")
            anchor_recovered_count = _metric_int(candidate_anchor_result, "recovered_count")
            anchor_total = _metric_int(candidate_anchor_result, "total")
            anchor_forget_rate = _metric_float(candidate_anchor_result, "forget_rate")
            anchor_score = (
                float(self.recover_reward) * anchor_recovered_count
                - float(self.forget_penalty) * anchor_forget_count
            )
            anchor_net_score = anchor_score / max(anchor_total, 1)
            correct_gain = _metric_int(candidate_task_result, "correct") - current_correct
            primary_score_delta = candidate_primary_score - current_primary_score
            format_compliance_rate = _metric_float(candidate_task_result, "format_compliance_rate")
            validation_errors = candidate.get("validation_errors")
            reasons: list[str] = []
            if isinstance(validation_errors, list) and validation_errors:
                reasons.append("invalid_artifact")
            candidate_metrics = candidate_result.get("metrics", {})
            if isinstance(candidate_metrics, dict) and candidate_metrics.get("invalid_candidate"):
                reasons.append("invalid_candidate_evaluation")
            if (
                isinstance(candidate_metrics, dict)
                and int(candidate_metrics.get("behavior_comparable_count", 0) or 0) > 0
                and int(candidate_metrics.get("behavior_change_count", 0) or 0) == 0
            ):
                reasons.append("no_behavior_change")
            if primary_score_delta + 1e-12 < self.min_primary_score_delta:
                reasons.append("primary_score_delta_below_minimum")
            if correct_gain < self.min_correct_gain:
                reasons.append("correct_gain_below_minimum")
            if self.historical_loss_budget is not None:
                if anchor_forget_count > self.historical_loss_budget:
                    reasons.append("historical_loss_budget_exceeded")
            elif self.selection_objective == "stability":
                if forget_count > self.max_forget_count:
                    reasons.append("forget_count_exceeded")
                if anchor_forget_count > self.max_forget_count:
                    reasons.append("historical_anchor_forget_count_exceeded")
            # A null historical_loss_budget is B_n = infinity. Legacy configs
            # without this field retain the original objective-specific gate.
            if format_compliance_rate + 1e-12 < self.min_format_compliance_rate:
                reasons.append("format_compliance_below_minimum")
            rejection_details.append(
                {
                    "candidate_id": candidate_id,
                    "reasons": reasons,
                    "primary_score_delta": primary_score_delta,
                    "correct_gain": correct_gain,
                    "forget_count": forget_count,
                    "anchor_forget_count": anchor_forget_count,
                    "anchor_recovered_count": anchor_recovered_count,
                    "anchor_score": anchor_score,
                    "anchor_net_score": anchor_net_score,
                    "anchor_forget_rate": anchor_forget_rate,
                    "anchor_task_primary_scores": _metric_value(
                        candidate_anchor_result, "task_primary_scores", {}
                    ),
                    "format_compliance_rate": format_compliance_rate,
                    "behavior_comparable_count": _metric_int(candidate_result, "behavior_comparable_count"),
                    "behavior_change_count": _metric_int(candidate_result, "behavior_change_count"),
                }
            )
            if self.selection_objective == "stability":
                rank_key = (
                    -float(forget_count),
                    -float(anchor_forget_count),
                    -anchor_forget_rate,
                    float(anchor_recovered_count),
                    candidate_primary_score,
                    format_compliance_rate,
                )
            else:
                rank_key = (
                    candidate_primary_score,
                    -float(anchor_forget_count),
                    float(correct_gain),
                    primary_score_delta,
                    format_compliance_rate,
                )
            if not reasons and (best_rank_key is None or rank_key > best_rank_key):
                best = candidate
                best_rank_key = rank_key
        if best is None:
            self.history.append(
                {
                    "accepted": False,
                    "reason": "no_candidate_passed_commit_gate",
                    "current_validation_primary_score": current_primary_score,
                    "commit_gate": {
                        "selection_objective": self.selection_objective,
                        "commit_policy": commit_policy,
                        "historical_loss_budget": self.historical_loss_budget,
                        "min_primary_score_delta": self.min_primary_score_delta,
                        "min_correct_gain": self.min_correct_gain,
                        "max_forget_count": self.max_forget_count,
                        "min_format_compliance_rate": self.min_format_compliance_rate,
                        "recover_reward": self.recover_reward,
                        "forget_penalty": self.forget_penalty,
                        "min_anchor_score": self.min_anchor_score,
                        "min_anchor_net_score": self.min_anchor_net_score,
                        "max_anchor_forget_rate": self.max_anchor_forget_rate,
                    },
                    "candidate_rejections": rejection_details,
                }
            )
            return None
        best_result = candidate_results.get(str(best.get("candidate_id")), {})
        best_task_result = _scoped_result(best_result, "current_task_metrics")
        best_anchor_result = _scoped_result(best_result, "historical_anchor_metrics")
        best_primary_score = _primary_score(best_task_result)
        best_anchor_forget_count = _metric_int(best_anchor_result, "forget_count")
        best_anchor_recovered_count = _metric_int(best_anchor_result, "recovered_count")
        best_anchor_total = _metric_int(best_anchor_result, "total")
        best_anchor_score = (
            float(self.recover_reward) * best_anchor_recovered_count
            - float(self.forget_penalty) * best_anchor_forget_count
        )
        self.history.append(
            {
                "accepted": True,
                "reason": "candidate_passed_commit_gate",
                "candidate_id": best.get("candidate_id"),
                "component": best.get("component"),
                "source_version": best.get("source_version"),
                "current_validation_primary_score": current_primary_score,
                "candidate_validation_primary_score": best_primary_score,
                "primary_score_delta": best_primary_score - current_primary_score,
                "commit_gate": {
                    "selection_objective": self.selection_objective,
                    "commit_policy": commit_policy,
                    "min_primary_score_delta": self.min_primary_score_delta,
                    "min_correct_gain": self.min_correct_gain,
                    "max_forget_count": self.max_forget_count,
                    "min_format_compliance_rate": self.min_format_compliance_rate,
                    "recover_reward": self.recover_reward,
                    "forget_penalty": self.forget_penalty,
                    "min_anchor_score": self.min_anchor_score,
                    "min_anchor_net_score": self.min_anchor_net_score,
                    "max_anchor_forget_rate": self.max_anchor_forget_rate,
                },
                "anchor_forget_count": best_anchor_forget_count,
                "anchor_recovered_count": best_anchor_recovered_count,
                "anchor_score": best_anchor_score,
                "anchor_net_score": best_anchor_score / max(best_anchor_total, 1),
                "candidate_rejections": rejection_details,
                "correct_gain": _metric_int(
                    _scoped_result(
                        candidate_results.get(str(best.get("candidate_id")), {}),
                        "current_task_metrics",
                    ),
                    "correct",
                ) - current_correct,
            }
        )
        return best

    def shortlist_candidates(
        self,
        candidates: list[dict[str, object]],
        screening_results: dict[str, dict[str, object]],
        current_screening_result: dict[str, object],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Rank cheap screening runs using only plasticity and stability."""
        if len(candidates) <= 1 or self.screening_ratio <= 0.0 or self.screening_ratio >= 1.0:
            return list(candidates), []
        scored: list[tuple[float, str, dict[str, object], dict[str, object]]] = []
        diagnostics: list[dict[str, object]] = []
        for candidate in candidates:
            candidate_id = str(candidate.get("candidate_id", ""))
            result = screening_results.get(candidate_id, {})
            stats = self._balanced_stats(result, current_screening_result)
            metrics = result.get("metrics", {})
            invalid = bool(isinstance(metrics, dict) and metrics.get("invalid_candidate"))
            validation_errors = candidate.get("validation_errors")
            reasons: list[str] = []
            if isinstance(validation_errors, list) and validation_errors:
                reasons.append("invalid_artifact")
            if invalid:
                reasons.append("invalid_candidate_evaluation")
            if stats["format_compliance_rate"] + 1e-12 < self.min_format_compliance_rate:
                reasons.append("format_compliance_below_minimum")
            diagnostic = {
                "candidate_id": candidate_id,
                "reasons": reasons,
                "plasticity_score": stats["plasticity_score"],
                "stability_score": stats["stability_score"],
                "joint_score": stats["joint_score"],
                "primary_score_delta": stats["primary_score_delta"],
                "anchor_forget_count": stats["anchor_forget_count"],
                "anchor_forget_rate": stats["anchor_forget_rate"],
            }
            diagnostics.append(diagnostic)
            if not reasons:
                scored.append((float(stats["joint_score"]), candidate_id, candidate, diagnostic))
        if not scored:
            return [], diagnostics
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        keep = max(int(self.screening_keep_top_k), 1)
        if len(scored) > keep:
            best_score = scored[0][0]
            close_count = sum(
                best_score - item[0] <= float(self.screening_close_score_margin) + 1e-12
                for item in scored
            )
            if close_count > keep:
                keep = min(max(int(self.screening_close_keep_top_k), keep), close_count)
        promoted_ids = {item[1] for item in scored[:keep]}
        for diagnostic in diagnostics:
            diagnostic["promoted"] = str(diagnostic.get("candidate_id", "")) in promoted_ids
        return [item[2] for item in scored[:keep]], diagnostics

    def balanced_score(
        self,
        candidate_result: dict[str, object],
        current_val_result: dict[str, object],
    ) -> dict[str, float | int | bool]:
        return self._balanced_stats(candidate_result, current_val_result)

    def _select_balanced_candidate(
        self,
        candidates: list[dict[str, object]],
        candidate_results: dict[str, dict[str, object]],
        current_val_result: dict[str, object],
        *,
        anchor_context: dict[str, object],
    ) -> dict[str, object] | None:
        anchor_required = bool(anchor_context.get("required", False))
        anchor_ready = bool(anchor_context.get("ready", not anchor_required))
        expected_anchor_total = int(anchor_context.get("total", 0) or 0)
        rejection_details: list[dict[str, object]] = []
        eligible: list[tuple[dict[str, object], dict[str, float | int | bool]]] = []
        for candidate in candidates:
            candidate_id = str(candidate.get("candidate_id", ""))
            candidate_result = candidate_results.get(candidate_id, {})
            stats = self._balanced_stats(candidate_result, current_val_result)
            reasons: list[str] = []
            validation_errors = candidate.get("validation_errors")
            metrics = candidate_result.get("metrics", {})
            if isinstance(validation_errors, list) and validation_errors:
                reasons.append("invalid_artifact")
            if not isinstance(metrics, dict) or metrics.get("invalid_candidate"):
                reasons.append("invalid_candidate_evaluation")
            if isinstance(metrics, dict) and metrics.get("screening_only"):
                reasons.append("screening_not_promoted")
            if (
                isinstance(metrics, dict)
                and int(metrics.get("behavior_comparable_count", 0) or 0) > 0
                and int(metrics.get("behavior_change_count", 0) or 0) == 0
            ):
                reasons.append("no_behavior_change")
            if not anchor_ready:
                reasons.append("historical_anchor_requirement_not_met")
            if anchor_required:
                required_coverage = math.ceil(expected_anchor_total * self.min_anchor_evaluation_coverage)
                if int(stats["anchor_total"]) < required_coverage:
                    reasons.append("historical_anchor_evaluation_coverage_below_minimum")
            if float(stats["primary_score_delta"]) + 1e-12 < self.min_primary_score_delta:
                reasons.append("primary_score_delta_below_minimum")
            if int(stats["correct_gain"]) < self.min_correct_gain:
                reasons.append("correct_gain_below_minimum")
            if float(stats["format_compliance_rate"]) + 1e-12 < self.min_format_compliance_rate:
                reasons.append("format_compliance_below_minimum")
            if anchor_required:
                if (
                    self.max_anchor_forget_rate is not None
                    and float(stats["anchor_forget_rate"]) > float(self.max_anchor_forget_rate) + 1e-12
                ):
                    reasons.append("historical_anchor_forget_rate_exceeded")
                if (
                    self.max_anchor_forget_count is not None
                    and int(stats["anchor_forget_count"]) > int(self.max_anchor_forget_count)
                ):
                    reasons.append("historical_anchor_forget_count_exceeded")
            if float(stats["joint_score"]) <= float(self.min_joint_score) + 1e-12:
                reasons.append("joint_score_below_minimum")
            detail = {"candidate_id": candidate_id, "reasons": reasons, **stats}
            rejection_details.append(detail)
            if not reasons:
                eligible.append((candidate, stats))
        commit_gate = self._balanced_commit_gate(anchor_context)
        if not eligible:
            self.history.append(
                {
                    "accepted": False,
                    "reason": "no_candidate_passed_commit_gate",
                    "commit_gate": commit_gate,
                    "anchor_context": anchor_context,
                    "candidate_rejections": rejection_details,
                }
            )
            return None

        probabilities: dict[str, float]
        random_draw: float | None = None
        if len(eligible) == 1 or self.selection_mode == "argmax":
            selected, selected_stats = max(
                eligible,
                key=lambda item: (float(item[1]["joint_score"]), str(item[0].get("candidate_id", ""))),
            )
            probabilities = {
                str(candidate.get("candidate_id", "")): float(candidate is selected)
                for candidate, _ in eligible
            }
        else:
            selected, selected_stats, probabilities, random_draw = self._gibbs_sample(eligible)
        selected_id = str(selected.get("candidate_id", ""))
        self.history.append(
            {
                "accepted": True,
                "reason": "candidate_passed_balanced_commit_gate",
                "candidate_id": selected_id,
                "component": selected.get("component"),
                "source_version": selected.get("source_version"),
                "commit_gate": commit_gate,
                "anchor_context": anchor_context,
                "selection_mode": self.selection_mode,
                "selection_probabilities": probabilities,
                "random_draw": random_draw,
                "selection_seed": self.selection_seed,
                "candidate_rejections": rejection_details,
                **selected_stats,
            }
        )
        return selected

    def _balanced_stats(
        self,
        candidate_result: dict[str, object],
        current_val_result: dict[str, object],
    ) -> dict[str, float | int | bool]:
        current_task = _scoped_result(current_val_result, "current_task_metrics")
        candidate_task = _scoped_result(candidate_result, "current_task_metrics")
        candidate_anchor = _scoped_result(candidate_result, "historical_anchor_metrics")
        primary_score_delta = _primary_score(candidate_task) - _primary_score(current_task)
        correct_gain = _metric_int(candidate_task, "correct") - _metric_int(current_task, "correct")
        anchor_total = _metric_int(candidate_anchor, "total")
        anchor_forget_count = _metric_int(candidate_anchor, "forget_count")
        anchor_recovered_count = _metric_int(candidate_anchor, "recovered_count")
        anchor_forget_rate = _metric_float(candidate_anchor, "forget_rate")
        anchor_net_score = (
            float(self.recover_reward) * anchor_recovered_count
            - float(self.forget_penalty) * anchor_forget_count
        ) / max(anchor_total, 1)
        plasticity_score = _clip(primary_score_delta / self.plasticity_scale, -1.0, 1.0)
        stability_active = anchor_total > 0
        stability_score = (
            _clip(anchor_net_score / self.stability_scale, -1.0, 1.0)
            if stability_active
            else 0.0
        )
        joint_score = (
            self.plasticity_weight * plasticity_score
            + (1.0 - self.plasticity_weight) * stability_score
            if stability_active
            else plasticity_score
        )
        return {
            "primary_score_delta": primary_score_delta,
            "correct_gain": correct_gain,
            "plasticity_score": plasticity_score,
            "anchor_total": anchor_total,
            "anchor_forget_count": anchor_forget_count,
            "anchor_recovered_count": anchor_recovered_count,
            "anchor_forget_rate": anchor_forget_rate,
            "anchor_net_score": anchor_net_score,
            "stability_score": stability_score,
            "stability_active": stability_active,
            "joint_score": joint_score,
            "format_compliance_rate": _metric_float(candidate_task, "format_compliance_rate"),
            "behavior_comparable_count": _metric_int(candidate_result, "behavior_comparable_count"),
            "behavior_change_count": _metric_int(candidate_result, "behavior_change_count"),
        }

    def _gibbs_sample(
        self,
        eligible: list[tuple[dict[str, object], dict[str, float | int | bool]]],
    ) -> tuple[dict[str, object], dict[str, float | int | bool], dict[str, float], float]:
        best_score = max(float(stats["joint_score"]) for _, stats in eligible)
        weights = [
            math.exp((float(stats["joint_score"]) - best_score) / self.selection_temperature)
            for _, stats in eligible
        ]
        total_weight = sum(weights)
        probabilities = {
            str(candidate.get("candidate_id", "")): weight / total_weight
            for (candidate, _), weight in zip(eligible, weights)
        }
        seed_material = "|".join(
            sorted(str(candidate.get("candidate_id", "")) for candidate, _ in eligible)
        )
        seed_offset = int(sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(int(self.selection_seed) + seed_offset)
        draw = rng.random()
        cumulative = 0.0
        for item, weight in zip(eligible, weights):
            cumulative += weight / total_weight
            if draw <= cumulative + 1e-15:
                return item[0], item[1], probabilities, draw
        return eligible[-1][0], eligible[-1][1], probabilities, draw

    def _balanced_commit_gate(self, anchor_context: dict[str, object]) -> dict[str, object]:
        return {
            "selection_objective": self.selection_objective,
            "commit_policy": self.commit_policy,
            "plasticity_weight": self.plasticity_weight,
            "plasticity_scale": self.plasticity_scale,
            "stability_scale": self.stability_scale,
            "min_primary_score_delta": self.min_primary_score_delta,
            "min_correct_gain": self.min_correct_gain,
            "min_format_compliance_rate": self.min_format_compliance_rate,
            "recover_reward": self.recover_reward,
            "forget_penalty": self.forget_penalty,
            "max_anchor_forget_rate": self.max_anchor_forget_rate,
            "max_anchor_forget_count": self.max_anchor_forget_count,
            "min_joint_score": self.min_joint_score,
            "selection_mode": self.selection_mode,
            "selection_temperature": self.selection_temperature,
            "anchor_context": anchor_context,
        }

def _append_jsonl_and_pretty(
    path: Path,
    payload: dict[str, object],
    *,
    write_pretty: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    if not write_pretty:
        return
    pretty_path = path.with_suffix(".json")
    records: list[object] = []
    if pretty_path.exists():
        try:
            loaded = json.loads(pretty_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                records = loaded
        except json.JSONDecodeError:
            records = []
    records.append(payload)
    atomic_write_json(pretty_path, records)


def _normalize_artifact_content(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    content = value
    nested_content = value.get("content")
    if isinstance(nested_content, dict) and isinstance(nested_content.get("sections"), list):
        content = nested_content
    sections = content.get("sections")
    if not isinstance(sections, list) or not any(str(section).strip() for section in sections):
        return {}
    return {
        "template_id": str(content.get("template_id") or "candidate_prompt_template"),
        "description": str(content.get("description") or "Candidate prompt template."),
        "sections": [str(section) for section in sections if str(section).strip()],
        **({"answer_instructions": content["answer_instructions"]} if isinstance(content.get("answer_instructions"), dict) else {}),
    }
def _primary_score(result: dict[str, object]) -> float:
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        return 0.0
    value = metrics.get("primary_score")
    return float(value or 0.0)


def _metric_int(result: dict[str, object], name: str) -> int:
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        return 0
    return int(metrics.get(name, 0) or 0)


def _metric_float(result: dict[str, object], name: str) -> float:
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        return 0.0
    return float(metrics.get(name, 0.0) or 0.0)


def _metric_value(result: dict[str, object], name: str, default: object = None) -> object:
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        return default
    return metrics.get(name, default)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def _scoped_result(result: dict[str, object], key: str) -> dict[str, object]:
    scoped_metrics = result.get(key)
    if isinstance(scoped_metrics, dict):
        return {"metrics": scoped_metrics}
    return result


def _artifact_validation_errors(component: str, content: dict[str, object]) -> list[str]:
    sections = content.get("sections", [])
    text = "\n".join(str(section) for section in sections).lower() if isinstance(sections, list) else ""
    required_placeholders = {
        "task_interface_structuring": {"{raw_visible_input}"},
        "router_workflow": {"{model_visible_input}", "{selected_skills}"},
        "memory_selector": {
            "{current_input}",
            "{workflow_decision}",
            "{memory_candidates}",
            "{retrieval_top_k}",
        },
        "skill_selector": {
            "{model_visible_input}",
            "{skill_candidates}",
        },
        "tool_selector": {
            "{model_visible_input}",
            "{workflow_decision}",
            "{selected_skills}",
            "{tool_candidates}",
            "{heuristic_tools}",
        },
        "router_context": {
            "{model_visible_input}",
            "{workflow_decision}",
            "{selected_memory}",
            "{selected_skills}",
            "{selected_tools}",
        },
    }.get(component, set())
    errors = [
        f"missing_placeholder:{placeholder}"
        for placeholder in sorted(required_placeholders)
        if placeholder not in text
    ]
    render_values = {
        "model_visible_input": "{}",
        "raw_visible_input": "{}",
        "current_input": "{}",
        "workflow_decision": "{}",
        "memory_candidates": "[]",
        "retrieval_top_k": 0,
        "skill_candidates": "[]",
        "tool_candidates": "[]",
        "heuristic_tools": "[]",
        "selected_memory": "[]",
        "selected_skills": "[]",
        "selected_tools": "[]",
        "available_capabilities": "[]",
        "capability_results": "[]",
    }
    try:
        PromptTemplate.from_dict(content).render(render_values)
    except (KeyError, ValueError, IndexError) as exc:
        errors.append(f"prompt_not_renderable:{type(exc).__name__}:{exc}")
    if component == "task_interface_structuring":
        for field_name in ("evidence", "recognizable_goal", "recognizable_constraints"):
            if field_name not in text:
                errors.append(f"missing_hcl_interface_field:{field_name}")
        for forbidden_field in ("answer_policy", "input_characteristics", "downstream_hints"):
            if forbidden_field in text:
                errors.append(f"non_minimal_hcl_interface_field:{forbidden_field}")
    if component == "memory_selector":
        if "selected" not in text or "memory_id" not in text:
            errors.append("missing_memory_selection_schema")
        if "zero" not in text and "select nothing" not in text:
            errors.append("selector_must_allow_empty_selection")
    if component == "skill_selector":
        if "selected" not in text or "skill_name" not in text:
            errors.append("missing_skill_selection_schema")
        if "zero" not in text and "select nothing" not in text:
            errors.append("selector_must_allow_empty_selection")
    if component == "tool_selector":
        if "selected" not in text or "tool_name" not in text:
            errors.append("missing_tool_selection_schema")
        if "zero" not in text and "select nothing" not in text:
            errors.append("selector_must_allow_empty_selection")
    if component != "router_context":
        return errors
    if not all(token in text for token in ("correct_feedback", "previous_attempt")):
        errors.append("missing_memory_safety_fields")
    if "may be incorrect" not in text and "can be incorrect" not in text:
        errors.append("missing_incorrect_attempt_warning")
    if "final answer only" not in text and "output only the final" not in text:
        errors.append("missing_final_only_rule")
    forbidden = (
        "show your work",
        "show your reasoning",
        "step-by-step reasoning",
        "reason step-by-step",
        "provide a step-by-step",
        "show the derivation",
    )
    for phrase in forbidden:
        if phrase in text:
            errors.append(f"visible_reasoning_forbidden:{phrase}")
    return errors


def _extract_json_with_diagnostics(text: str) -> tuple[dict[str, Any], str, str | None]:
    cleaned = str(text).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    parsed, error = _decode_json_object(cleaned)
    if parsed is not None:
        status = "valid" if cleaned.startswith("{") and cleaned.endswith("}") else "extracted"
        return parsed, status, None
    repaired = _repair_common_candidate_json(cleaned)
    if repaired != cleaned:
        parsed, repaired_error = _decode_json_object(repaired)
        if parsed is not None:
            return parsed, "repaired", None
        error = repaired_error or error
    return {}, "failed", error


def _decode_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    attempts = [text]
    start = text.find("{")
    if start > 0:
        attempts.append(text[start:])
    last_error: str | None = None
    for candidate in attempts:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = f"{exc.msg} at line {exc.lineno} column {exc.colno}"
            try:
                data, _ = json.JSONDecoder().raw_decode(candidate)
            except json.JSONDecodeError:
                continue
        if isinstance(data, dict):
            return data, None
        last_error = "top-level JSON value is not an object"
    return None, last_error


def _repair_common_candidate_json(text: str) -> str:
    """Repair one observed bracket transposition without guessing candidate semantics."""
    repaired = text.replace(
        '"{memory_candidates}"}],"artifact_diff"',
        '"{memory_candidates}"]},"artifact_diff"',
    )
    return repaired.replace(
        '"{memory_candidates}"}],"artifact_diff_summary"',
        '"{memory_candidates}"],"artifact_diff_summary"',
    )
