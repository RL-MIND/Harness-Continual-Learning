from __future__ import annotations

import json
from typing import Any

from .base_manager import BaseManager
from .capability import CapabilityMap
from .completion_contract import (
    build_completion_contract,
    contract_progress,
    evaluate_completion_contract,
)
from .common import event_digest
from .decomposer import TaskDecomposer
from .evaluator import ContinualEvaluator
from .health import SkillHealthStore
from .memory import ExperienceMemory
from .optimizer import ContinualOptimizer
from .router import AdaptiveRouter
from .synthesizer import SkillSynthesizer
from .task_interface import TaskInterface
from .telemetry import ExecutionRecorder, inventory_from
from .tester import SkillTestGate

class HarnessOrchestrator:
    def __init__(
        self,
        *,
        odyssey,
        model_name: str,
        ckpt_dir: str = "ckpt",
        resume: bool = True,
        memory_config_path: str | None = None,
    ):
        self.odyssey = odyssey
        self.model_name = model_name
        self.memory = ExperienceMemory(
            ckpt_dir=ckpt_dir,
            resume=resume,
            embedding_model=getattr(odyssey.skill_manager, "embedding_model", None),
            backend_config=memory_config_path,
        )
        self.capabilities = CapabilityMap(
            odyssey.skill_manager,
            ckpt_dir=ckpt_dir,
            include_harness_skills=self.memory.use_harness_learned_skills,
        )
        self.task_interface = TaskInterface(model_name=model_name)
        self.decomposer = TaskDecomposer(model_name=model_name)
        self.router = AdaptiveRouter(model_name=model_name)
        self.evaluator = ContinualEvaluator(model_name=model_name)
        self.optimizer = ContinualOptimizer(model_name=model_name, ckpt_dir=ckpt_dir)
        self.synthesizer = SkillSynthesizer(model_name=model_name)
        self.test_gate = SkillTestGate(ckpt_dir=ckpt_dir, resume=resume)
        self.execution_recorder = ExecutionRecorder(ckpt_dir=ckpt_dir, resume=resume)
        self.health = SkillHealthStore(ckpt_dir=ckpt_dir, resume=resume)
        self.base_manager = BaseManager(ckpt_dir=ckpt_dir, resume=resume)
        self.refresh_prompt_overrides()

    def refresh_prompt_overrides(self) -> None:
        overrides = self.optimizer.prompt_overrides if self.memory.prompt_updates_enabled else {}
        self.task_interface.system_prompt = self.task_interface.base_prompt
        self.router.system_prompt = self.router.base_prompt
        self.evaluator.task_prompt = self.evaluator.base_task_prompt
        self.evaluator.update_prompt = self.evaluator.base_update_prompt
        self.optimizer.system_prompt = self.optimizer.base_prompt
        if "task_interface" in overrides:
            self.task_interface.system_prompt = (
                self.task_interface.system_prompt
                + "\n\nApproved continual update:\n"
                + overrides["task_interface"].get("instruction", "")
            )
        if "adaptive_router" in overrides:
            self.router.system_prompt = (
                self.router.system_prompt
                + "\n\nApproved continual update:\n"
                + overrides["adaptive_router"].get("instruction", "")
            )
        if "continual_evaluator" in overrides:
            self.evaluator.task_prompt = (
                self.evaluator.task_prompt
                + "\n\nApproved continual update:\n"
                + overrides["continual_evaluator"].get("instruction", "")
            )
            self.evaluator.update_prompt = (
                self.evaluator.update_prompt
                + "\n\nApproved continual update:\n"
                + overrides["continual_evaluator"].get("instruction", "")
            )
        if "continual_optimizer" in overrides:
            self.optimizer.system_prompt = (
                self.optimizer.system_prompt
                + "\n\nApproved continual update:\n"
                + overrides["continual_optimizer"].get("instruction", "")
            )
        disabled_outputs = []
        if not self.memory.skill_updates_enabled:
            disabled_outputs.append("capability_updates must be []")
        if not self.memory.prompt_updates_enabled:
            disabled_outputs.append("prompt_updates must be {}")
        if disabled_outputs:
            self.optimizer.system_prompt += (
                "\n\nFixed experiment update policy: "
                + "; ".join(disabled_outputs)
                + ". Only propose enabled update families."
            )

    def _execute_protected_code(self, parsed: dict[str, Any], *, skill_name: str) -> dict[str, Any]:
        """Execute one Harness action with current persistent assets guarded."""
        guarded = dict(parsed)
        guarded["exec_code"] = self.base_manager.protection_call() + "\n" + parsed["exec_code"]
        return self.odyssey.execute_harness_code(guarded, skill_name=skill_name)

    def _execute_protected_skill(self, skill_name: str) -> dict[str, Any]:
        if skill_name not in self.odyssey.skill_manager.skills:
            raise KeyError(f"Unknown harness skill: {skill_name}")
        parsed = self.odyssey._parse_skill_code(
            self.odyssey.skill_manager.skills[skill_name]["code"]
        )
        return self._execute_protected_code(parsed, skill_name=skill_name)

    def _duplicate_primitive_is_blocked(
        self,
        primitive_key: str,
        failed_primitive_attempts: set[str],
    ) -> bool:
        return (
            self.memory.retry_suppression_enabled
            and primitive_key in failed_primitive_attempts
        )

    def _planning_feedback(self, last_feedback: dict[str, Any] | None) -> dict[str, Any] | None:
        if not self.memory.condition_on_failure_feedback:
            return None
        return last_feedback

    def _planning_events(
        self,
        events: list[tuple[str, dict[str, Any]]] | None,
    ) -> list[tuple[str, dict[str, Any]]]:
        if self.memory.condition_on_failure_feedback:
            return list(events or [])
        # Retry-only samples may observe the changed Minecraft state, but must
        # not receive the previous attempt's chat/error/save transcript. The
        # latest observe event contains inventory, status, nearby blocks and
        # other world evidence needed for a fresh generation.
        for event in reversed(events or []):
            if event[0] == "observe":
                return [event]
        return []

    def run_instruction(
        self,
        instruction: str,
        *,
        events: list[tuple[str, dict[str, Any]]] | None = None,
        reset_env: bool = False,
        reset_mode: str = "soft",
        max_steps: int = 3,
        max_replans: int = 2,
        completion_contract: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if reset_env or events is None:
            events = self.odyssey.env.reset(
                options={
                    "mode": reset_mode,
                    "wait_ticks": self.odyssey.env_wait_ticks,
                    "username": self.odyssey.username,
                }
            )
            self.odyssey.last_events = events

        completion_contract = completion_contract or build_completion_contract(
            instruction,
            inventory_from(events),
            model_name=self.model_name,
        )
        initial_progress = contract_progress(completion_contract, inventory_from(events))
        initial_abstract = self.memory.retrieve_abstract(instruction)
        initial_capabilities = self.capabilities.summarize(instruction)
        subgoals = self.decomposer.decompose(
            instruction=instruction,
            events=self._planning_events(events),
            abstract_memory=initial_abstract,
            capability_summary=initial_capabilities,
            completion_contract=completion_contract,
            contract_progress=initial_progress,
        )
        final_result: dict[str, Any] = {"success": False, "subgoals": subgoals}
        completed_subgoals: list[str] = []
        trajectory: list[dict[str, Any]] = []
        # HCL updates are task-boundary work. Keep raw records while acting,
        # then optimize one representative trajectory after the task reaches a
        # terminal outcome. This prevents learning calls from blocking every
        # ordinary Minecraft action.
        learning_records: list[dict[str, Any]] = []
        # These blocks are scoped to one top-level instruction. They prevent a
        # failed capability from consuming every retry, without globally
        # deleting a skill that might be valid in another environment.
        diagnostic_names = set(self.capabilities.diagnostic_skill_names())
        # A diagnostic is an observation aid, not a state-changing capability.
        # Historical runtime failures must not remove the only way to inspect a
        # new world state; it still has the per-subgoal diagnostic-step budget.
        blocked_skills = (
            {
                name: context
                for name, context in self.health.repeatedly_failed_contexts().items()
                if name not in diagnostic_names
            }
            if self.memory.skill_health_updates_enabled and self.memory.retry_suppression_enabled
            else {}
        )
        subgoal_index = 0
        replan_count = 0
        while subgoal_index < len(subgoals):
            active_instruction = subgoals[subgoal_index]
            last_feedback: dict[str, Any] | None = None
            subgoal_result: dict[str, Any] = {}
            action_steps = 0
            diagnostic_steps = 0
            failed_primitive_attempts: set[str] = set()
            # Diagnostics are observation aids, not task-progress actions. Give
            # them a small separate budget so one failed diagnostic cannot
            # consume the actions needed to satisfy a resource contract.
            for step_index in range(max_steps + 2):
                if action_steps >= max_steps:
                    break
                contract_result = evaluate_completion_contract(completion_contract, inventory_from(events))
                if contract_result and contract_result["success"]:
                    evaluation = {
                        "success": True,
                        "critique": contract_result["critique"],
                        "evidence": contract_result["evidence"],
                        "regression_risks": contract_result["regression_risks"],
                        "contract_evaluation": contract_result,
                    }
                    subgoal_result = {
                        "success": True,
                        "needs_user": False,
                        "task_state": {
                            "goal": active_instruction,
                            "instruction": active_instruction,
                            "completion_contract": completion_contract,
                            "contract_progress": contract_progress(completion_contract, inventory_from(events)),
                        },
                        "route": {
                            "decision": "contract_satisfied",
                            "skill_name": "",
                            "reasoning": "Fixed completion contract is already satisfied.",
                        },
                        "evaluation": evaluation,
                        "inventory": inventory_from(events),
                    }
                    trajectory.append(
                        {
                            "type": "contract_check",
                            "original_instruction": instruction,
                            "subgoal": active_instruction,
                            "subgoal_index": subgoal_index,
                            "step_index": step_index,
                            "replan_count": replan_count,
                            "evaluation": evaluation,
                            "success": True,
                        }
                    )
                    break
                planning_feedback = self._planning_feedback(last_feedback)
                query = (
                    active_instruction
                    if not planning_feedback
                    else f"{active_instruction}\n{planning_feedback}"
                )
                abstract = self.memory.retrieve_abstract(query)
                capability_summary = self.capabilities.summarize(query)
                capability_summary["update_policy"] = dict(self.memory.update_policy)
                candidate_names = capability_summary.get("candidate_skill_names", [])
                capability_summary["skill_health"] = (
                    self.health.summarize(candidate_names)
                    if self.memory.skill_health_updates_enabled
                    else {}
                )
                capability_summary["blocked_skill_names"] = sorted(blocked_skills)
                capability_summary["scenario_failures"] = list(blocked_skills.values())
                capability_summary["base_state"] = self.base_manager.public_state()
                task_state = self.task_interface.structure(
                    instruction=active_instruction,
                    events=self._planning_events(events),
                    tool_returns=[planning_feedback] if planning_feedback else [],
                    abstract_memory=abstract,
                    capability_summary=capability_summary,
                    completion_contract=completion_contract,
                    contract_progress=contract_progress(completion_contract, inventory_from(events)),
                )
                task_state["completion_contract"] = completion_contract
                task_state["contract_progress"] = contract_progress(completion_contract, inventory_from(events))
                task_state["base_state"] = self.base_manager.public_state()
                if task_state.get("task_kind") == "observation":
                    observation_route = {
                        "decision": "observation_check",
                        "skill_name": "",
                        "reasoning": "The model classified this as a read-only state check.",
                    }
                    evaluation = self.evaluator.judge_task(
                        task_state=task_state,
                        route=observation_route,
                        events=events,
                        inventory=inventory_from(events),
                    )
                    trajectory.append(
                        {
                            "type": "observation_check",
                            "original_instruction": instruction,
                            "subgoal": active_instruction,
                            "subgoal_index": subgoal_index,
                            "step_index": step_index,
                            "replan_count": replan_count,
                            "route": observation_route,
                            "evaluation": evaluation,
                            "success": evaluation["success"],
                        }
                    )
                    if evaluation["success"]:
                        subgoal_result = {
                            "success": True,
                            "needs_user": False,
                            "task_state": task_state,
                            "route": observation_route,
                            "evaluation": evaluation,
                            "inventory": inventory_from(events),
                        }
                        break
                    last_feedback = evaluation
                route = self._primitive_contract_fallback(
                    capability_summary=capability_summary,
                    completion_contract=completion_contract,
                    inventory=inventory_from(events),
                    blocked_skills=blocked_skills,
                )
                if route is None:
                    route = self.router.route(
                        task_state=task_state,
                        abstract_memory=abstract,
                        capability_summary=capability_summary,
                        last_feedback=planning_feedback,
                        completion_contract=completion_contract,
                        contract_progress=task_state["contract_progress"],
                    )

                asset_intent = route.get("persistent_asset", {})
                if asset_intent.get("action") == "place_or_reuse":
                    asset_name = asset_intent.get("block_name", "")
                    asset_request = self.base_manager.maintain(asset_name, events)
                    if asset_request is None:
                        last_feedback = {
                            "success": False,
                            "critique": "Persistent asset name was invalid; choose a concrete Minecraft block name.",
                        }
                        continue
                    events_before = events
                    action_steps += 1
                    try:
                        parsed = self.odyssey._parse_skill_code(asset_request["code"])
                        exec_result = self._execute_protected_code(
                            parsed, skill_name="persistentWorldAsset"
                        )
                    except Exception as exc:  # noqa: BLE001
                        exec_result = {"skill_name": "persistentWorldAsset", "events": events, "error": str(exc)}
                    events = exec_result.get("events") or events
                    self.base_manager.reconcile(events)
                    inventory = inventory_from(events)
                    asset_ready = self.base_manager.asset_ready(asset_name)
                    evaluation = {
                        "success": asset_ready and not exec_result.get("error"),
                        "critique": (
                            "Model-designated persistent asset was verified at its registered coordinate."
                            if asset_ready and not exec_result.get("error")
                            else "Persistent asset was not verified at its registered coordinate."
                        ),
                        "evidence": [self.base_manager.public_state()],
                        "regression_risks": [],
                    }
                    if exec_result.get("error"):
                        evaluation["critique"] += f" Execution error: {exec_result['error']}"
                    structured_execution = self.execution_recorder.build_record(
                        instruction=active_instruction,
                        step_index=step_index,
                        route=route,
                        events_before=events_before,
                        events_after=events,
                        execution=exec_result,
                        evaluation=evaluation,
                    )
                    self.execution_recorder.append(structured_execution)
                    trajectory.append({
                        "type": "persistent_asset", "original_instruction": instruction,
                        "subgoal": active_instruction, "subgoal_index": subgoal_index,
                        "step_index": step_index, "route": route,
                        "execution": structured_execution, "evaluation": evaluation,
                        "success": evaluation["success"],
                    })
                    subgoal_result = {
                        "success": evaluation["success"], "needs_user": False,
                        "task_state": task_state, "route": route, "evaluation": evaluation,
                        "inventory": inventory,
                    }
                    last_feedback = evaluation
                    if evaluation["success"]:
                        break
                    continue

                requirements = self.capabilities.world_asset_requirements(route)
                if requirements:
                    registered_assets, missing_roles = self.base_manager.assets_for_requirements(requirements)
                    if missing_roles:
                        last_feedback = {
                            "success": False,
                            "critique": "Selected capability requires unregistered world assets: " + ", ".join(missing_roles),
                            "base_state": self.base_manager.public_state(),
                        }
                        trajectory.append({
                            "type": "missing_world_asset", "original_instruction": instruction,
                            "subgoal": active_instruction, "subgoal_index": subgoal_index,
                            "step_index": step_index, "route": route,
                            "missing_roles": missing_roles, "success": False,
                        })
                        continue
                    try:
                        parsed = self.odyssey._parse_skill_code(self.base_manager.reach_code(registered_assets))
                        reach_result = self._execute_protected_code(parsed, skill_name="reachRegisteredAsset")
                    except Exception as exc:  # noqa: BLE001
                        reach_result = {"events": events, "error": str(exc)}
                    events = reach_result.get("events") or events
                    self.base_manager.reconcile(events)
                    if reach_result.get("error"):
                        last_feedback = {
                            "success": False,
                            "critique": "Registered world asset could not be reached: " + reach_result["error"],
                        }
                        continue

                if route["decision"] == "blocked_skill":
                    last_feedback = {
                        "success": False,
                        "critique": route.get("reasoning", "Blocked skill selected."),
                        "blocked_skill_names": sorted(blocked_skills),
                    }
                    trajectory.append(
                        {
                            "type": "blocked_route",
                            "original_instruction": instruction,
                            "subgoal": active_instruction,
                            "subgoal_index": subgoal_index,
                            "step_index": step_index,
                            "replan_count": replan_count,
                            "route": route,
                            "success": False,
                        }
                    )
                    continue

                if route["decision"] == "ask_user":
                    guarded_route = self._guard_unsatisfied_contract(
                        route,
                        completion_contract,
                        task_state["contract_progress"],
                        capability_summary,
                    )
                    if guarded_route is not None:
                        route = guarded_route
                    else:
                        trajectory.append(
                            {
                                "type": "attempt",
                                "original_instruction": instruction,
                                "subgoal": active_instruction,
                                "subgoal_index": subgoal_index,
                                "step_index": step_index,
                                "replan_count": replan_count,
                                "route": route,
                                "success": False,
                                "needs_user": True,
                            }
                        )
                        subgoal_result = {
                            "success": False,
                            "needs_user": True,
                            "task_state": task_state,
                            "route": route,
                            "inventory": inventory_from(events),
                            "message": route.get("reasoning", "Harness needs more information."),
                        }
                        break
                if route["decision"] == "request_new_skill":
                    created = self._try_synthesize_missing_skill(task_state, route, capability_summary)
                    raw_record = self.memory.append_raw(
                        {
                            "instruction": active_instruction,
                            "original_instruction": instruction,
                            "subgoal_index": subgoal_index,
                            "task_state": task_state,
                            "route": route,
                            "events_before": event_digest(events),
                            "execution": {
                                "skipped": True,
                                "reason": "missing capability",
                                "created_skill": created,
                            },
                        }
                    )
                    evaluation = {
                        "success": False,
                        "critique": "No existing skill was selected; router requested a new capability.",
                        "evidence": [],
                        "regression_risks": [],
                    }
                    trajectory.append(
                        {
                            "type": "attempt",
                            "original_instruction": instruction,
                            "subgoal": active_instruction,
                            "subgoal_index": subgoal_index,
                            "step_index": step_index,
                            "replan_count": replan_count,
                            "route": route,
                            "evaluation": evaluation,
                            "created_skill": created,
                            "success": False,
                        }
                    )
                    learning_records.append(
                        {
                            "task_state": task_state,
                            "route": route,
                            "raw_record": raw_record,
                            "evaluation": evaluation,
                            "capability_summary": capability_summary,
                        }
                    )
                    if created.get("applied"):
                        last_feedback = {
                            "success": False,
                            "critique": f"Created learned skill {created.get('name')}; route again to execute it.",
                        }
                        continue
                    subgoal_result = {
                        "success": False,
                        "needs_user": False,
                        "task_state": task_state,
                        "route": route,
                        "evaluation": evaluation,
                        "created_skill": created,
                    }
                    break

                if route["decision"] == "use_primitive":
                    primitive_name = route.get("primitive_name", "")
                    primitive_args = route.get("primitive_args", [])
                    primitive_key = json.dumps(
                        [primitive_name, primitive_args], ensure_ascii=False, sort_keys=True, default=str
                    )
                    if self._duplicate_primitive_is_blocked(
                        primitive_key,
                        failed_primitive_attempts,
                    ):
                        last_feedback = {
                            "success": False,
                            "critique": (
                                "Blocked an identical primitive retry because the previous call used the same "
                                "arguments and left inventory unchanged. Change the arguments or strategy. "
                                "For craftItem, count means recipe executions, not desired output quantity."
                            ),
                            "blocked_primitive": primitive_name,
                            "blocked_args": primitive_args,
                        }
                        continue
                    temporary = self.synthesizer.render_ephemeral_primitive(
                        primitive_name,
                        primitive_args,
                    )
                    if temporary is None:
                        last_feedback = {
                            "success": False,
                            "critique": "Primitive arguments failed the allow-listed plan validator; choose a skill or request a composition.",
                        }
                        continue
                    primitive_test = self.test_gate.validate_update(
                        temporary,
                        known_skill_names=set(capability_summary.get("all_skill_names", [])),
                    )
                    if not primitive_test.get("passed"):
                        last_feedback = {
                            "success": False,
                            "critique": "Primitive plan failed static validation; choose a skill or request a composition.",
                        }
                        continue
                    events_before = events
                    action_steps += 1
                    try:
                        parsed = self.odyssey._parse_skill_code(temporary["code"])
                        exec_result = self._execute_protected_code(
                            parsed,
                            skill_name=f"primitive:{primitive_name}",
                        )
                    except Exception as exc:  # noqa: BLE001
                        exec_result = {
                            "skill_name": f"primitive:{primitive_name}",
                            "events": events,
                            "error": str(exc),
                        }
                    events = exec_result.get("events") or events
                    self.base_manager.reconcile(events)
                    inventory = inventory_from(events)
                    if (
                        self.memory.retry_suppression_enabled
                        and inventory == inventory_from(events_before)
                    ):
                        failed_primitive_attempts.add(primitive_key)
                    contract_evaluation = evaluate_completion_contract(completion_contract, inventory)
                    if contract_evaluation is not None:
                        evaluation = {
                            "success": contract_evaluation["success"],
                            "critique": contract_evaluation["critique"],
                            "evidence": contract_evaluation["evidence"],
                            "regression_risks": contract_evaluation["regression_risks"],
                            "contract_evaluation": contract_evaluation,
                        }
                    else:
                        evaluation = self.evaluator.judge_task(
                            task_state=task_state, route=route, events=events, inventory=inventory
                        )
                    if exec_result.get("error"):
                        evaluation["success"] = False
                        evaluation["critique"] = (
                            evaluation.get("critique", "") + f" Execution error: {exec_result['error']}"
                        ).strip()
                    structured_execution = self.execution_recorder.build_record(
                        instruction=active_instruction,
                        step_index=step_index,
                        route=route,
                        events_before=events_before,
                        events_after=events,
                        execution=exec_result,
                        evaluation=evaluation,
                    )
                    self.execution_recorder.append(structured_execution)
                    trajectory.append({
                        "type": "attempt", "original_instruction": instruction,
                        "subgoal": active_instruction, "subgoal_index": subgoal_index,
                        "step_index": step_index, "replan_count": replan_count,
                        "route": route, "execution": structured_execution,
                        "evaluation": evaluation, "success": evaluation["success"],
                    })
                    if self.memory.skill_health_updates_enabled:
                        self.health.update(f"primitive:{primitive_name}", evaluation["success"], evaluation)
                    raw_record = self.memory.append_raw({
                        "instruction": active_instruction, "original_instruction": instruction,
                        "subgoal_index": subgoal_index, "step_index": step_index,
                        "task_state": task_state, "route": route,
                        "execution": exec_result, "structured_execution": structured_execution,
                        "evaluation": evaluation, "completion_contract": completion_contract,
                        "events_after": event_digest(events),
                    })
                    learning_records.append({
                        "task_state": task_state, "route": route, "raw_record": raw_record,
                        "evaluation": evaluation, "capability_summary": capability_summary,
                    })
                    subgoal_result = {
                        "success": evaluation["success"], "needs_user": False,
                        "task_state": task_state, "route": route, "evaluation": evaluation,
                        "inventory": inventory, "raw_record": raw_record,
                    }
                    last_feedback = evaluation
                    if evaluation["success"]:
                        break
                    continue

                events_before = events
                is_diagnostic = route.get("skill_name") in self.capabilities.diagnostic_skill_names()
                if is_diagnostic and self.memory.retry_suppression_enabled:
                    if diagnostic_steps >= 1:
                        last_feedback = {
                            "success": False,
                            "critique": "Diagnostic budget exhausted; choose an executable progress-making capability.",
                        }
                        continue
                    diagnostic_steps += 1
                else:
                    action_steps += 1
                try:
                    exec_result = self._execute_protected_skill(route["skill_name"])
                except Exception as exc:  # noqa: BLE001
                    exec_result = {
                        "skill_name": route["skill_name"],
                        "events": events,
                        "error": str(exc),
                    }
                events = exec_result.get("events") or events
                self.base_manager.reconcile(events)
                inventory = inventory_from(events)
                contract_evaluation = evaluate_completion_contract(completion_contract, inventory)
                if contract_evaluation is not None:
                    # A fixed contract is already an observation-grounded
                    # success predicate. Calling an LLM here adds latency but
                    # cannot improve the truth value of the inventory check.
                    evaluation = {
                        "success": contract_evaluation["success"],
                        "critique": contract_evaluation["critique"],
                        "evidence": contract_evaluation["evidence"],
                        "regression_risks": contract_evaluation["regression_risks"],
                        "contract_evaluation": contract_evaluation,
                    }
                else:
                    evaluation = self.evaluator.judge_task(
                        task_state=task_state,
                        route=route,
                        events=events,
                        inventory=inventory,
                    )
                if exec_result.get("error"):
                    contract_result = evaluation.get("contract_evaluation")
                    if not (isinstance(contract_result, dict) and contract_result.get("success")):
                        evaluation["success"] = False
                    evaluation["critique"] = (
                        evaluation.get("critique", "") + f" Execution error: {exec_result['error']}"
                    ).strip()
                structured_execution = self.execution_recorder.build_record(
                    instruction=active_instruction,
                    step_index=step_index,
                    route=route,
                    events_before=events_before,
                    events_after=events,
                    execution=exec_result,
                    evaluation=evaluation,
                )
                self.execution_recorder.append(structured_execution)
                trajectory.append(
                    {
                        "type": "attempt",
                        "original_instruction": instruction,
                        "subgoal": active_instruction,
                        "subgoal_index": subgoal_index,
                        "step_index": step_index,
                        "replan_count": replan_count,
                        "route": route,
                        "execution": structured_execution,
                        "evaluation": evaluation,
                        "success": evaluation["success"],
                    }
                )
                failure_context = self._failure_context_if_blocked(
                    route=route,
                    evaluation=evaluation,
                    events_before=events_before,
                    events_after=events,
                    completion_contract=completion_contract,
                )
                if failure_context and self.memory.retry_suppression_enabled:
                    blocked_skills[route.get("skill_name", "")] = failure_context
                    risks = evaluation.get("regression_risks")
                    if not isinstance(risks, list):
                        risks = []
                        evaluation["regression_risks"] = risks
                    risks.append(
                        f"Blocked {route.get('skill_name')} for the remainder of this task after no contract progress."
                    )
                if self.memory.skill_health_updates_enabled:
                    self.health.update(
                        route.get("skill_name", ""),
                        evaluation["success"],
                        evaluation,
                        failure_context=failure_context,
                    )
                raw_record = self.memory.append_raw(
                    {
                        "instruction": active_instruction,
                        "original_instruction": instruction,
                        "subgoal_index": subgoal_index,
                        "step_index": step_index,
                        "task_state": task_state,
                        "route": route,
                        "execution": exec_result,
                        "structured_execution": structured_execution,
                        "evaluation": evaluation,
                        "completion_contract": completion_contract,
                        "events_after": event_digest(events),
                    }
                )
                learning_records.append(
                    {
                        "task_state": task_state,
                        "route": route,
                        "raw_record": raw_record,
                        "evaluation": evaluation,
                        "capability_summary": capability_summary,
                    }
                )
                subgoal_result = {
                    "success": evaluation["success"],
                    "needs_user": False,
                    "task_state": task_state,
                    "route": route,
                    "evaluation": evaluation,
                    "inventory": inventory,
                    "raw_record": raw_record,
                }
                last_feedback = evaluation
                if evaluation["success"]:
                    break
            final_result = {
                **subgoal_result,
                "subgoals": subgoals,
                "completed_subgoals": completed_subgoals,
                "subgoal_index": subgoal_index,
                "replan_count": replan_count,
                "trajectory": trajectory,
            }
            if final_result.get("needs_user"):
                break
            if final_result.get("success"):
                completed_subgoals.append(active_instruction)
                final_result["completed_subgoals"] = completed_subgoals
                subgoal_index += 1
                continue
            if replan_count >= max_replans:
                break
            if self.memory.condition_on_failure_feedback:
                replanned = self._replan_after_failure(
                    original_instruction=instruction,
                    failed_subgoal=active_instruction,
                    completed_subgoals=completed_subgoals,
                    failure_result=final_result,
                    events=events,
                    completion_contract=completion_contract,
                )
            else:
                replanned = self._replan_from_world_state(
                    original_instruction=instruction,
                    events=events,
                    completion_contract=completion_contract,
                )
            replan_count += 1
            trajectory.append(
                {
                    "type": "replan",
                    "original_instruction": instruction,
                    "failed_subgoal": active_instruction,
                    "subgoal_index": subgoal_index,
                    "replan_count": replan_count,
                    "new_subgoals": replanned,
                    "failure": {
                        "route": final_result.get("route", {}),
                        "evaluation": final_result.get("evaluation", {}),
                    },
                }
            )
            if replanned:
                if self.memory.condition_on_failure_feedback:
                    subgoals = completed_subgoals + replanned
                    subgoal_index = len(completed_subgoals)
                else:
                    # A retry-only replan is a fresh sample conditioned only
                    # on the current world. Do not carry textual plan history
                    # into the next decomposition/execution cycle.
                    subgoals = replanned
                    completed_subgoals = []
                    subgoal_index = 0
                final_result["subgoals"] = subgoals
                final_result["replanned_remaining_subgoals"] = replanned
                continue
            break
        final_result["trajectory"] = trajectory
        self._optimize_at_task_boundary(learning_records)
        return final_result

    def _primitive_contract_fallback(
        self,
        *,
        capability_summary: dict[str, Any],
        completion_contract: dict[str, Any] | None,
        inventory: dict[str, Any],
        blocked_skills: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Prefer a declared primitive fallback after a no-progress skill failure.

        The mapping comes from primitive metadata, not task names. A primitive
        is eligible only when its declared transformation can satisfy every
        remaining input/output quantity from the live inventory.
        """
        if not blocked_skills or not completion_contract:
            return None
        progress = contract_progress(completion_contract, inventory) or {}
        targets = progress.get("targets", [])
        for primitive in capability_summary.get("primitive_reference", []):
            fallback = primitive.get("contract_fallback") or {}
            if fallback.get("kind") != "raw_to_ingot":
                continue
            prefix = fallback.get("input_prefix", "")
            suffix = fallback.get("output_suffix", "")
            fuel = fallback.get("fuel", "")
            for consumed in targets:
                input_item = str(consumed.get("item", ""))
                if consumed.get("comparison") != "<=" or not input_item.startswith(prefix):
                    continue
                output_item = input_item[len(prefix):] + suffix
                produced = next(
                    (
                        target for target in targets
                        if target.get("item") == output_item
                        and target.get("comparison", ">=") == ">="
                        and int(target.get("remaining_count", 0)) > 0
                    ),
                    None,
                )
                if not produced:
                    continue
                count = int(produced["remaining_count"])
                if int(inventory.get(input_item, 0)) < count or int(inventory.get(fuel, 0)) < 1:
                    continue
                if int(consumed.get("remaining_count", 0)) < count:
                    continue
                return {
                    "decision": "use_primitive",
                    "skill_name": "",
                    "primitive_name": primitive.get("name", ""),
                    "primitive_args": [input_item, fuel, count],
                    "reasoning": "A blocked skill made no contract progress; a declared primitive transformation can directly satisfy the remaining contract.",
                    "expected_effect": f"Consume {count} {input_item} and add at least {count} {output_item}.",
                    "fallback": "If this primitive fails, request a new validated composition.",
                    "world_asset_dependencies": [
                        requirement.get("role", "")
                        for requirement in primitive.get("world_asset_requirements", [])
                        if requirement.get("role")
                    ],
                    "persistent_asset": {"action": "none", "block_name": ""},
                    "new_skill_request": {},
                }
        return None

    def _failure_context_if_blocked(
        self,
        *,
        route: dict[str, Any],
        evaluation: dict[str, Any],
        events_before: list[tuple[str, dict[str, Any]]] | None,
        events_after: list[tuple[str, dict[str, Any]]] | None,
        completion_contract: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        skill_name = route.get("skill_name", "")
        if not skill_name or evaluation.get("success"):
            return None
        if skill_name in self.capabilities.diagnostic_skill_names():
            return None
        before = contract_progress(completion_contract, inventory_from(events_before))
        after = contract_progress(completion_contract, inventory_from(events_after))
        if not before or not after:
            return None
        made_progress = any(
            int(new.get("remaining_count", 0)) < int(old.get("remaining_count", 0))
            for old, new in zip(before.get("targets", []), after.get("targets", []))
        )
        observable_progress = self._observable_progress(route, events_before, events_after)
        critique = str(evaluation.get("critique", ""))
        execution_error = "execution error" in critique.lower() or "error" in critique.lower()
        # A capability may complete a necessary, observable state transition
        # (for example, equipping the required tool) without changing the
        # task's inventory contract yet.  That is progress, not evidence that
        # the selected skill should be blacklisted.  State evidence is more
        # reliable than a textual execution error, which can be emitted after
        # a bridge action has already taken effect.
        if made_progress or observable_progress:
            return None
        return {
            "skill_name": skill_name,
            "reason": "execution_error" if execution_error else "no_contract_progress",
            "contract_before": before,
            "contract_after": after,
            "critique": critique,
        }

    def _observable_progress(
        self,
        route: dict[str, Any],
        events_before: list[tuple[str, dict[str, Any]]] | None,
        events_after: list[tuple[str, dict[str, Any]]] | None,
    ) -> bool:
        """Recognize durable state transitions outside an inventory contract.

        Completion contracts intentionally measure consumable inventory goals.
        Equipment and placed-world assets are separate observable state, so an
        action that changes one of those must not be treated as a no-op solely
        because the final task is still incomplete.
        """
        def observation(events: list[tuple[str, dict[str, Any]]] | None) -> dict[str, Any]:
            if not events:
                return {}
            for event_type, event in reversed(events):
                if event_type == "observe" and isinstance(event, dict):
                    return event
            return {}

        def main_hand(event: dict[str, Any]) -> Any:
            status = event.get("status", {}) if isinstance(event, dict) else {}
            equipment = status.get("equipment", []) if isinstance(status, dict) else []
            return equipment[4] if isinstance(equipment, list) and len(equipment) > 4 else None

        expected_hand = ""
        if route.get("decision") == "use_primitive" and route.get("primitive_name") == "equipItem":
            args = route.get("primitive_args", [])
            if len(args) == 1 and isinstance(args[0], str):
                expected_hand = args[0]
        elif route.get("decision") == "use_skill":
            skill = self.capabilities.skill_manager.skills.get(route.get("skill_name", ""), {})
            plan = (skill.get("metadata", {}) or {}).get("primitive_plan", [])
            if isinstance(plan, list):
                for step in plan:
                    args = step.get("args", []) if isinstance(step, dict) else []
                    if (
                        isinstance(step, dict)
                        and step.get("op") == "equipItem"
                        and len(args) == 1
                        and isinstance(args[0], str)
                    ):
                        expected_hand = args[0]
                        break

        before = observation(events_before)
        after = observation(events_after)
        if expected_hand and main_hand(before) != expected_hand and main_hand(after) == expected_hand:
            return True

        # Asset observations are intentionally tied to the route's structured
        # asset intent.  Mere exploration also changes blockRecords, and must
        # not turn an unrelated failed skill into apparent progress.
        asset_name = str((route.get("persistent_asset") or {}).get("block_name") or "")
        if not asset_name:
            return False

        def asset_records(event: dict[str, Any]) -> set[str]:
            records = event.get("blockRecords", []) if isinstance(event, dict) else []
            result: set[str] = set()
            if not isinstance(records, list):
                return result
            for record in records:
                if isinstance(record, str):
                    result.add(record)
                    continue
                if not isinstance(record, dict):
                    continue
                name = record.get("name") or record.get("blockName")
                if isinstance(name, str):
                    result.add(name)
            return result

        return asset_name not in asset_records(before) and asset_name in asset_records(after)

    def _optimize_at_task_boundary(
        self,
        learning_records: list[dict[str, Any]],
    ) -> None:
        """Run the HCL learning loop once after a task terminal outcome.

        Prefer the most recent failed execution when present: it is the most
        actionable evidence for a repair. Otherwise learn from the final
        successful execution. Persistent capability synthesis is immediate
        only when the selected experiment policy enables skill updates.
        """
        if not learning_records or not any(
            (
                self.memory.memory_updates_enabled,
                self.memory.skill_updates_enabled,
                self.memory.prompt_updates_enabled,
            )
        ):
            return
        selected = learning_records[-1]
        for record in reversed(learning_records):
            execution = record.get("raw_record", {}).get("execution", {})
            if not record.get("evaluation", {}).get("success") or execution.get("error"):
                selected = record
                break
        try:
            self._optimize(
                selected["task_state"],
                selected["route"],
                selected["raw_record"],
                selected["evaluation"],
                selected["capability_summary"],
            )
        except Exception as exc:  # noqa: BLE001
            # Continual optimization is post-task learning. It must never
            # invalidate an already executed task transaction.
            self.memory.append_proposed_update(
                {
                    "optimization_error": str(exc),
                    "source_raw_time": selected.get("raw_record", {}).get("time"),
                }
            )

    def _replan_after_failure(
        self,
        *,
        original_instruction: str,
        failed_subgoal: str,
        completed_subgoals: list[str],
        failure_result: dict[str, Any],
        events: list[tuple[str, dict[str, Any]]] | None,
        completion_contract: dict[str, Any] | None,
    ) -> list[str]:
        feedback = {
            "failed_subgoal": failed_subgoal,
            "route": failure_result.get("route", {}),
            "evaluation": failure_result.get("evaluation", {}),
            "created_skill": failure_result.get("created_skill", {}),
        }
        query = f"{original_instruction}\nfailed_subgoal: {failed_subgoal}\nfeedback: {feedback}"
        abstract = self.memory.retrieve_abstract(query)
        capability_summary = self.capabilities.summarize(query)
        capability_summary["update_policy"] = dict(self.memory.update_policy)
        candidate_names = capability_summary.get("candidate_skill_names", [])
        capability_summary["skill_health"] = (
            self.health.summarize(candidate_names)
            if self.memory.skill_health_updates_enabled
            else {}
        )
        replanned = self.decomposer.decompose(
            instruction=original_instruction,
            events=events,
            abstract_memory=abstract,
            capability_summary=capability_summary,
            failure_feedback=feedback,
            completed_subgoals=completed_subgoals,
            completion_contract=completion_contract,
            contract_progress=contract_progress(completion_contract, inventory_from(events)),
        )
        completed = {item.strip().lower() for item in completed_subgoals}
        filtered = [item for item in replanned if item.strip().lower() not in completed]
        return filtered or [failed_subgoal]

    def _replan_from_world_state(
        self,
        *,
        original_instruction: str,
        events: list[tuple[str, dict[str, Any]]] | None,
        completion_contract: dict[str, Any] | None,
    ) -> list[str]:
        """Generate a fresh retry plan without trajectory/failure conditioning."""
        abstract = self.memory.retrieve_abstract(original_instruction)
        capability_summary = self.capabilities.summarize(original_instruction)
        capability_summary["update_policy"] = dict(self.memory.update_policy)
        capability_summary["skill_health"] = {}
        capability_summary["blocked_skill_names"] = []
        capability_summary["scenario_failures"] = []
        capability_summary["base_state"] = self.base_manager.public_state()
        return self.decomposer.decompose(
            instruction=original_instruction,
            events=self._planning_events(events),
            abstract_memory=abstract,
            capability_summary=capability_summary,
            completion_contract=completion_contract,
            contract_progress=contract_progress(
                completion_contract,
                inventory_from(events),
            ),
        )

    def _guard_unsatisfied_contract(
        self,
        route: dict[str, Any],
        completion_contract: dict[str, Any] | None,
        progress: dict[str, Any] | None,
        capability_summary: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Prevent a local subgoal from terminating an unfinished hard goal."""
        if not completion_contract or not progress or progress.get("satisfied"):
            return None
        remaining = [target for target in progress.get("targets", []) if target.get("remaining_count", 0) > 0]
        candidates = capability_summary.get("candidate_skills", [])
        diagnostic_names = set(self.capabilities.diagnostic_skill_names())
        for candidate in candidates:
            name = candidate.get("name")
            text = f"{name or ''} {candidate.get('description', '')}".lower()
            if not name or name in diagnostic_names:
                continue
            if any(
                target.get("item", "").lower() in text
                or any(term in text for term in target.get("item", "").lower().split("_") if len(term) > 2)
                for target in remaining
            ):
                return {
                    "decision": "use_skill",
                    "skill_name": name,
                    "reasoning": "Contract guard selected a capability because the original inventory target remains unsatisfied.",
                    "expected_effect": f"Make progress on remaining targets: {remaining}",
                    "fallback": "Re-evaluate the observation-grounded contract after execution.",
                    "new_skill_request": {},
                }
        return {
            "decision": "request_new_skill",
            "skill_name": "",
            "reasoning": "Completion contract remains unsatisfied and no matching executable capability was available.",
            "expected_effect": f"Make progress on remaining targets: {remaining}",
            "fallback": "Synthesize a bounded capability using existing primitives.",
            "new_skill_request": {
                "name": "fulfillRemainingContract",
                "description": "Fulfill the remaining observation-grounded inventory targets without violating protected inventory.",
                "requirements": [str(target.get("item", "")) for target in remaining],
            },
        }

    def _try_synthesize_missing_skill(
        self,
        task_state: dict[str, Any],
        route: dict[str, Any],
        capability_summary: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.memory.skill_updates_enabled:
            return {
                "applied": False,
                "reason": "skill updates are disabled by the memory-backend experiment policy",
            }
        try:
            proposal = self.synthesizer.propose(
                task_state=task_state,
                route=route,
                capability_summary=capability_summary,
            )
            update = self.synthesizer.render_update(proposal)
            if not update:
                return {"applied": False, "reason": proposal.get("reason", "synthesizer did not produce a primitive plan")}
            test_result = self.test_gate.validate_update(
                update,
                known_skill_names=set(capability_summary.get("all_skill_names", [])),
            )
            update["test_status"] = "passed" if test_result.get("passed") else "failed"
            if not test_result.get("passed"):
                return {"applied": False, "reason": "skill failed static test gate", "test": test_result}
            live_result = None
            if (update.get("test_spec") or {}).get("run_live"):
                live_result = self.test_gate.live_smoke_test(self.odyssey, update)
                if not live_result.get("passed"):
                    update["test_status"] = "failed_live"
                    return {"applied": False, "reason": "skill failed live smoke test", "test": test_result, "live_test": live_result}
            applied = self.capabilities.apply_skill_update(update)
            return {**applied, "test": test_result, "live_test": live_result}
        except Exception as exc:  # noqa: BLE001
            return {"applied": False, "reason": f"synthesis failed: {exc}"}

    def _optimize(
        self,
        task_state: dict[str, Any],
        route: dict[str, Any],
        raw_record: dict[str, Any],
        evaluation: dict[str, Any],
        capability_summary: dict[str, Any],
    ) -> None:
        candidate = self.optimizer.propose_update(
            task_state=task_state,
            route=route,
            raw_record=raw_record,
            evaluation=evaluation,
            abstract_memory=self.memory.abstract_memory,
            capability_summary=capability_summary,
        )
        # Enforce the experiment policy before evaluation or persistence, even
        # if an LLM ignores the output restriction in its system prompt.
        if not self.memory.memory_updates_enabled:
            candidate["abstract_memory"] = {}
        if not self.memory.skill_updates_enabled:
            candidate["capability_updates"] = []
        if not self.memory.prompt_updates_enabled:
            candidate["prompt_updates"] = {}
        decision = self.evaluator.judge_update(
            task_state=task_state,
            raw_record=raw_record,
            candidate_update=candidate,
            abstract_memory=self.memory.abstract_memory,
            capability_summary=capability_summary,
        )
        decision_safe_parts = decision.get("safe_parts")
        if isinstance(decision_safe_parts, dict):
            if not self.memory.memory_updates_enabled:
                decision_safe_parts["abstract_memory"] = {}
            if not self.memory.skill_updates_enabled:
                decision_safe_parts["capability_updates"] = []
            if not self.memory.prompt_updates_enabled:
                decision_safe_parts["prompt_updates"] = {}
        self.memory.append_proposed_update(
            {
                "candidate": candidate,
                "evaluation": decision,
                "source_raw_time": raw_record.get("time"),
            }
        )
        if not decision.get("approved"):
            return
        safe_parts = decision.get("safe_parts") or {}
        if not isinstance(safe_parts, dict):
            safe_parts = {}
        abstract_update = safe_parts.get("abstract_memory") or candidate.get("abstract_memory") or {}
        if not isinstance(abstract_update, dict):
            abstract_update = {}
        if self.memory.memory_updates_enabled:
            self.memory.merge_abstract(
                abstract_update,
                source_raw_time=raw_record.get("time"),
                context={
                    "task_kind": task_state.get("task_kind", "unknown"),
                    "skills": [route.get("skill_name", "")],
                    "conditions": task_state.get("constraints", []),
                    "outcome": "success" if evaluation.get("success") else "failure",
                },
            )
        prompt_updates = safe_parts.get("prompt_updates") or candidate.get("prompt_updates") or {}
        if self.memory.prompt_updates_enabled:
            self.optimizer.apply_prompt_updates(prompt_updates)
            self.refresh_prompt_overrides()
        capability_updates = safe_parts.get("capability_updates") or candidate.get("capability_updates") or []
        if not isinstance(capability_updates, list):
            capability_updates = []
        if not self.memory.skill_updates_enabled:
            capability_updates = []
        for update in capability_updates:
            test_result = self.test_gate.validate_update(
                update,
                known_skill_names=set(capability_summary.get("all_skill_names", [])),
            )
            if not isinstance(update, dict):
                self.memory.append_proposed_update(
                    {
                        "applied_capability_update": {
                            "applied": False,
                            "reason": "capability update must be a JSON object",
                            "test": test_result,
                        },
                        "source_raw_time": raw_record.get("time"),
                    }
                )
                continue
            update["test_status"] = "passed" if test_result.get("passed") else "failed"
            if not test_result.get("passed"):
                self.memory.append_proposed_update(
                    {
                        "applied_capability_update": {
                            "applied": False,
                            "reason": "skill failed static test gate",
                            "test": test_result,
                        },
                        "source_raw_time": raw_record.get("time"),
                    }
                )
                continue
            live_result = None
            if (update.get("test_spec") or {}).get("run_live"):
                live_result = self.test_gate.live_smoke_test(self.odyssey, update)
                if not live_result.get("passed"):
                    self.memory.append_proposed_update(
                        {
                            "applied_capability_update": {
                                "applied": False,
                                "reason": "skill failed live smoke test",
                                "test": test_result,
                                "live_test": live_result,
                            },
                            "source_raw_time": raw_record.get("time"),
                        }
                    )
                    continue
            result = self.capabilities.apply_skill_update(update)
            self.memory.append_proposed_update(
                {
                    "applied_capability_update": result,
                    "source_raw_time": raw_record.get("time"),
                    "test": test_result,
                    "live_test": live_result,
                }
            )
