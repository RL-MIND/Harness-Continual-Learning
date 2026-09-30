from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .llm import StructuredLLMProtocol
from .schemas import TaskEvidence


TASK_TYPE_NAMES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}
TASK_TYPE_IDS = {value: key for key, value in TASK_TYPE_NAMES.items()}
TASK_INTERFACE_HISTORY_MAX_STEPS = 12


class TaskInterface:
    """LLM-driven task interpretation with deterministic validity boundaries."""

    def __init__(
        self,
        llm: StructuredLLMProtocol,
        episode_progress_enabled: bool = True,
    ):
        self.llm = llm
        self.episode_progress_enabled = episode_progress_enabled

    @staticmethod
    def _string_list(value: Any, limit: int = 30) -> List[str]:
        if not isinstance(value, list):
            return []
        return [
            str(item).strip()
            for item in value[:limit]
            if isinstance(item, str) and str(item).strip()
        ]

    @classmethod
    def _normalize_progress(cls, value: Any) -> Dict[str, Any]:
        raw = value if isinstance(value, dict) else {}

        def optional_count(name: str) -> Optional[int]:
            count = raw.get(name)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                return None
            return count

        return {
            "inventory": cls._string_list(raw.get("inventory")),
            "completed_subgoals": cls._string_list(raw.get("completed_subgoals")),
            "pending_subgoals": cls._string_list(raw.get("pending_subgoals")),
            "current_subgoal": raw.get("current_subgoal", "").strip()
            if isinstance(raw.get("current_subgoal", ""), str)
            else "",
            "visited_locations": cls._string_list(raw.get("visited_locations")),
            "failed_or_unproductive_actions": cls._string_list(
                raw.get("failed_or_unproductive_actions")
            ),
            "required_count": optional_count("required_count"),
            "completed_count": optional_count("completed_count"),
            "loop_risk": raw.get("loop_risk")
            if isinstance(raw.get("loop_risk"), bool)
            else False,
            "uncertainties": cls._string_list(raw.get("uncertainties")),
        }

    @staticmethod
    def _goal_from_initial_observation(initial_observation: str) -> str:
        match = re.search(
            r"your\s+task\s+is\s+to\s*:\s*(.+?)(?:\r?\n|$)",
            initial_observation,
            flags=re.IGNORECASE,
        )
        if not match:
            return ""
        return " ".join(match.group(1).strip().split())

    @staticmethod
    def _task_type_from_goal(goal: str) -> str:
        """Infer an ALFWorld task type from goal text without dataset metadata."""
        normalized = " ".join(goal.lower().split())
        if not normalized:
            return "unknown"
        if re.search(r"\b(?:look at|examine)\b.*\b(?:light|\w*lamp)\b", normalized):
            return "look_at_obj_in_light"
        if re.search(r"\b(?:clean|wash)\b", normalized):
            return "pick_clean_then_place_in_recep"
        if re.search(r"\b(?:heat|warm)\b", normalized):
            return "pick_heat_then_place_in_recep"
        if re.search(r"\b(?:cool|chill)\b", normalized):
            return "pick_cool_then_place_in_recep"
        if re.search(
            r"\b(?:two|2)\b.*\b(?:put|place|move|bring)\b"
            r"|\b(?:put|place|move|bring)\b.*\b(?:two|2)\b",
            normalized,
        ):
            return "pick_two_obj_and_place"
        if re.search(r"\b(?:put|place|move|bring)\b", normalized):
            return "pick_and_place_simple"
        return "unknown"

    @staticmethod
    def _grounded_recent_history(
        episode_history: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Keep only recent environment-grounded transitions for state tracking."""
        return [
            {
                key: item.get(key)
                for key in (
                    "step",
                    "observation",
                    "action",
                    "next_observation",
                    "reward",
                )
                if key in item
            }
            for item in episode_history[-TASK_INTERFACE_HISTORY_MAX_STEPS:]
        ]

    def structure(
        self,
        interaction_id: str,
        observation: str,
        initial_observation: str,
        admissible_commands: List[str],
        game_file: str,
        step: int,
        max_steps: int,
        known_goal: Optional[str] = None,
        known_task_type: Optional[str] = None,
        task_type_hint: Optional[str] = None,
        episode_history: Optional[List[Dict[str, Any]]] = None,
        known_progress: Optional[Dict[str, Any]] = None,
    ) -> TaskEvidence:
        commands = sorted(set(admissible_commands))
        history = list(episode_history or [])
        progress_history = self._grounded_recent_history(history)
        continuing_episode = known_goal is not None
        if self.episode_progress_enabled:
            progress_instruction = (
                "Return progress as a JSON object with inventory, completed_subgoals, "
                "pending_subgoals, current_subgoal, visited_locations, "
                "failed_or_unproductive_actions, required_count, completed_count, "
                "loop_risk, and uncertainties. Infer progress only from grounded episode "
                "evidence; use empty lists/strings or null counts when unknown. "
            )
            progress_key = ", progress"
        else:
            progress_instruction = "Do not generate an episode progress object. "
            progress_key = ""
        if continuing_episode:
            system_prompt = (
                "You are the Task Interface state tracker of a continual-learning "
                "ALFWorld harness. The task was parsed at the beginning of this episode. "
                "Do not reinterpret, replace, or restate the established goal or task type. "
                "Update only the grounded current evidence, task-specific constraints, and "
                "episode progress using the latest observation and the chronological "
                "recent_episode_history. Evidence must distinguish observed facts from "
                "uncertainty. The supplied prior_progress summarizes older transitions. "
                "Use prior actions and their resulting observations to identify completed "
                "subgoals, invalid actions, and loops. "
                f"{progress_instruction}Keep every string and list concise. Return exactly one "
                f"JSON object with evidence, constraints{progress_key}, rationale only."
            )
            llm_payload = {
                "established_task": {
                    "goal": known_goal,
                    "task_type": known_task_type,
                },
                "current_observation": observation,
                "admissible_commands": commands,
                "step": step,
                "max_steps": max_steps,
                "prior_progress": known_progress or {},
                "omitted_history_steps": max(0, len(history) - len(progress_history)),
                "recent_episode_history": progress_history,
            }
            interface_mode = "progress_update"
        else:
            system_prompt = (
                "You are the Task Interface of a continual-learning ALFWorld harness. "
                "Parse the task once from the initial environment-and-goal description. "
                "Return JSON with task_type, goal, evidence, constraints"
                f"{progress_key}, rationale. task_type must be one of "
                f"{list(TASK_TYPE_IDS)}. Evidence must summarize only facts supported by "
                "the initial/current observation and useful history; do not claim hidden "
                "facts. Constraints must be concise and task-specific. "
                f"{progress_instruction}Use episode_history only as grounded interaction "
                "evidence. Do not use task-type-specific canned plans. The metadata hint is "
                "supporting evidence, not a substitute for interpreting the task. Return "
                "one concise JSON object only."
            )
            llm_payload = {
                "initial_observation": initial_observation,
                "current_observation": observation,
                "admissible_commands": commands,
                "step": step,
                "max_steps": max_steps,
                "task_type_metadata_hint": task_type_hint,
                "episode_history": progress_history,
            }
            interface_mode = "initial_parse"
        fallback_error = ""
        try:
            result = self.llm.complete_json(
                "task_interface",
                system_prompt,
                llm_payload,
            )
        except RuntimeError as exc:
            message = str(exc)
            if not message.startswith("task_interface did not return valid JSON:"):
                raise
            result = {}
            fallback_error = message
        proposed_goal = " ".join(str(result.get("goal", "")).split())
        extracted_goal = self._goal_from_initial_observation(initial_observation)
        proposed_type = str(result.get("task_type", ""))
        if fallback_error:
            task_type = (
                known_task_type
                or self._task_type_from_goal(extracted_goal)
                or "unknown"
            )
        else:
            task_type = (
                known_task_type
                or (proposed_type if proposed_type in TASK_TYPE_IDS else None)
                or (task_type_hint if task_type_hint in TASK_TYPE_IDS else None)
                or "unknown"
            )
        goal = (
            known_goal
            or proposed_goal
            or extracted_goal
            or "Complete the task described in the initial environment observation."
        )
        evidence = str(result.get("evidence", "")).strip()
        if not evidence:
            evidence = f"Goal: {goal}\nCurrent observation: {observation}"

        model_constraints = result.get("constraints", [])
        constraints = [
            str(item).strip()
            for item in model_constraints
            if isinstance(item, str) and str(item).strip()
        ]
        hard_constraints = [
            f"Choose exactly one of the {len(commands)} admissible commands.",
            f"Finish within {max_steps} environment steps.",
            "Never invent an action or object identifier absent from admissible_commands.",
        ]
        constraints = list(dict.fromkeys(constraints + hard_constraints))
        if self.episode_progress_enabled:
            progress_source = (
                known_progress
                if fallback_error and isinstance(known_progress, dict)
                else result.get("progress")
            )
            progress = self._normalize_progress(progress_source)
        else:
            progress = {}
        if fallback_error:
            interface_mode = f"{interface_mode}_json_fallback"
        return TaskEvidence(
            interaction_id=interaction_id,
            task_type=task_type,
            evidence=evidence,
            goal=goal,
            constraints=constraints,
            admissible_commands=commands,
            game_file=game_file,
            initial_observation=initial_observation,
            step=step,
            episode_history=history,
            progress=progress,
            interface_trace={
                "component": "llm_task_interface",
                "mode": interface_mode,
                "rationale": str(result.get("rationale", "")),
                "proposed_task_type": proposed_type,
                "fallback": bool(fallback_error),
                "fallback_error": fallback_error,
                "goal_source": (
                    "established"
                    if known_goal
                    else "model"
                    if proposed_goal
                    else "initial_observation"
                    if extracted_goal
                    else "generic_fallback"
                ),
            },
        )
