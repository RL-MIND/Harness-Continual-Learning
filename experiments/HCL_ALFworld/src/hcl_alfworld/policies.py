from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .llm import StructuredLLMProtocol, make_llm
from .schemas import RoutingContext


ACTION_DIALOGUE_HISTORY_MAX_STEPS = 15


class Policy(ABC):
    @abstractmethod
    def select_action(self, context: RoutingContext, info: Dict[str, Any]) -> str:
        raise NotImplementedError


class OpenAICompatiblePolicy(Policy):
    """LLM action executor grounded by the routed harness context."""

    def __init__(self, llm: StructuredLLMProtocol, action_retries: int = 2):
        self.llm = llm
        self.action_retries = action_retries
        self.last_trace: Dict[str, Any] = {}

    @staticmethod
    def _episode_dialogue(context: RoutingContext) -> list[Dict[str, str]]:
        """Render the recent grounded transitions as a coherent ReAct conversation."""
        task = context.task
        opening = task.initial_observation.strip() or (
            f"Task goal: {task.goal}\nInitial evidence: {task.evidence}"
        )
        history = task.episode_history[-ACTION_DIALOGUE_HISTORY_MAX_STEPS:]
        omitted_steps = len(task.episode_history) - len(history)
        if omitted_steps:
            opening += (
                f"\n\nContext note: {omitted_steps} earlier transitions are omitted from "
                "this action prompt. current_step.progress preserves the accumulated task "
                "state; the dialogue below contains the most recent transitions."
            )
        dialogue = [
            {
                "role": "user",
                "content": f"Initial environment and task:\n{opening}",
            }
        ]
        for item in history:
            assessment = str(item.get("state_assessment", "")).strip()
            plan = str(item.get("next_step_plan", "")).strip()
            rationale = str(item.get("action_rationale", "")).strip()
            assistant_parts = []
            if assessment:
                assistant_parts.append(f"State assessment: {assessment}")
            if plan:
                assistant_parts.append(f"Next-step plan: {plan}")
            elif rationale:
                assistant_parts.append(f"Decision rationale: {rationale}")
            assistant_parts.append(f"Action: {item.get('action', '')}")
            dialogue.append(
                {"role": "assistant", "content": "\n".join(assistant_parts)}
            )
            feedback = f"Observation: {item.get('next_observation', '')}"
            if item.get("reward") is not None:
                feedback += f"\nEnvironment reward: {item.get('reward')}"
            dialogue.append({"role": "user", "content": feedback})
        return dialogue

    def select_action(self, context: RoutingContext, info: Dict[str, Any]) -> str:
        del info
        payload: Dict[str, Any] = {
            "task": {
                "task_type": context.task.task_type,
                "goal": context.task.goal,
                "constraints": context.task.constraints,
            },
            "episode_dialogue": self._episode_dialogue(context),
            "current_step": {
                "step": context.task.step,
                "observation": context.task.evidence,
                "progress": context.task.progress,
            },
            "indexed_admissible_commands": [
                {"index": index, "action": action}
                for index, action in enumerate(context.task.admissible_commands)
            ],
            "strategies": [
                {
                    "title": item.title,
                    "procedure": item.procedure,
                    "applicability": item.applicability,
                    "failure_boundaries": item.failure_boundaries,
                    "validation_count": item.validation_count,
                }
                for item in context.strategies
            ],
            "memories": [
                {
                    "goal": item.get("goal"),
                    "success": item.get("success"),
                    "failure_reason": item.get("failure_reason"),
                    "steps": [
                        {
                            "observation": step.get("observation"),
                            "action": step.get("action"),
                            "next_observation": step.get("next_observation"),
                        }
                        for step in item.get("steps", [])
                    ],
                    **(
                        {
                            "memory_id": item.get("memory_id"),
                            "memory_type": item.get("memory_type"),
                            "similarity": item.get("similarity"),
                        }
                        if item.get("memory_type")
                        else {}
                    ),
                    **(
                        {"script": item.get("script")}
                        if item.get("script")
                        else {}
                    ),
                    **(
                        {"reflection": item.get("reflection")}
                        if item.get("reflection")
                        else {}
                    ),
                    **(
                        {"confidence": item.get("confidence")}
                        if "confidence" in item
                        else {}
                    ),
                }
                for item in context.memories
            ],
            "capabilities": [
                {
                    "name": item.name,
                    "kind": item.kind,
                    "description": item.description,
                    "function": item.function,
                    "dependencies": item.dependencies,
                    "procedure": item.procedure,
                    "preconditions": item.preconditions,
                    "success_conditions": item.success_conditions,
                    "failure_modes": item.failure_modes,
                    "status": item.status,
                    "confidence": item.confidence,
                    "provider_component": item.provider_component,
                    "validation_evidence": item.validation_evidence[-5:],
                }
                for item in context.capabilities
            ],
            "router_rationale": context.trace.get("rationale", ""),
        }
        attempts = []
        for attempt in range(self.action_retries + 1):
            try:
                result = self.llm.complete_json(
                    "action_policy",
                    (
                        "You control an ALFWorld text agent. Reason about task progress, current "
                        "evidence, the structured episode progress, the chronological "
                        "recent episode_dialogue, retrieved experience, failure "
                        "boundaries and capability preconditions. Use prior steps to avoid loops and "
                        "do not repeat a failed action unless the state has materially changed. "
                        "First give a concise grounded state_assessment, then a concise "
                        "next_step_plan, and finally choose the single action that executes that "
                        "plan. Do not invent hidden state or provide a long chain of thought. "
                        "Choose exactly one entry from indexed_admissible_commands. Return its "
                        "integer index, never generate or copy an action string. Memories are "
                        "advisory and may be stale. Return JSON with state_assessment, "
                        "next_step_plan, action_index, and rationale."
                    ),
                    payload,
                )
            except RuntimeError as exc:
                message = str(exc)
                if not message.startswith("action_policy did not return valid JSON:"):
                    raise
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "proposed_action_index": None,
                        "valid": False,
                        "error": message,
                    }
                )
                # complete_json already exhausted its own structured retries. Do not
                # abort a parallel evaluation batch over one malformed model response.
                break
            proposed_index = result.get("action_index")
            valid_index = (
                isinstance(proposed_index, int)
                and not isinstance(proposed_index, bool)
                and 0 <= proposed_index < len(context.task.admissible_commands)
            )
            attempt_trace = {
                "attempt": attempt + 1,
                "proposed_action_index": proposed_index,
                "valid": valid_index,
                "state_assessment": str(result.get("state_assessment", "")),
                "next_step_plan": str(result.get("next_step_plan", "")),
                "rationale": str(result.get("rationale", "")),
            }
            attempts.append(attempt_trace)
            if valid_index:
                action = context.task.admissible_commands[proposed_index]
                self.last_trace = {
                    "component": "llm_action_policy",
                    "action": action,
                    "action_index": proposed_index,
                    "state_assessment": str(result.get("state_assessment", "")),
                    "next_step_plan": str(result.get("next_step_plan", "")),
                    "rationale": str(result.get("rationale", "")),
                    "attempts": attempts,
                }
                return action
            payload["invalid_previous_action_index"] = proposed_index
            payload["valid_action_index_range"] = [
                0,
                max(0, len(context.task.admissible_commands) - 1),
            ]
            payload["correction"] = (
                "Return one integer action_index from the supplied valid range."
            )
        if "look" in context.task.admissible_commands:
            self.last_trace = {
                "component": "hard_action_fallback",
                "action": "look",
                "attempts": attempts,
            }
            return "look"
        fallback = context.task.admissible_commands[0] if context.task.admissible_commands else "look"
        self.last_trace = {
            "component": "hard_action_fallback",
            "action": fallback,
            "attempts": attempts,
        }
        return fallback


def make_policy(
    name: str,
    config: Dict[str, Any],
    seed: int,
    llm: Optional[StructuredLLMProtocol] = None,
) -> Policy:
    del seed
    if name.lower() != "openai":
        raise ValueError(f"Unknown policy: {name}. This project supports only policy=openai.")
    shared_llm = llm or make_llm(config)
    return OpenAICompatiblePolicy(
        shared_llm,
        action_retries=int(config.get("action_retries", config.get("max_retries", 2))),
    )
