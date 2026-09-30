from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Protocol

from .json_utils import atomic_write_json


class Model(Protocol):
    def generate(self, prompt: str, *, state: dict[str, Any]) -> str:
        ...


@dataclass(frozen=True)
class Skill:
    skill_name: str
    description: str
    applies_when: tuple[str, ...]
    workflow_template: tuple[str, ...]
    tool_plan: tuple[dict[str, object], ...]
    prompt_hints: tuple[str, ...]
    memory_usage: tuple[str, ...]
    common_failures: tuple[str, ...]
    do_not_use_when: tuple[str, ...]

    def view(self) -> dict[str, object]:
        return {
            "skill_name": self.skill_name,
            "description": self.description,
            "applies_when": list(self.applies_when),
            "workflow_template": list(self.workflow_template),
            "tool_plan": [dict(item) for item in self.tool_plan],
            "prompt_hints": list(self.prompt_hints),
            "memory_usage": list(self.memory_usage),
            "common_failures": list(self.common_failures),
            "do_not_use_when": list(self.do_not_use_when),
        }


class SkillRegistry:
    """Read-only skill source.

    The default registry is deliberately empty. A directory is loaded only
    when it is explicitly configured, so repository-shipped task-specific
    templates cannot become an accidental continual-learning prior.
    """

    def __init__(self, skill_dir: str | Path | None = None) -> None:
        self.skill_dir = Path(skill_dir) if skill_dir else None
        self.skills = self._load_skills(self.skill_dir) if self.skill_dir else []

    def available_skill_views(self) -> list[dict[str, object]]:
        return [skill.view() for skill in self.skills]

    @staticmethod
    def _load_skills(skill_dir: Path) -> list[Skill]:
        if not skill_dir.exists():
            return []
        skills: list[Skill] = []
        for path in sorted(skill_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                skills.append(_skill_from_dict(data, path.stem))
        return skills


class EvolvingSkillMemory:
    """Skills induced only from accepted abstract memory.

    Unlike the optional static registry, this store starts empty and is part
    of the memory transaction. The model sees de-identified abstractions, not
    examples, answers, task names, or dataset names.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        model: Model | None = None,
        mode: str = "llm",
        max_skills: int = 6,
        max_new_tokens: int = 1536,
        min_source_tasks: int = 2,
        invalid_output_retries: int = 1,
    ) -> None:
        self.path = Path(path)
        self.model = model
        self.mode = str(mode or "llm").lower()
        self.max_skills = max(int(max_skills), 0)
        self.max_new_tokens = max(int(max_new_tokens), 1)
        self.min_source_tasks = max(int(min_source_tasks), 1)
        self.invalid_output_retries = max(int(invalid_output_retries), 0)
        self.records = self._load()

    def clear(self) -> None:
        self.records = []
        self._persist()

    def available_skill_views(self) -> list[dict[str, object]]:
        return [
            _skill_from_dict(item, f"evolved_skill_{index + 1}").view()
            for index, item in enumerate(self.records)
        ]

    def evolve(self, abstract_records: list[dict[str, Any]]) -> dict[str, object]:
        if self.max_skills <= 0:
            self.clear()
            return {"updated": True, "skill_count": 0, "reason": "max_skills_zero"}
        if self.mode not in {"llm", "model", "ds", "deepseek"} or self.model is None:
            return {"updated": False, "skill_count": len(self.records), "reason": "llm_disabled"}
        source_tasks = {
            str(item.get("internal_source", {}).get("task_name", ""))
            for item in abstract_records
            if isinstance(item.get("internal_source"), dict)
            and str(item.get("internal_source", {}).get("task_name", ""))
        }
        if len(source_tasks) < self.min_source_tasks:
            return {
                "updated": False,
                "skill_count": len(self.records),
                "reason": "insufficient_cross_task_abstract_evidence",
                "source_task_count": len(source_tasks),
                "required_source_task_count": self.min_source_tasks,
            }
        evidence = [_abstract_skill_view(item) for item in abstract_records]
        evidence = [item for item in evidence if item]
        if not evidence:
            return {"updated": False, "skill_count": len(self.records), "reason": "empty_abstract_memory"}
        prompt = _render_skill_evolution_prompt(
            current_skills=self.available_skill_views(),
            abstractions=evidence,
            max_skills=self.max_skills,
        )
        raw_outputs: list[str] = []
        accepted: list[dict[str, object]] = []
        rejected: list[str] = []
        output_reason = "invalid_skill_evolution_output"
        for attempt in range(self.invalid_output_retries + 1):
            raw_output = self.model.generate(
                prompt,
                state={
                    "memory_phase": "skill_evolution",
                    "memory_attempt": attempt + 1,
                    "internal_sample_id": "accepted_abstract_memory",
                    "max_new_tokens": self.max_new_tokens,
                },
            )
            raw_outputs.append(raw_output)
            decoded = _extract_json(raw_output)
            raw_skills = decoded.get("skills") if isinstance(decoded, dict) else None
            if not isinstance(raw_skills, list):
                rejected = ["schema:skills_must_be_list"]
                output_reason = "invalid_skill_evolution_output"
            else:
                accepted, rejected = _normalize_skill_list(raw_skills, max_skills=self.max_skills)
                output_reason = "all_generated_skills_rejected" if raw_skills and not accepted else ""
                if not raw_skills:
                    accepted = []
                    output_reason = ""
            if not output_reason:
                break
            if attempt < self.invalid_output_retries:
                prompt = _render_skill_repair_prompt(
                    rejected_output=raw_output,
                    rejection_reasons=rejected,
                    max_skills=self.max_skills,
                )
        if output_reason:
            return {
                "updated": False,
                "skill_count": len(self.records),
                "reason": output_reason,
                "rejected": rejected,
                "raw_outputs": raw_outputs,
            }
        self.records = accepted
        self._persist()
        return {
            "updated": True,
            "skill_count": len(self.records),
            "reason": "skills_rebuilt_from_abstract_memory",
            "rejected": rejected,
            "raw_outputs": raw_outputs,
        }

    def _load(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        return [dict(item) for item in data if isinstance(item, dict)]

    def _persist(self) -> None:
        atomic_write_json(self.path, self.records)


def _render_skill_evolution_prompt(
    *,
    current_skills: list[dict[str, object]],
    abstractions: list[dict[str, object]],
    max_skills: int,
) -> str:
    payload = {
        "current_general_skills": current_skills,
        "accepted_abstract_memory": abstractions,
        "max_skills": max_skills,
    }
    return "\n".join(
        [
            "You maintain the skill layer of a continual-learning harness.",
            "A skill is a further abstraction of accepted abstract memory, not a dataset template.",
            "Rebuild the complete skill list by merging, generalizing, deduplicating, and pruning the current skills.",
            "Keep only knowledge reusable across different domains and task families.",
            "A valid skill must apply unchanged to at least two substantially different domains; generalize date/person/location/number/rule extraction into evidence identification or verification.",
            "Never mention or encode task names, dataset names, sample IDs, exact questions, exact answers, labels, entity names, or benchmark-specific signatures.",
            "Do not infer a skill directly from a current input; use only accepted_abstract_memory below.",
            "Return one valid JSON object only: {\"skills\":[...]}.",
            "Each skill must contain: skill_name, description, applies_when, workflow_template, prompt_hints, memory_usage, common_failures, do_not_use_when.",
            "All fields except skill_name and description are arrays of concise strings. tool_plan must be an empty array. workflow_template must contain at least three executable steps.",
            f"Return at most {max_skills} skills. Returning an empty list is valid when the evidence is not general enough.",
            "Input:",
            json.dumps(payload, ensure_ascii=False, indent=2),
        ]
    )


def _render_skill_repair_prompt(
    *,
    rejected_output: str,
    rejection_reasons: list[str],
    max_skills: int,
) -> str:
    return "\n".join(
        [
            "The proposed continual-learning skills were rejected as task/domain specific or malformed.",
            "Rewrite the complete list as cross-domain meta-skills shared by evidence tasks, constraint tasks, and quantitative tasks.",
            "Synthesize common operations such as decomposition, support checking, consistency checking, and final verification; do not preserve one skill per domain.",
            "Forbidden content includes dates, years, persons, locations, entities, numbers, arithmetic, options, labels, binary decisions, and named rules.",
            "Do not mention any original task, dataset, sample, question, answer, or entity.",
            "Return one JSON object only with key skills. Every workflow_template needs at least three executable steps and tool_plan must be [].",
            f"Return at most {max_skills} skills; return an empty list if no cross-domain synthesis is justified.",
            "Rejection reasons:",
            json.dumps(rejection_reasons, ensure_ascii=False),
            "Rejected proposal:",
            str(rejected_output)[:8000],
        ]
    )


def _normalize_skill_list(
    raw_skills: list[object],
    *,
    max_skills: int,
) -> tuple[list[dict[str, object]], list[str]]:
    accepted: list[dict[str, object]] = []
    rejected: list[str] = []
    seen_names: set[str] = set()
    for index, item in enumerate(raw_skills):
        if not isinstance(item, dict):
            rejected.append(f"item_{index}:not_object")
            continue
        normalized, reason = _normalize_evolved_skill(item, index=index)
        if normalized is None:
            rejected.append(f"item_{index}:{reason}")
            continue
        name = str(normalized["skill_name"])
        if name in seen_names:
            rejected.append(f"item_{index}:duplicate_name")
            continue
        accepted.append(normalized)
        seen_names.add(name)
        if len(accepted) >= max_skills:
            break
    return accepted, rejected


def _abstract_skill_view(item: dict[str, Any]) -> dict[str, object]:
    view = {
        "role": str(item.get("abstract_role") or "policy"),
        "guidance": str(item.get("guidance") or ""),
        "applicable_when": str(item.get("applicable_when") or ""),
        "reusable_pattern": str(item.get("reusable_pattern") or ""),
        "failure_mode": str(item.get("failure_mode") or ""),
    }
    return view if view["guidance"] or view["failure_mode"] else {}


def _normalize_evolved_skill(
    data: dict[str, Any],
    *,
    index: int,
) -> tuple[dict[str, object] | None, str]:
    combined = json.dumps(data, ensure_ascii=False).lower()
    forbidden = (
        "musique",
        "proofwriter",
        "gsm8k",
        "hotpotqa",
        "strategyqa",
        "dataset",
        "benchmark",
        "task_id",
        "sample_id",
    )
    if any(token in combined for token in forbidden):
        return None, "task_or_dataset_identifier"
    content_tokens = set(re.findall(r"[a-z0-9]+", combined))
    domain_specific_tokens = {
        "arithmetic",
        "binary",
        "choice",
        "date",
        "dates",
        "entity",
        "entities",
        "label",
        "location",
        "locations",
        "number",
        "numbers",
        "numeric",
        "numerical",
        "option",
        "person",
        "persons",
        "rule",
        "rules",
        "year",
        "years",
    }
    if content_tokens & domain_specific_tokens:
        return None, "skill_content_is_task_or_domain_specific"
    description = str(data.get("description") or "").strip()
    workflow = _string_tuple(data.get("workflow_template"))
    applies_when = _string_tuple(data.get("applies_when"))
    if not description or not applies_when:
        return None, "missing_general_skill_fields"
    raw_name = str(data.get("skill_name") or f"evolved_skill_{index + 1}").strip().lower()
    skill_name = re.sub(r"[^a-z0-9]+", "_", raw_name).strip("_") or f"evolved_skill_{index + 1}"
    narrow_name_tokens = {
        "date",
        "person",
        "location",
        "entity",
        "arithmetic",
        "numeric",
        "binary",
        "multiple",
        "choice",
        "rule",
        "multihop",
        "qa",
    }
    if narrow_name_tokens & set(skill_name.split("_")):
        return None, "skill_name_is_task_or_domain_specific"
    if not workflow:
        workflow = (
            "Identify the applicable evidence and constraints.",
            description,
            "Verify the result against the known failure conditions.",
        )
    return {
        "skill_name": skill_name[:80],
        "description": description[:600],
        "applies_when": list(applies_when[:8]),
        "workflow_template": list(workflow[:12]),
        "tool_plan": [],
        "prompt_hints": list(_string_tuple(data.get("prompt_hints"))[:8]),
        "memory_usage": list(_string_tuple(data.get("memory_usage"))[:8]),
        "common_failures": list(_string_tuple(data.get("common_failures"))[:8]),
        "do_not_use_when": list(_string_tuple(data.get("do_not_use_when"))[:8]),
    }, ""


def _skill_from_dict(data: dict[str, Any], fallback_name: str) -> Skill:
    return Skill(
        skill_name=str(data.get("skill_name") or fallback_name),
        description=str(data.get("description") or ""),
        applies_when=_string_tuple(data.get("applies_when")),
        workflow_template=_string_tuple(data.get("workflow_template")),
        tool_plan=tuple(dict(item) for item in data.get("tool_plan", []) if isinstance(item, dict)),
        prompt_hints=_string_tuple(data.get("prompt_hints")),
        memory_usage=_string_tuple(data.get("memory_usage")),
        common_failures=_string_tuple(data.get("common_failures")),
        do_not_use_when=_string_tuple(data.get("do_not_use_when")),
    )


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = str(text).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start < 0:
            return {}
        try:
            value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


__all__ = ["EvolvingSkillMemory", "Skill", "SkillRegistry"]
