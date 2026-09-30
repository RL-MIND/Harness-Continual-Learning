from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PromptTemplate:
    template_id: str
    sections: list[str]
    answer_instructions: dict[str, str]
    description: str = ""

    @classmethod
    def from_path(cls, path: str | Path) -> "PromptTemplate":
        data = _read_json(path)
        return cls.from_dict(data, template_id_fallback=Path(path).stem)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, template_id_fallback: str = "inline_template") -> "PromptTemplate":
        sections = [str(section) for section in data.get("sections", []) if str(section).strip()]
        if not sections:
            raise ValueError(f"Prompt template has no sections: {template_id_fallback}")
        answer_instructions = {
            str(key): str(value)
            for key, value in dict(data.get("answer_instructions", {})).items()
        }
        return cls(
            template_id=str(data.get("template_id") or template_id_fallback),
            description=str(data.get("description", "")),
            sections=sections,
            answer_instructions=answer_instructions,
        )

    def render(self, values: dict[str, object]) -> str:
        """Render named HCL placeholders while leaving JSON braces literal.

        Prompt artifacts frequently describe JSON objects. ``str.format`` treats
        every brace in such examples as formatting syntax, which made otherwise
        valid optimizer candidates unrenderable.  HCL templates only need simple
        ``{placeholder_name}`` substitutions, so keep that contract explicit.
        """
        parts: list[str] = []
        for section in self.sections:
            rendered = _render_section(section, values).strip()
            if rendered:
                parts.append(rendered)
        return "\n".join(parts)

_PLACEHOLDER = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})")


def _render_section(section: str, values: dict[str, object]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise KeyError(name)
        return str(values[name])

    # Protect established literal-brace escapes before substituting. This also
    # ensures braces contained in an injected JSON value are never rewritten.
    left_literal = "\x00HCL_LEFT_BRACE\x00"
    right_literal = "\x00HCL_RIGHT_BRACE\x00"
    protected = section.replace("{{", left_literal).replace("}}", right_literal)
    rendered = _PLACEHOLDER.sub(replace, protected)
    return rendered.replace(left_literal, "{").replace(right_literal, "}")


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Prompt template must be a JSON object: {path}")
    return data
