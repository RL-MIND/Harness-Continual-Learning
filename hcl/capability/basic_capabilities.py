from __future__ import annotations

import ast
import json
import operator
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .external_models import ExternalModelServiceClient


CAPABILITY_DIR = Path(__file__).resolve().parent
DEFAULT_JS_TOOL_DIR = CAPABILITY_DIR / "js_tools"
JS_TOOL_RUNNER = CAPABILITY_DIR / "js_tool_runner.js"


class CapabilityExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CapabilityExecutionContext:
    task_id: str = ""
    attachments: tuple[dict[str, Any], ...] = ()
    text_candidates: tuple[dict[str, str], ...] = ()
    allowed_corpora: tuple[str, ...] = ()

    def attachment(self, attachment_id: str) -> dict[str, Any] | None:
        for index, item in enumerate(self.attachments):
            item_id = str(item.get("attachment_id") or item.get("id") or f"attachment_{index}")
            if item_id == attachment_id:
                return dict(item)
        return None


@dataclass(frozen=True)
class CapabilityTool:
    name: str
    description: str
    parameters: dict[str, Any]
    executor: Callable[[dict[str, Any]], dict[str, Any]]
    source: str = "python"
    tool_path: str | None = None
    tags: tuple[str, ...] = ()
    kind: str = "local_function"
    requirements: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def chat_tool_schema(self, *, strict: bool = False) -> dict[str, Any]:
        function: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
        if strict:
            function["strict"] = True
        return {"type": "function", "function": function}

class BasicCapabilityRegistry:
    """Local HCL capabilities with chat-completions function schemas."""

    def __init__(
        self,
        *,
        js_tool_dir: str | Path | None = None,
        strict_tools: bool = False,
        js_tool_timeout_seconds: float = 8.0,
        external_capabilities_enabled: bool = False,
        external_service_url: str | None = None,
        external_timeout_seconds: float = 15.0,
        semantic_search_enabled: bool = True,
        cross_modal_match_enabled: bool = True,
    ) -> None:
        self.js_tool_dir = Path(js_tool_dir) if js_tool_dir else DEFAULT_JS_TOOL_DIR
        self.strict_tools = strict_tools
        self.js_tool_timeout_seconds = float(js_tool_timeout_seconds)
        self.node_path = shutil.which("node")
        self.tools: dict[str, CapabilityTool] = {
            "arithmetic_calculator": CapabilityTool(
                name="arithmetic_calculator",
                description=(
                    "Evaluate a numeric arithmetic expression containing only numbers, "
                    "parentheses, and basic arithmetic operators."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Arithmetic expression, e.g. '(20 / 2) * 20'.",
                        },
                        "precision": {
                            "type": "integer",
                            "description": "Decimal places for non-integer results.",
                            "minimum": 0,
                            "maximum": 12,
                        },
                    },
                    "required": ["expression", "precision"],
                    "additionalProperties": False,
                },
                executor=self._arithmetic_calculator,
                tags=("arithmetic", "math", "numeric"),
            ),
        }
        self.tools.update(self._load_js_tools(self.js_tool_dir))
        if external_capabilities_enabled:
            self.tools.update(
                self._external_model_tools(
                    service_url=str(external_service_url or ""),
                    timeout_seconds=external_timeout_seconds,
                    semantic_search_enabled=semantic_search_enabled,
                    cross_modal_match_enabled=cross_modal_match_enabled,
                )
            )

    def chat_tools(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        selected = self._selected_tools(names)
        return [tool.chat_tool_schema(strict=self.strict_tools) for tool in selected]

    def available_tool_views(self) -> list[dict[str, Any]]:
        return self.tool_views(None)

    def tool_views(self, names: list[str] | None) -> list[dict[str, Any]]:
        views: list[dict[str, Any]] = []
        for tool in self._selected_tools(names):
            views.append(
                {
                    "tool_name": tool.name,
                    "description": tool.description,
                    "source": tool.source,
                    "tool_path": tool.tool_path,
                    "tags": list(tool.tags),
                    "kind": tool.kind,
                    "requirements": list(tool.requirements),
                    "limitations": list(tool.limitations),
                    "chat_tool": tool.chat_tool_schema(strict=self.strict_tools),
                    "status": "available",
                }
            )
        return views

    def capability_summaries(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "kind": tool.kind,
                "description": tool.description,
                "requirements": list(tool.requirements),
                "limitations": list(tool.limitations),
            }
            for tool in self.tools.values()
        ]

    def eligible_tool_views(
        self,
        model_visible: dict[str, Any],
        execution_context: CapabilityExecutionContext,
    ) -> list[dict[str, Any]]:
        del model_visible
        names: list[str] = []
        for tool in self.tools.values():
            has_image = any(
                str(item.get("type", "")).lower() == "image"
                for item in execution_context.attachments
            )
            if "image_attachment" in tool.requirements and not has_image:
                continue
            if "text_candidates" in tool.requirements and not execution_context.text_candidates:
                continue
            names.append(tool.name)
        return self.tool_views(names)

    def select_tool_names(
        self,
        model_visible: dict[str, Any],
        workflow_decision: dict[str, object],
    ) -> list[str]:
        text = _visible_text(model_visible).lower()
        workflow_text = json.dumps(workflow_decision, ensure_ascii=False).lower()
        requested = [str(item).lower() for item in workflow_decision.get("tool_needs", []) or []]
        joined_request = " ".join(requested)
        selected: list[str] = []
        for item in requested:
            selected.extend(self._matching_requested_tools(item))

        if any(
            marker in f"{text} {workflow_text} {joined_request}"
            for marker in (
                "numeric",
                "number",
                "arithmetic",
                "calculate",
                "math",
                "total",
                "how many",
                "how much",
            )
        ):
            selected.append(self._preferred_tool_name("js_arithmetic_calculator", "arithmetic_calculator"))
        return _dedupe(selected)

    def execute(
        self,
        name: str,
        arguments: dict[str, Any] | str | None,
        *,
        execution_context: CapabilityExecutionContext | None = None,
    ) -> dict[str, Any]:
        tool = self.tools.get(name)
        if tool is None:
            raise CapabilityExecutionError(f"Unknown capability tool: {name}")
        if arguments is None:
            parsed: dict[str, Any] = {}
        elif isinstance(arguments, str):
            try:
                value = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                raise CapabilityExecutionError(f"Invalid JSON arguments for {name}: {exc}") from exc
            if not isinstance(value, dict):
                raise CapabilityExecutionError(f"Arguments for {name} must decode to an object.")
            parsed = value
        elif isinstance(arguments, dict):
            parsed = dict(arguments)
        else:
            raise CapabilityExecutionError(f"Arguments for {name} must be a JSON object.")
        _validate_arguments(name, parsed, tool.parameters)
        if name == "semantic_search":
            parsed = self._authorize_semantic_search(parsed, execution_context)
        elif name == "cross_modal_match":
            parsed = self._resolve_cross_modal_attachment(parsed, execution_context)
        return tool.executor(parsed)

    def _authorize_semantic_search(
        self,
        arguments: dict[str, Any],
        execution_context: CapabilityExecutionContext | None,
    ) -> dict[str, Any]:
        if execution_context is None:
            raise CapabilityExecutionError("semantic_search requires a capability execution context")
        corpus_id = str(arguments.get("corpus_id", ""))
        allowed = tuple(execution_context.allowed_corpora)
        if allowed and corpus_id not in allowed:
            raise CapabilityExecutionError(f"Corpus is not allowed for this task: {corpus_id}")
        return dict(arguments)

    def _resolve_cross_modal_attachment(
        self,
        arguments: dict[str, Any],
        execution_context: CapabilityExecutionContext | None,
    ) -> dict[str, Any]:
        if execution_context is None:
            raise CapabilityExecutionError("cross_modal_match requires a capability execution context")
        attachment_id = str(arguments.get("attachment_id", ""))
        attachment = execution_context.attachment(attachment_id)
        if attachment is None:
            raise CapabilityExecutionError(f"Unknown attachment for current task: {attachment_id}")
        resolved = dict(arguments)
        resolved["attachment"] = attachment
        return resolved

    def _selected_tools(self, names: list[str] | None) -> list[CapabilityTool]:
        if names is None:
            return list(self.tools.values())
        return [self.tools[name] for name in names if name in self.tools]

    def _preferred_tool_name(self, *names: str) -> str:
        for name in names:
            if name in self.tools:
                return name
        return names[-1]

    def _matching_requested_tools(self, request: str) -> list[str]:
        request = request.strip().lower()
        if not request:
            return []
        for name in self.tools:
            if request == name.lower():
                return [name]
        if request in {"arithmetic", "math", "numeric", "calculate", "arithmetic_reasoning"}:
            return [self._preferred_tool_name("js_arithmetic_calculator", "arithmetic_calculator")]
        if request in {
            "retrieval", "semantic retrieval", "semantic search", "text retrieval", "knowledge search"
        } and "semantic_search" in self.tools:
            return ["semantic_search"]
        if request in {
            "multimodal", "cross modal", "cross-modal matching", "image text matching", "image matching"
        } and "cross_modal_match" in self.tools:
            return ["cross_modal_match"]
        return []

    def _external_model_tools(
        self,
        *,
        service_url: str,
        timeout_seconds: float,
        semantic_search_enabled: bool,
        cross_modal_match_enabled: bool,
    ) -> dict[str, CapabilityTool]:
        client = ExternalModelServiceClient(service_url, timeout_seconds)
        tools: dict[str, CapabilityTool] = {}
        if semantic_search_enabled:
            tools["semantic_search"] = CapabilityTool(
                name="semantic_search",
                description=(
                    "Retrieve semantically or structurally similar answer-free training examples "
                    "and passages from the configured reasoning corpus. Useful for difficult "
                    "multi-hop or uncertain cases; it does not search the current prompt or internet."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1, "maxLength": 8000},
                        "corpus_id": {"type": "string", "minLength": 1},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                    "required": ["query", "corpus_id", "top_k"],
                    "additionalProperties": False,
                },
                executor=client.semantic_search,
                source="external_model_service",
                tags=("retrieval", "semantic", "text", "knowledge", "bge-m3"),
                kind="text_semantic_retrieval",
                limitations=("Searches configured indexes only; it does not access the internet.",),
            )
        if cross_modal_match_enabled:
            tools["cross_modal_match"] = CapabilityTool(
                name="cross_modal_match",
                description="Score or rank text candidates against an image attached to the current task.",
                parameters={
                    "type": "object",
                    "properties": {
                        "attachment_id": {"type": "string", "minLength": 1},
                        "candidates": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 100,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string", "minLength": 1},
                                    "text": {"type": "string", "minLength": 1, "maxLength": 4000},
                                },
                                "required": ["id", "text"],
                                "additionalProperties": False,
                            },
                        },
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                    "required": ["attachment_id", "candidates", "top_k"],
                    "additionalProperties": False,
                },
                executor=client.cross_modal_match,
                source="external_model_service",
                tags=("multimodal", "image", "text", "matching", "siglip2"),
                kind="image_text_matching",
                requirements=("image_attachment", "text_candidates"),
                limitations=("Matching only; not OCR, caption generation, or visual question answering.",),
            )
        return tools

    def _load_js_tools(self, js_tool_dir: Path) -> dict[str, CapabilityTool]:
        if self.node_path is None or not JS_TOOL_RUNNER.exists() or not js_tool_dir.exists():
            return {}
        tools: dict[str, CapabilityTool] = {}
        for path in sorted(js_tool_dir.glob("*.js")):
            if path.name == JS_TOOL_RUNNER.name:
                continue
            try:
                metadata = self._read_js_tool_metadata(path)
            except CapabilityExecutionError:
                continue
            name = str(metadata.get("name") or path.stem)
            description = str(metadata.get("description") or f"JavaScript capability tool {name}.")
            parameters = metadata.get("parameters")
            if not isinstance(parameters, dict):
                continue
            tags = tuple(str(item) for item in metadata.get("tags", []) if str(item).strip())
            tools[name] = CapabilityTool(
                name=name,
                description=description,
                parameters=parameters,
                executor=self._make_js_executor(path),
                source="javascript",
                tool_path=str(path),
                tags=tags,
            )
        return tools

    def _read_js_tool_metadata(self, path: Path) -> dict[str, Any]:
        output = self._run_js_tool("metadata", path, {})
        metadata = output.get("metadata")
        if not isinstance(metadata, dict):
            raise CapabilityExecutionError(f"JS tool metadata must be an object: {path}")
        return metadata

    def _make_js_executor(self, path: Path) -> Callable[[dict[str, Any]], dict[str, Any]]:
        def execute(arguments: dict[str, Any]) -> dict[str, Any]:
            output = self._run_js_tool("execute", path, arguments)
            result = output.get("result")
            if isinstance(result, dict):
                return result
            return {"result": result}

        return execute

    def _run_js_tool(self, mode: str, path: Path, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.node_path is None:
            raise CapabilityExecutionError("Node.js is not available for JavaScript capability tools.")
        if not JS_TOOL_RUNNER.exists():
            raise CapabilityExecutionError(f"JavaScript tool runner is missing: {JS_TOOL_RUNNER}")
        try:
            completed = subprocess.run(
                [
                    self.node_path,
                    str(JS_TOOL_RUNNER),
                    mode,
                    str(path),
                    json.dumps(arguments, ensure_ascii=False),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.js_tool_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise CapabilityExecutionError(f"JavaScript tool timed out: {path}") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise CapabilityExecutionError(f"JavaScript tool failed: {path}: {detail[:1000]}")
        try:
            decoded = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise CapabilityExecutionError(f"JavaScript tool returned invalid JSON: {path}") from exc
        if not isinstance(decoded, dict):
            raise CapabilityExecutionError(f"JavaScript tool returned non-object JSON: {path}")
        return decoded

    def _arithmetic_calculator(self, arguments: dict[str, Any]) -> dict[str, Any]:
        expression = str(arguments.get("expression", ""))
        precision = int(arguments.get("precision", 6))
        if not expression.strip():
            raise CapabilityExecutionError("expression is required")
        result = _safe_eval_arithmetic(expression)
        if isinstance(result, float) and result.is_integer():
            rendered = str(int(result))
        elif isinstance(result, float):
            rendered = f"{result:.{max(min(precision, 12), 0)}f}".rstrip("0").rstrip(".")
        else:
            rendered = str(result)
        return {"expression": expression, "result": rendered, "numeric_result": result}

def _visible_text(model_visible: dict[str, Any]) -> str:
    chunks: list[str] = []
    evidence = model_visible.get("evidence", {})
    if isinstance(evidence, dict):
        chunks.append(str(evidence.get("summary", "")))
        for item in evidence.get("items", []) or []:
            if isinstance(item, dict):
                chunks.append(str(item.get("content", "")))
    goal = model_visible.get("recognizable_goal", {})
    if isinstance(goal, dict):
        chunks.append(str(goal.get("content", "")))
    for item in model_visible.get("recognizable_constraints", []) or []:
        if isinstance(item, dict):
            chunks.append(str(item.get("content", "")))
    return "\n".join(chunks)


def _validate_arguments(name: str, arguments: dict[str, Any], schema: dict[str, Any]) -> None:
    """Validate the JSON-Schema subset used by built-in HCL capabilities."""
    required = schema.get("required", [])
    for key in required if isinstance(required, list) else []:
        if key not in arguments:
            raise CapabilityExecutionError(f"Missing required argument for {name}: {key}")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return
    if schema.get("additionalProperties") is False:
        unexpected = sorted(set(arguments) - set(properties))
        if unexpected:
            raise CapabilityExecutionError(f"Unexpected argument for {name}: {unexpected[0]}")
    for key, value in arguments.items():
        item_schema = properties.get(key)
        if isinstance(item_schema, dict):
            _validate_value(name, key, value, item_schema)


def _validate_value(name: str, key: str, value: Any, schema: dict[str, Any]) -> None:
    expected = schema.get("type")
    type_ok = {
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
    }.get(str(expected), True)
    if not type_ok:
        raise CapabilityExecutionError(f"Argument {key} for {name} must be {expected}.")
    if isinstance(value, str):
        if len(value) < int(schema.get("minLength", 0)):
            raise CapabilityExecutionError(f"Argument {key} for {name} is too short.")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            raise CapabilityExecutionError(f"Argument {key} for {name} is too long.")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise CapabilityExecutionError(f"Argument {key} for {name} is below minimum.")
        if "maximum" in schema and value > schema["maximum"]:
            raise CapabilityExecutionError(f"Argument {key} for {name} is above maximum.")
    if isinstance(value, list):
        if len(value) < int(schema.get("minItems", 0)):
            raise CapabilityExecutionError(f"Argument {key} for {name} has too few items.")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise CapabilityExecutionError(f"Argument {key} for {name} has too many items.")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                nested_key = f"{key}[{index}]"
                _validate_value(name, nested_key, item, item_schema)
                if isinstance(item, dict):
                    nested_schema = dict(item_schema)
                    _validate_arguments(name, item, nested_schema)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


_ARITHMETIC_BINOPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ARITHMETIC_UNARYOPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval_arithmetic(expression: str) -> float | int:
    tree = ast.parse(expression, mode="eval")
    return _eval_arithmetic_node(tree.body)


def _eval_arithmetic_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        op = _ARITHMETIC_BINOPS.get(type(node.op))
        if op is None:
            raise CapabilityExecutionError(f"Unsupported arithmetic operator: {type(node.op).__name__}")
        left = _eval_arithmetic_node(node.left)
        right = _eval_arithmetic_node(node.right)
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _ARITHMETIC_UNARYOPS.get(type(node.op))
        if op is None:
            raise CapabilityExecutionError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op(_eval_arithmetic_node(node.operand))
    raise CapabilityExecutionError(f"Unsupported arithmetic expression node: {type(node).__name__}")
