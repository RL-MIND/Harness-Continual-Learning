from __future__ import annotations

import json
import importlib
import tempfile
import unittest
from pathlib import Path

class _FakeModel:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.states: list[dict[str, object]] = []

    def generate(self, prompt: str, *, state: dict[str, object]) -> str:
        self.states.append(dict(state))
        if not self.outputs:
            raise AssertionError("unexpected model call")
        return self.outputs.pop(0)


def _chunk() -> dict[str, object]:
    question = (
        "Mercury completes one revolution in 88 Earth days. "
        "How many Earth years is that? Return a decimal number."
    )
    return {
        "task_id": "smoke/mercury",
        "hcl_interface": {
            "evidence": {
                "summary": "Mercury takes 88 Earth days per revolution.",
                "items": [{"modality": "text", "content": question, "source": "question"}],
            },
            "recognizable_goal": {"content": "Convert 88 days to Earth years."},
            "recognizable_constraints": [],
        },
        "task_context": {"files": [], "capability_inputs": {}},
    }


class RouterManagedArithmeticArgumentsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        global Router, RouterConfig, CapabilityExecutionContext
        router_module = importlib.import_module("hcl.router")
        capability_module = importlib.import_module("hcl.capability")
        Router = router_module.Router
        RouterConfig = router_module.RouterConfig
        CapabilityExecutionContext = capability_module.CapabilityExecutionContext

    def test_builds_executes_and_caches_safe_arithmetic_arguments(self) -> None:
        model = _FakeModel([
            json.dumps({
                "tool_name": "arithmetic_calculator",
                "arguments": {"expression": "88 / 365.25", "precision": 6},
            })
        ])
        with tempfile.TemporaryDirectory() as temp_dir:
            router = Router(
                model,
                RouterConfig(
                    capability_enabled=True,
                    capability_execution_mode="router_managed",
                    tool_argument_cache_path=str(Path(temp_dir) / "arguments.jsonl"),
                ),
            )
            workflow = {
                "workflow_name": "conversion",
                "steps": ["Convert 88 days to years."],
                "use_memory": False,
                "use_tools": True,
                "memory_query": "",
                "tool_needs": ["arithmetic_calculator"],
                "rationale": "Compute 88 / 365.25.",
            }
            selected = [{"tool_name": "arithmetic_calculator"}]
            context = CapabilityExecutionContext(task_id="smoke/mercury")

            first = router.execute_selected_capabilities(
                _chunk(), workflow, selected, execution_context=context
            )
            second = router.execute_selected_capabilities(
                _chunk(), workflow, selected, execution_context=context
            )

        self.assertEqual(first[0]["status"], "success")
        self.assertEqual(first[0]["arguments"]["expression"], "88 / 365.25")
        self.assertEqual(first[0]["output"]["result"], "0.240931")
        self.assertEqual(second, first)
        self.assertEqual(len(model.states), 1)
        self.assertEqual(model.states[0]["router_phase"], "build_tool_arguments")

    def test_rejects_unsafe_expression(self) -> None:
        model = _FakeModel([
            json.dumps({
                "tool_name": "arithmetic_calculator",
                "arguments": {"expression": "__import__('os').system('id')", "precision": 2},
            })
        ])
        router = Router(
            model,
            RouterConfig(
                capability_enabled=True,
                capability_execution_mode="router_managed",
                tool_argument_invalid_output_retries=0,
            ),
        )
        result = router.execute_selected_capabilities(
            _chunk(),
            {
                "workflow_name": "unsafe",
                "steps": [],
                "use_memory": False,
                "use_tools": True,
                "memory_query": "",
                "tool_needs": ["arithmetic_calculator"],
                "rationale": "",
            },
            [{"tool_name": "arithmetic_calculator"}],
            execution_context=CapabilityExecutionContext(task_id="smoke/unsafe"),
        )
        self.assertEqual(result[0]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
