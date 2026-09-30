from __future__ import annotations

import argparse
import importlib
import json
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8007")
    parser.add_argument("--model", default="local-model")
    args = parser.parse_args()

    package = "hcl"
    models = importlib.import_module(f"{package}.models")
    router_module = importlib.import_module(f"{package}.router")
    task_interface_module = importlib.import_module(f"{package}.task_interface")

    model = models.ChatCompletionsAPIModel(
        name=args.model,
        base_url=args.base_url,
        api_key="EMPTY",
        max_new_tokens=512,
        answer_max_new_tokens=128,
        temperature=0.0,
        system_prompt=(
            "You are a concise reasoning assistant. Use successful tool evidence "
            "and return only the requested final answer."
        ),
        timeout_seconds=300,
        max_retries=2,
        cleanup_output=True,
        enable_image_input=False,
        chat_template_kwargs={"enable_thinking": False},
        json_object_for_structured_phases=False,
    )
    task_config = task_interface_module.TaskInterfaceConfig(mode="rule")
    task_interface = task_interface_module.TaskInterface(task_config)
    example = {
        "task_id": "tool_smoke/mercury",
        "task_name": "tool_smoke",
        "task_type": "math_word_problem",
        "question": (
            "Mercury completes one revolution in 88 Earth days. "
            "Using 365.25 days per Earth year, how many Earth years is that? "
            "Return only a decimal rounded to two decimal places."
        ),
        "answer": "0.24",
    }
    chunk = task_interface.build_taskinterfacechunk(example, 0)

    with tempfile.TemporaryDirectory(prefix="hcl_tool_smoke_") as temp_dir:
        cache_root = Path(temp_dir)
        router = router_module.Router(
            model,
            router_module.RouterConfig(
                route_name="router_managed_arithmetic_smoke",
                use_llm_workflow_selector=True,
                use_memory_selector=False,
                memory_always_on=False,
                use_skill_selector=False,
                use_tool_selector=True,
                tool_auto_select=True,
                use_chat_tool_calls=False,
                tool_max_rounds=0,
                workflow_selector_cache_path=str(cache_root / "workflow.jsonl"),
                tool_selector_cache_path=str(cache_root / "tool_selection.jsonl"),
                tool_argument_cache_path=str(cache_root / "tool_arguments.jsonl"),
                final_generation_cache_path=str(cache_root / "final.jsonl"),
                capability_enabled=True,
                capability_execution_mode="router_managed",
                semantic_search_enabled=False,
                cross_modal_match_enabled=False,
            ),
        )
        result = router.run(chunk, task_config)

    trace = result["trace"]
    capability_results = trace.get("capability_results") or []
    if len(capability_results) != 1:
        raise SystemExit(f"expected one capability result, got: {capability_results!r}")
    capability = capability_results[0]
    if capability.get("tool_name") != "arithmetic_calculator":
        raise SystemExit(f"wrong selected capability: {capability!r}")
    if capability.get("status") != "success":
        raise SystemExit(f"capability did not execute successfully: {capability!r}")
    arguments = capability.get("arguments") or {}
    output = capability.get("output") or {}
    numeric_result = float(output.get("numeric_result"))
    if abs(numeric_result - (88 / 365.25)) > 1e-12:
        raise SystemExit(f"wrong calculator result: {capability!r}")
    if "0.240930869" not in str(trace.get("prompt", "")):
        raise SystemExit("successful calculator result was not injected into final context")

    print(json.dumps({
        "status": "passed",
        "selected_tools": [item.get("tool_name") for item in trace.get("selected_tools", [])],
        "capability": capability,
        "final_answer": result.get("answer"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
