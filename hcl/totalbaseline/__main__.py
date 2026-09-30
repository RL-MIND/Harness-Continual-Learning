from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..dataset import DatasetLoader, load_task_stream_specs
from ..evaluator import ExactMatchEvaluator
from ..json_utils import atomic_write_json
from ..models import ChatCompletionsAPIModel


def main() -> int:
    parser = argparse.ArgumentParser(description="Raw multimodal zero-shot baseline")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    task_stream = _resolve(config["task_stream_path"], base_dir)
    output_dir = _resolve(config["output_dir"], base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = dict(config["model"])
    model_config["backend"] = "chat_completions"
    model_config.setdefault("enable_image_input", True)
    model_config.setdefault("chat_template_kwargs", {"enable_thinking": False})
    model_config.setdefault("log_path", str(output_dir / "llm_calls.txt"))
    model = ChatCompletionsAPIModel(**model_config)
    loader = DatasetLoader()
    evaluator = ExactMatchEvaluator()
    split = str(config.get("split", "test"))
    limit = int(config.get("limit_per_task", 500))
    workers = max(int(config.get("workers", 1)), 1)
    resume = bool(config.get("resume", True))
    prediction_path = output_dir / "predictions.jsonl"
    cached = _load_predictions(prediction_path) if resume else {}

    per_task: dict[str, dict[str, object]] = {}
    all_examples: list[dict[str, Any]] = []
    all_predictions: list[dict[str, object]] = []
    specs = load_task_stream_specs(task_stream)
    total = sum(
        len(loader.load_task_spec(spec, split=split, dataset_index=index, limit=limit))
        for index, spec in enumerate(specs)
    )
    done = 0
    print(f"[raw-baseline] start tasks={len(specs)} examples={total} image_input=true hcl=false", flush=True)
    for dataset_index, spec in enumerate(specs):
        task_name = str(spec["task_name"])
        examples = loader.load_task_spec(spec, split=split, dataset_index=dataset_index, limit=limit)
        answers: dict[str, object] = {}
        pending: dict[object, tuple[str, int]] = {}
        executor = ThreadPoolExecutor(max_workers=workers)
        for example in examples:
            done += 1
            task_id = str(example["task_id"])
            if task_id in cached:
                answers[task_id] = cached[task_id]
            else:
                print(f"[raw-baseline] request item={done}/{total} task={task_name} id={task_id}", flush=True)
                future = executor.submit(_generate, model, example, task_id)
                pending[future] = (task_id, done)
        try:
            for future in as_completed(pending):
                task_id, item = pending[future]
                answer = future.result()
                _append_prediction(prediction_path, {
                    "task_id": task_id,
                    "task_name": task_name,
                    "answer": answer,
                    "input_policy": "raw_visible_input_with_image_no_hcl",
                })
                cached[task_id] = answer
                answers[task_id] = answer
                print(f"[raw-baseline] completed item={item}/{total} task={task_name} id={task_id}", flush=True)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        predictions = [
            {"task_id": str(example["task_id"]), "answer": answers[str(example["task_id"])]}
            for example in examples
        ]
        _rewrite_predictions(prediction_path, task_name, predictions)
        result = evaluator.evaluate(examples, predictions, split=f"{split}:{task_name}")
        per_task[task_name] = dict(result["metrics"])
        all_examples.extend(examples)
        all_predictions.extend(predictions)
        atomic_write_json(output_dir / "metrics.partial.json", {
            "completed_tasks": list(per_task), "per_task_metrics": per_task,
        })

    overall = evaluator.evaluate(all_examples, all_predictions, split=split)["metrics"]
    result = {
        "baseline": "raw_multimodal_visible_input_no_hcl",
        "task_stream_path": str(task_stream),
        "split": split,
        "limit_per_task": limit,
        "model": model_config,
        "overall_metrics": overall,
        "per_task_metrics": per_task,
    }
    atomic_write_json(output_dir / "metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


def _generate(model: ChatCompletionsAPIModel, example: dict[str, Any], task_id: str) -> str:
    return model.generate(
        _raw_visible_input(example),
        state={
            "phase": "final_answer",
            "task_id": task_id,
            "image": example.get("image"),
            "task_context": {
                "files": ([{"type": "image", "path": example["image"]}] if example.get("image") else []),
            },
        },
    )


def _raw_visible_input(example: dict[str, Any]) -> str:
    question = str(example.get("question") or "")
    additions: list[str] = []
    options = example.get("options")
    if options not in (None, [], {}):
        additions.append("Visible answer options:\n" + json.dumps(options, ensure_ascii=False))
    visible_context = example.get("visible_context")
    if visible_context not in (None, "", [], {}):
        additions.append("Visible context:\n" + json.dumps(visible_context, ensure_ascii=False))
    return "\n\n".join((question, *additions)) if additions else question


def _resolve(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _load_predictions(path: Path) -> dict[str, object]:
    rows: dict[str, object] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(row, dict)
            and row.get("task_id")
            and _is_valid_cached_answer(str(row.get("task_name", "")), row.get("answer"))
        ):
            rows[str(row["task_id"])] = row.get("answer", "")
    return rows


def _is_valid_cached_answer(task_name: str, answer: object) -> bool:
    if task_name == "coco_detection":
        return _is_valid_detection_answer(answer)
    if task_name == "refcoco_grounding":
        return _is_valid_grounding_answer(answer)
    return bool(str(answer or "").strip())


def _is_valid_detection_answer(answer: object) -> bool:
    text = str(answer or "").strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[len("```json") : -len("```")].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[len("```") : -len("```")].strip()
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return False
    return isinstance(value, list) and all(
        isinstance(item, dict)
        and isinstance(item.get("bbox_2d"), list)
        and len(item["bbox_2d"]) == 4
        and isinstance(item.get("label"), str)
        for item in value
    )


def _is_valid_grounding_answer(answer: object) -> bool:
    text = str(answer or "").strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[len("```json") : -len("```")].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[len("```") : -len("```")].strip()
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return False
    bbox = value.get("bbox") if isinstance(value, dict) else value
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(item) for item in bbox)
    except (TypeError, ValueError):
        return False
    return x2 > x1 and y2 > y1


def _append_prediction(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rewrite_predictions(
    path: Path,
    task_name: str,
    predictions: list[dict[str, object]],
) -> None:
    """Compact one task while retaining completed rows from other tasks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    retained: list[dict[str, object]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and str(row.get("task_name", "")) != task_name:
                retained.append(row)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in retained:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        for prediction in predictions:
            handle.write(json.dumps({
                "task_id": prediction["task_id"],
                "task_name": task_name,
                "answer": prediction["answer"],
                "input_policy": "raw_visible_input_with_image_no_hcl",
            }, ensure_ascii=False) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
