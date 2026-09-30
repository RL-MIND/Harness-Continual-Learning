from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


SPLIT_ALIASES = {
    "val": ("val", "validation", "eval"),
    "validation": ("validation", "val", "eval"),
    "eval": ("eval", "validation", "val"),
}


class DatasetLoader:
    def load(self, source: str | Path | Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(source, (str, Path)):
            return self._load_path(Path(source))
        examples: list[dict[str, Any]] = []
        for index, item in enumerate(source):
            examples.append(_example_from_record(item, index=index))
        return examples

    def _load_path(self, path: Path) -> list[dict[str, Any]]:
        if path.is_dir():
            for candidate in (path / "test.json", path / "eval.json", path / "val.json", path / "train.json"):
                if candidate.exists():
                    return self._load_path(candidate)
            json_files = sorted(path.glob("*.json"))
            if len(json_files) == 1:
                return self._load_path(json_files[0])
            raise ValueError(f"Unsupported dataset directory shape: {path}")
        if path.suffix == ".jsonl":
            rows = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
            return self.load(rows)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ("examples", "data", "instances"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:
                data = []
        if not isinstance(data, list):
            raise ValueError(f"Dataset must be a list or an object with examples: {path}")
        return self.load(data)

    def load_task_stream(
        self,
        task_stream_path: str | Path,
        *,
        split: str,
        limit_per_task: int | None = None,
    ) -> list[dict[str, Any]]:
        specs = load_task_stream_specs(task_stream_path)
        examples: list[dict[str, Any]] = []
        for dataset_index, spec in enumerate(specs):
            examples.extend(
                self.load_task_spec(
                    spec,
                    split=split,
                    dataset_index=dataset_index,
                    limit=limit_per_task,
                )
            )
        return examples

    def load_task_spec(
        self,
        spec: dict[str, Any],
        *,
        split: str,
        dataset_index: int,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Load one task without losing its boundary in a continual stream."""
        split_path = _split_path(spec, split)
        if not split_path:
            return []
        examples = self.load(split_path)
        if limit is not None:
            examples = examples[: max(limit, 0)]
        task_name = str(spec.get("task_name", ""))
        for task_index, example in enumerate(examples):
            source_task_id = str(example.get("task_id") or task_index)
            example["source_task_id"] = source_task_id
            example["task_id"] = f"{task_name}/{split}/{source_task_id}"
            example["dataset_index"] = dataset_index
            example["task_index"] = task_index
            example["task_name"] = task_name
            example["task_type"] = str(spec.get("task_type", ""))
            example["split"] = split
            if spec.get("image_root") and example.get("image"):
                image_path = Path(str(example["image"]))
                if not image_path.is_absolute():
                    example["image"] = str(Path(str(spec["image_root"])) / image_path)
                elif not image_path.exists():
                    example["image"] = str(Path(str(spec["image_root"])) / image_path.name)
        return examples


def load_task_stream_specs(task_stream_path: str | Path) -> list[dict[str, Any]]:
    path = Path(task_stream_path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Task stream must be a JSON object: {path}")
    base_dir = path.parent.parent if path.parent.name == "configs" else path.parent
    specs: list[dict[str, Any]] = []
    for task in raw.get("tasks", []):
        if not isinstance(task, dict):
            continue
        splits = dict(task.get("splits", {}))
        for split, split_path in list(splits.items()):
            if split_path and not Path(str(split_path)).is_absolute():
                splits[split] = str((base_dir / str(split_path)).resolve())
        image_root = task.get("image_root")
        if image_root and not Path(str(image_root)).is_absolute():
            image_root = str((base_dir / str(image_root)).resolve())
        specs.append(
            {
                "task_name": str(task.get("task_name", "")),
                "task_type": str(task.get("task_type", "")),
                "modality": list(task.get("modality", ["text"])),
                "interactions": int(task.get("interactions", task.get("iterations", 1)) or 1),
                "splits": splits,
                "image_root": image_root,
            }
        )
    return specs


def _split_path(spec: dict[str, Any], split: str) -> str | None:
    splits = spec.get("splits", {})
    if not isinstance(splits, dict):
        return None
    aliases = SPLIT_ALIASES.get(split, (split,))
    for alias in aliases:
        value = splits.get(alias)
        if value:
            return str(value)
    return None


def _example_from_record(record: dict[str, Any], *, index: int) -> dict[str, Any]:
    return {
        "task_id": str(record.get("task_id") or record.get("question_id") or record.get("id") or index),
        "question": str(record.get("question") or record.get("prompt") or record.get("text") or record.get("input") or ""),
        "answer": record.get("answer") or record.get("target") or record.get("output"),
        "image": record.get("image") or record.get("image_path"),
        "task_name": str(record.get("task_name", "")),
        "task_type": str(record.get("task_type", "")),
        "split": str(record.get("split", "")),
        "metadata": {
            **dict(record.get("metadata", {})),
            **({"answers": record["answers"]} if isinstance(record.get("answers"), list) else {}),
        },
        **({"options": record.get("options", record.get("choices"))} if record.get("options", record.get("choices")) is not None else {}),
        **({"visible_context": record["visible_context"]} if record.get("visible_context") is not None else {}),
        **{
            key: record[key]
            for key in ("tool_output", "environment_observation", "user_feedback", "action_trace")
            if record.get(key) is not None
        },
    }
