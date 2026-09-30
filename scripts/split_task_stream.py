from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def read_records(path: Path) -> tuple[list[dict[str, Any]], str]:
    if path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [dict(row) for row in rows if isinstance(row, dict)], "jsonl"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [dict(row) for row in raw if isinstance(row, dict)], "list"
    if isinstance(raw, dict):
        for key in ("data", "examples", "instances"):
            value = raw.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value if isinstance(row, dict)], key
    raise ValueError(f"Unsupported dataset shape: {path}")


def write_records(path: Path, records: list[dict[str, Any]], shape: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if shape == "jsonl":
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        return
    payload: Any = records if shape == "list" else {shape: records}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_records(
    records: list[dict[str, Any]],
    *,
    val_ratio: float,
    min_val: int,
    max_val: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        return [], []
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    val_size = max(min_val, int(round(len(shuffled) * val_ratio)))
    if max_val is not None:
        val_size = min(val_size, max_val)
    val_size = max(1, min(val_size, len(shuffled) - 1)) if len(shuffled) > 1 else 1
    return shuffled[val_size:], shuffled[:val_size]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split each task's train file into deterministic train and validation files."
    )
    parser.add_argument("--config", type=Path, required=True, help="Task-stream config to read.")
    parser.add_argument("--output-config", type=Path, help="Updated task-stream config to write.")
    parser.add_argument("--output-dir", type=Path, help="Common output directory for split files.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--min-val", type=int, default=1)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1.")
    if args.min_val < 0 or (args.max_val is not None and args.max_val < 1):
        raise ValueError("Validation limits must be non-negative and --max-val must be positive.")

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_base = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    output_root = args.output_dir.resolve() if args.output_dir else None
    changed = False

    for task_index, task in enumerate(config.get("tasks", [])):
        task_name = str(task.get("task_name", "task"))
        splits = task.get("splits", {})
        if not isinstance(splits, dict) or not splits.get("train"):
            print(f"SKIP {task_name}: no train split")
            continue
        train_path = _resolve_manifest_path(str(splits["train"]), config_base)
        records, shape = read_records(train_path)
        train_records, val_records = split_records(
            records,
            val_ratio=args.val_ratio,
            min_val=args.min_val,
            max_val=args.max_val,
            seed=args.seed + task_index,
        )
        target_dir = output_root / task_name if output_root else train_path.parent
        suffix = ".jsonl" if shape == "jsonl" else ".json"
        split_train_path = target_dir / f"train_split{suffix}"
        split_val_path = target_dir / f"validation_split{suffix}"
        print(
            f"{task_name}: {len(records)} -> train={len(train_records)} "
            f"validation={len(val_records)} ({split_train_path}, {split_val_path})"
        )
        if args.dry_run:
            continue
        if not args.overwrite and (split_train_path.exists() or split_val_path.exists()):
            raise FileExistsError(
                f"Split file already exists for {task_name}; use --overwrite to replace."
            )
        write_records(split_train_path, train_records, shape)
        write_records(split_val_path, val_records, shape)
        splits["train"] = _portable_path(split_train_path, config_base)
        splits["validation"] = _portable_path(split_val_path, config_base)
        splits["val"] = splits["validation"]
        splits.setdefault("anchor", splits["validation"])
        changed = True

    if args.output_config and changed and not args.dry_run:
        args.output_config.parent.mkdir(parents=True, exist_ok=True)
        args.output_config.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    elif changed and not args.dry_run:
        print("Split files written. Pass --output-config to write an updated task-stream config.")
    return 0


def _resolve_manifest_path(raw_path: str, config_base: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (config_base / path).resolve()


def _portable_path(path: Path, config_base: Path) -> str:
    try:
        return str(path.resolve().relative_to(config_base.resolve()))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
