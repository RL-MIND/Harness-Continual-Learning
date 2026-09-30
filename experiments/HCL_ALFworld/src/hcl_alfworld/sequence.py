from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

from .environment import read_task_type, validate_data_dir
from .task_interface import TASK_TYPE_NAMES


def _index_split(data_dir: Path, split: str) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = defaultdict(list)
    split_dir = data_dir / "json_2.1.1" / split
    for game_file in sorted(split_dir.rglob("game.tw-pddl")):
        if "movable" in str(game_file).lower() or "sliced" in str(game_file).lower():
            continue
        try:
            with game_file.open("r", encoding="utf-8") as handle:
                game_data = json.load(handle)
            if not game_data.get("solvable", False):
                continue
            relative_game = game_file.resolve().relative_to(data_dir.resolve())
            result[read_task_type(game_file)].append(str(relative_game))
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    return dict(result)


def build_sequence(
    data_dir: str | Path,
    output_path: str | Path,
    task_order: Iterable[int],
    train_episodes_per_task: int,
    eval_episodes_per_task: Union[int, str],
    train_split: str,
    eval_split: str,
    seed: int,
) -> Dict[str, Any]:
    data_dir = Path(data_dir).resolve()
    validation = validate_data_dir(data_dir)
    if not validation["valid"]:
        raise FileNotFoundError("Incomplete ALFWorld data: " + ", ".join(validation["missing"]))
    rng = random.Random(seed)
    train_index = _index_split(data_dir, train_split)
    eval_index = _index_split(data_dir, eval_split)
    use_all_eval_games = str(eval_episodes_per_task).lower() == "all"
    eval_limit = None if use_all_eval_games else int(eval_episodes_per_task)
    phases: List[Dict[str, Any]] = []
    for phase_index, task_id in enumerate(task_order):
        task_type = TASK_TYPE_NAMES[int(task_id)]
        train_games = list(train_index.get(task_type, []))
        eval_games = list(eval_index.get(task_type, []))
        rng.shuffle(train_games)
        rng.shuffle(eval_games)
        if len(train_games) < train_episodes_per_task:
            raise ValueError(f"Only {len(train_games)} train games available for {task_type}.")
        if eval_limit is not None and len(eval_games) < eval_limit:
            raise ValueError(f"Only {len(eval_games)} eval games available for {task_type}.")
        phases.append(
            {
                "phase": phase_index,
                "task_id": int(task_id),
                "task_type": task_type,
                "train_games": train_games[:train_episodes_per_task],
                "eval_games": eval_games if eval_limit is None else eval_games[:eval_limit],
            }
        )
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "game_paths_relative_to": "data_dir",
        "train_split": train_split,
        "eval_split": eval_split,
        "task_order": [int(item) for item in task_order],
        "eval_episodes_per_task": "all" if use_all_eval_games else eval_limit,
        "phases": phases,
        "data_validation": {key: value for key, value in validation.items() if key != "data_dir"},
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    return manifest


def load_sequence(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
