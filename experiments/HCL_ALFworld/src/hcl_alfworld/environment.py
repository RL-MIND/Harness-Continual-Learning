from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List

import textworld
import textworld.gym


# TextWorld uses both a process-global Gym registry and a process-global TatSu
# PDDL/text-generation parsers. Parsing happens both during env.reset() and in
# every env.step() while TextWorld derives command text in _gather_infos(). All
# TextWorld calls that can enter TatSu must therefore share one lock. LLM calls,
# which dominate runtime, remain fully concurrent between episodes.
_ENV_INITIALIZATION_LOCK = threading.Lock()


class ALFWorldEpisodeEnv:
    """A deterministic one-game ALFWorld TextWorld adapter."""

    def __init__(
        self,
        game_file: str,
        data_dir: str | Path,
        max_steps: int = 50,
        domain_randomization: bool = False,
    ):
        self.game_file = str(Path(game_file).resolve())
        self.data_dir = Path(data_dir).resolve()
        self.max_steps = max_steps
        # ALFWorld creates its default cache at import time. Point it at the
        # configured project data before the lazy import, so even restricted
        # machines never try to write ~/.cache/alfworld.
        os.environ["ALFWORLD_DATA"] = str(self.data_dir)
        from alfworld.agents.environment.alfred_tw_env import (
            AlfredDemangler,
            AlfredInfos,
        )

        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        wrappers = [AlfredDemangler(shuffle=domain_randomization), AlfredInfos]
        # Gym's registry is process-global. Only registration/creation needs to be
        # serialized; completed environments can still execute concurrently.
        with _ENV_INITIALIZATION_LOCK:
            env_id = textworld.gym.register_games(
                [self.game_file],
                request_infos,
                batch_size=1,
                asynchronous=False,
                max_episode_steps=max_steps,
                wrappers=wrappers,
            )
            self.env = textworld.gym.make(env_id)

    def reset(self):
        with _ENV_INITIALIZATION_LOCK:
            return self.env.reset()

    def step(self, action: str):
        with _ENV_INITIALIZATION_LOCK:
            return self.env.step([action])

    def close(self) -> None:
        self.env.close()


def validate_data_dir(data_dir: str | Path) -> Dict[str, Any]:
    data_dir = Path(data_dir).resolve()
    required = [
        data_dir / "json_2.1.1" / "train",
        data_dir / "json_2.1.1" / "valid_seen",
        data_dir / "json_2.1.1" / "valid_unseen",
        data_dir / "logic" / "alfred.pddl",
        data_dir / "logic" / "alfred.twl2",
        data_dir / "detectors" / "mrcnn_alfred_objects_sep13_004.pth",
    ]
    missing = [str(path) for path in required if not path.exists()]
    counts = {}
    for split in ("train", "valid_seen", "valid_unseen"):
        root = data_dir / "json_2.1.1" / split
        counts[split] = len(list(root.rglob("game.tw-pddl"))) if root.exists() else 0
    return {
        "data_dir": str(data_dir),
        "missing": missing,
        "game_counts": counts,
        "valid": not missing,
    }


def read_task_type(game_file: str | Path) -> str:
    with Path(game_file).with_name("traj_data.json").open("r", encoding="utf-8") as handle:
        return str(json.load(handle)["task_type"])


def list_split_games(data_dir: str | Path, split: str) -> List[Dict[str, str]]:
    """List every game in an official ALFWorld split without balancing or sampling."""

    data_dir = Path(data_dir).resolve()
    split_dir = data_dir / "json_2.1.1" / split
    if not split_dir.exists():
        raise FileNotFoundError(f"ALFWorld split does not exist: {split_dir}")
    games = []
    for game_file in sorted(split_dir.rglob("game.tw-pddl")):
        games.append(
            {
                "game_file": str(game_file.resolve().relative_to(data_dir)),
                "task_type": read_task_type(game_file),
            }
        )
    return games
