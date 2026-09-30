from __future__ import annotations

import argparse
import json
import math
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_TASKS = Path("curriculum/harness_survival_curriculum.txt")
DEFAULT_USERNAME = "hcurrbot"
MINECRAFT_USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_]{1,16}$")


def load_tasks(path: Path) -> list[str]:
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tasks.append(stripped)
    return tasks


def result_summary(result: dict) -> dict:
    route = result.get("route") or {}
    evaluation = result.get("evaluation") or {}
    return {
        "success": result.get("success"),
        "needs_user": result.get("needs_user", False),
        "route_decision": route.get("decision", ""),
        "skill_name": route.get("skill_name", ""),
        "reasoning": route.get("reasoning", ""),
        "critique": evaluation.get("critique", ""),
        "subgoals": result.get("subgoals", []),
        "completed_subgoals": result.get("completed_subgoals", []),
        "replan_count": result.get("replan_count", 0),
        "inventory": result.get("inventory", {}),
    }


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def read_task_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def resolve_run_dir(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_dir():
        return candidate
    candidate = Path("ckpt/harness/curriculum_runs") / value
    if candidate.is_dir():
        return candidate
    raise ValueError(f"Resume run not found: {value}")


def latest_observation(events: list[tuple[str, dict]] | None) -> dict:
    for event_type, event in reversed(events or []):
        if event_type == "observe" and isinstance(event, dict):
            return event
    for _, event in reversed(events or []):
        if isinstance(event, dict) and "inventory" in event:
            return event
    return {}


def infer_resume_state(run_dir: Path, task_count: int, default_start: int) -> dict[str, Any]:
    """Recover legacy runs that predate resume_state.json.

    A task is committed only after its summary says success. If a process was
    killed while task N was running, task N is deliberately selected again;
    the Harness re-observes the persistent Minecraft world and finishes the
    original task contract rather than resetting the world.
    """
    state_path = run_dir / "resume_state.json"
    state = load_json(state_path)
    if state:
        return state
    successful = {
        int(record["task_index"])
        for record in read_task_records(run_dir / "tasks.jsonl")
        if isinstance(record.get("task_index"), int)
        and isinstance(record.get("summary"), dict)
        and record["summary"].get("success") is True
    }
    # Legacy runs did not retain an active-task write-ahead record. They may
    # also contain an error entry followed by later successful tasks because
    # the old CLI continued after errors. The best durable recovery point is
    # therefore the task immediately after the highest committed success.
    next_task = max(successful, default=default_start - 1) + 1
    return {
        "version": 1,
        "status": "legacy_recovered",
        "committed_task_indices": sorted(successful),
        "next_task_index": next_task,
        "active_task": None,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def save_resume_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(path, state)


def validate_username(username: str) -> None:
    if MINECRAFT_USERNAME_PATTERN.fullmatch(username):
        return
    raise ValueError(
        "Minecraft bot username must be 1-16 characters and contain only letters, numbers, or underscores. "
        f"Got {username!r} with length {len(username)}."
    )


def validate_spawn_coordinates(values: dict[str, float | None]) -> None:
    for name, value in values.items():
        if value is not None and not math.isfinite(value):
            raise ValueError(f"--spawn-{name} must be a finite number, got {value!r}.")


def spawn_skill_parameters(spawn: dict[str, float | None]) -> list[float]:
    """Build positional JS arguments without serializing Python None."""
    values = [
        float(spawn["x"]),
        float(spawn["z"]),
        float(spawn.get("yaw") or 0),
        float(spawn.get("pitch") or 0),
    ]
    if spawn.get("y") is not None:
        values.append(float(spawn["y"]))
    return values


def transaction_baseline(
    previous_active: dict[str, Any],
    observed_inventory: dict[str, Any],
) -> dict[str, Any]:
    """Return the write-ahead baseline, never a later resume observation."""
    baseline = previous_active.get("baseline_inventory")
    return dict(baseline) if isinstance(baseline, dict) else dict(observed_inventory)


def contract_uses_baseline(contract: Any, baseline_inventory: dict[str, Any]) -> bool:
    if not isinstance(contract, dict):
        return False
    saved = contract.get("baseline_inventory")
    if not isinstance(saved, dict):
        return False
    try:
        return {str(key): int(value) for key, value in saved.items()} == {
            str(key): int(value) for key, value in baseline_inventory.items()
        }
    except (TypeError, ValueError):
        return False


def contract_uses_semantic_compiler(contract: Any) -> bool:
    """Migrate older semantic interpretations using their original transaction baseline."""
    return (
        isinstance(contract, dict)
        and contract.get("semantic_contract_version") == 2
        and contract.get("semantic_source") in {
        "llm_validated",
        "narrow_rule_fallback",
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Run a sequential Harness continual-learning curriculum.")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS, help="Path to a text file with one task per line.")
    parser.add_argument("--start", type=int, default=1, help="1-based task index to start from.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of tasks to run.")
    parser.add_argument("--max-steps", type=int, default=3, help="Harness attempts per subgoal.")
    parser.add_argument("--max-replans", type=int, default=2, help="Replans after subgoal failure.")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop the curriculum when one task fails.")
    parser.add_argument("--pause", action="store_true", help="Wait for Enter before each task.")
    parser.add_argument("--username", default=DEFAULT_USERNAME, help="Minecraft bot username, max 16 characters.")
    parser.add_argument("--env-wait-ticks", type=int, default=40, help="Mineflayer wait ticks after reset/step.")
    parser.add_argument(
        "--memory-config",
        type=Path,
        default=None,
        help="Optional Harness backend JSON, e.g. conf/retry_baseline.json.",
    )
    parser.add_argument("--spawn-x", type=float, default=0.5, help="Fixed initial X coordinate for a new run.")
    parser.add_argument("--spawn-z", type=float, default=0.5, help="Fixed initial Z coordinate for a new run.")
    parser.add_argument(
        "--spawn-y",
        type=float,
        default=None,
        help="Optional fixed initial Y. By default use the terrain surface at spawn X/Z.",
    )
    parser.add_argument("--spawn-yaw", type=float, default=0.0, help="Fixed initial yaw.")
    parser.add_argument("--spawn-pitch", type=float, default=0.0, help="Fixed initial pitch.")
    parser.add_argument(
        "--resume-run",
        "--resume",
        dest="resume_run",
        help="Existing run id or run directory to resume without resetting the Minecraft world.",
    )
    args = parser.parse_args()
    validate_username(args.username)
    requested_spawn = {
        "x": args.spawn_x,
        "y": args.spawn_y,
        "z": args.spawn_z,
        "yaw": args.spawn_yaw,
        "pitch": args.spawn_pitch,
    }
    validate_spawn_coordinates(requested_spawn)
    spawn_parameters = spawn_skill_parameters(requested_spawn)

    tasks = load_tasks(args.tasks)
    if args.start < 1 or args.start > len(tasks):
        raise ValueError(f"--start must be between 1 and {len(tasks)}")
    if args.resume_run:
        run_dir = resolve_run_dir(args.resume_run)
        run_metadata = load_json(run_dir / "run.json") or {}
        saved_tasks_file = run_metadata.get("tasks_file")
        if saved_tasks_file and Path(saved_tasks_file) != args.tasks:
            raise ValueError(
                f"Resume run was created with tasks file {saved_tasks_file!r}; got {str(args.tasks)!r}."
            )
        saved_spawn = run_metadata.get("initial_spawn")
        if saved_spawn and saved_spawn.get("requested") != requested_spawn:
            print(
                "Resume mode ignores the newly supplied spawn coordinates and preserves "
                "the bot's live position from the existing transaction."
            )
        resume_state = infer_resume_state(run_dir, len(tasks), args.start)
        active_task = resume_state.get("active_task") or {}
        start_index = int(active_task.get("task_index") or resume_state.get("next_task_index") or args.start)
        if start_index > len(tasks):
            print(f"Run {run_dir.name} is already complete.")
            return
        run_id = run_dir.name
        resumed = True
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("ckpt/harness/curriculum_runs") / run_id
        start_index = args.start
        resumed = False
        resume_state = {
            "version": 1,
            "status": "ready",
            "committed_task_indices": [],
            "next_task_index": start_index,
            "active_task": None,
        }

    selected = tasks[start_index - 1 :]
    if args.limit is not None:
        selected = selected[: args.limit]

    summary_path = run_dir / "tasks.jsonl"
    steps_path = run_dir / "steps.jsonl"
    state_path = run_dir / "resume_state.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    if not resumed:
        write_json(
            run_dir / "run.json",
            {
                "run_id": run_id,
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "tasks_file": str(args.tasks),
                "start": args.start,
                "limit": args.limit,
                "max_steps": args.max_steps,
                "max_replans": args.max_replans,
                "username": args.username,
                "memory_config": str(args.memory_config) if args.memory_config else None,
                "initial_spawn": {
                    "requested": requested_spawn,
                    "resolved": None,
                },
            },
        )
    save_resume_state(state_path, resume_state)

    from main import build_odyssey

    agent = build_odyssey(
        environment="harness_curriculum",
        username=args.username,
        env_wait_ticks=args.env_wait_ticks,
        max_iterations=64,
        memory_config_path=str(args.memory_config) if args.memory_config else None,
        # Resume only Harness transaction/memory state. The original Odyssey
        # Action/Curriculum checkpoints are independent and need not exist.
        harness_resume=resumed,
    )
    from odyssey.harness.completion_contract import build_completion_contract

    print(f"Loaded {len(tasks)} curriculum tasks from {args.tasks}")
    print(f"Running {len(selected)} tasks starting at index {start_index}")
    print(f"Results will be written to {run_dir}")
    if resumed:
        print("Resume mode: preserving the existing Minecraft world, bot inventory, and position.")
    else:
        y_description = requested_spawn["y"] if requested_spawn["y"] is not None else "terrain surface"
        print(
            "Deterministic initial spawn: "
            f"x={requested_spawn['x']}, y={y_description}, z={requested_spawn['z']}"
        )
    print("Minecraft must already be running through PCL and opened to LAN.")

    try:
        if resumed:
            # Bridge.step requires a per-process reset handshake. A soft reset
            # only reconnects Mineflayer and observes the existing player; it
            # does not invoke respawnAndClear or clear inventory/world state.
            agent.last_events = agent.env.reset(
                options={
                    "mode": "soft",
                    "wait_ticks": agent.env_wait_ticks,
                    "username": agent.username,
                }
            )
            if not agent.last_events:
                agent.last_events = agent.env.step("")
            observation = latest_observation(agent.last_events)
            resume_state["resume_observation"] = {
                "time": datetime.now().isoformat(timespec="seconds"),
                "inventory": observation.get("inventory", {}),
                "position": (observation.get("status") or {}).get("position", {}),
            }
            resume_state["status"] = "running"
            save_resume_state(state_path, resume_state)
        else:
            agent.env.reset(
                options={
                    "mode": "hard",
                    "wait_ticks": agent.env_wait_ticks,
                    "username": agent.username,
                }
            )
            agent.run_raw_skill("odyssey/test_env/respawnAndClear.js", spawn_parameters)
            agent.last_events = agent.env.step("")
            initial_observation = latest_observation(agent.last_events)
            run_metadata = load_json(run_dir / "run.json") or {}
            initial_spawn = run_metadata.get("initial_spawn") or {"requested": requested_spawn}
            initial_spawn["resolved"] = (initial_observation.get("status") or {}).get("position", {})
            run_metadata["initial_spawn"] = initial_spawn
            write_json(run_dir / "run.json", run_metadata)

        for offset, task in enumerate(selected, start=start_index):
            # A curriculum file may be improved while a run is paused.  The
            # active transaction must nevertheless execute its originally
            # persisted instruction, otherwise its write-ahead contract and
            # the instruction being evaluated could silently diverge.
            persisted_active = resume_state.get("active_task") or {}
            if persisted_active.get("task_index") == offset and isinstance(persisted_active.get("instruction"), str):
                task = persisted_active["instruction"]
            print("\n" + "=" * 80)
            print(f"Task {offset}/{len(tasks)}")
            print(task)
            if args.pause:
                input("Press Enter to run this task...")

            # Write-ahead task record: after a forced stop this remains the
            # active task and resume will re-observe the world before retrying
            # this exact original instruction.
            previous_active = resume_state.get("active_task") or {}
            observed_inventory = latest_observation(agent.last_events).get("inventory", {})
            baseline_inventory = transaction_baseline(previous_active, observed_inventory)
            original_contract = previous_active.get("completion_contract")
            # A contract is part of the transaction. On resume it must be
            # anchored to the write-ahead inventory snapshot from when this
            # task first started, not to the partially completed live state.
            # The baseline check also migrates transactions produced by the
            # earlier resume implementation that used the later observation.
            if (
                not contract_uses_baseline(original_contract, baseline_inventory)
                or not contract_uses_semantic_compiler(original_contract)
            ):
                original_contract = build_completion_contract(
                    task,
                    baseline_inventory,
                    model_name=agent.action_agent_model_name,
                )
            resume_state["status"] = "running"
            resume_state["next_task_index"] = offset
            resume_state["active_task"] = {
                "task_index": offset,
                "instruction": task,
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "baseline_inventory": baseline_inventory,
                "completion_contract": original_contract,
            }
            save_resume_state(state_path, resume_state)

            try:
                result = agent.harness_run_instruction(
                    task,
                    reset_env=False,
                    reset_mode="hard",
                    max_steps=args.max_steps,
                    max_replans=args.max_replans,
                    completion_contract=original_contract,
                )
                summary = result_summary(result)
                print(f"success: {summary['success']}")
                print(f"route: {summary['route_decision']} {summary['skill_name']}")
                if summary["reasoning"]:
                    print(f"reason: {summary['reasoning']}")
                if summary["critique"]:
                    print(f"critique: {summary['critique']}")
                task_record = {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "task_index": offset,
                    "task": task,
                    "summary": summary,
                }
                append_jsonl(summary_path, task_record)
                write_json(
                    run_dir / f"task_{offset:03d}.json",
                    {
                        **task_record,
                        "steps": result.get("trajectory", []),
                    },
                )
                for entry in result.get("trajectory", []):
                    append_jsonl(
                        steps_path,
                        {
                            "task_index": offset,
                            "task": task,
                            **entry,
                        },
                    )
                if summary["success"]:
                    committed = set(resume_state.get("committed_task_indices", []))
                    committed.add(offset)
                    resume_state["committed_task_indices"] = sorted(committed)
                    resume_state["next_task_index"] = offset + 1
                    resume_state["active_task"] = None
                    resume_state["status"] = "ready"
                else:
                    # A failed task is deliberately not committed. If the
                    # process stops, resume returns here and evaluates the
                    # original contract against the live world state.
                    resume_state["status"] = "retryable_failure"
                save_resume_state(state_path, resume_state)
                if not summary["success"]:
                    print("Task was not committed; stopping so resume can retry this task transactionally.")
                    break
            except Exception as exc:  # noqa: BLE001
                print(f"error: {exc}")
                traceback.print_exc()
                error_record = {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "task_index": offset,
                    "task": task,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(summary_path, error_record)
                write_json(run_dir / f"task_{offset:03d}.json", error_record)
                resume_state["status"] = "interrupted"
                resume_state["next_task_index"] = offset
                # Keep active_task intact so resume retries this task.
                save_resume_state(state_path, resume_state)
                print("Task raised an exception; stopping so resume can retry this task transactionally.")
                break
    finally:
        if resume_state.get("next_task_index", 1) > len(tasks):
            resume_state["status"] = "completed"
            resume_state["active_task"] = None
            save_resume_state(state_path, resume_state)
        if resume_state.get("status") == "running":
            resume_state["status"] = "interrupted"
            save_resume_state(state_path, resume_state)
        agent.close()


if __name__ == "__main__":
    main()
