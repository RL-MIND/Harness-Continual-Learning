import argparse
import traceback

from main import build_odyssey


def main():
    parser = argparse.ArgumentParser(description="Run the interactive Odyssey Harness.")
    parser.add_argument(
        "--memory-config",
        default=None,
        help="Optional Harness backend JSON, e.g. conf/retry_baseline.json.",
    )
    args = parser.parse_args()
    agent = build_odyssey(
        environment="harness",
        username="harness_bot",
        env_wait_ticks=40,
        max_iterations=32,
        memory_config_path=args.memory_config,
    )
    print("Harness interactive mode. Type an instruction, or type exit to quit.")
    print("Minecraft must already be running and reachable by the configured Mineflayer port.")
    try:
        agent.env.reset(
            options={
                "mode": "hard",
                "wait_ticks": agent.env_wait_ticks,
                "username": agent.username,
            }
        )
        agent.run_raw_skill("odyssey/test_env/respawnAndClear.js")
        agent.last_events = agent.env.step("")
        reset_next = False
        while True:
            instruction = input("mc> ").strip()
            if not instruction:
                continue
            if instruction.lower() in {"exit", "quit", "q"}:
                break
            try:
                result = agent.harness_run_instruction(
                    instruction,
                    reset_env=reset_next,
                    reset_mode="hard",
                    max_steps=3,
                )
                reset_next = False
                print(f"success: {result.get('success')}")
                route = result.get("route") or {}
                if route:
                    print(f"route: {route.get('decision')} {route.get('skill_name', '')}")
                    if route.get("reasoning"):
                        print(f"reason: {route.get('reasoning')}")
                evaluation = result.get("evaluation") or {}
                if evaluation.get("critique"):
                    print(f"critique: {evaluation.get('critique')}")
                if result.get("needs_user"):
                    print(result.get("message", "Harness needs more information."))
            except Exception as exc:  # noqa: BLE001
                print(f"error: {exc}")
                traceback.print_exc()
    finally:
        agent.close()


if __name__ == "__main__":
    main()
