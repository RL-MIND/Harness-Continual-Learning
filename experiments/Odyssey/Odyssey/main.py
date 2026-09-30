import traceback

from odyssey.odyssey import Odyssey
from odyssey.utils import config
from odyssey.utils.logger import get_logger


logger = get_logger("main")
mc_port = config.get("MC_SERVER_PORT")
mc_host = config.get("MC_SERVER_HOST")
node_port = config.get("NODE_SERVER_PORT")
embedding_config = config.get("EMBEDDING") or config.get("SENTENT_EMBEDDING_DIR")
env_wait_ticks = 100


def build_odyssey(environment=None, username="bot", **kwargs):
    wait_ticks = kwargs.pop("env_wait_ticks", env_wait_ticks)
    return Odyssey(
        mc_port=mc_port,
        mc_host=mc_host,
        env_wait_ticks=wait_ticks,
        skill_library_dir="./skill_library",
        reload=True,
        embedding_dir=embedding_config,
        environment=environment,
        # The ordinary Odyssey agents start fresh for curriculum harness runs.
        # Harness transaction recovery is controlled separately by
        # ``harness_resume`` so it never attempts to load ckpt/action or
        # ckpt/curriculum.
        resume=False,
        server_port=node_port,
        username=username,
        **kwargs,
    )


def debug_lightweight():
    """Run a small LLM-driven task to craft a stone pickaxe and drop it."""
    odyssey_debug = build_odyssey(
        environment="subgoal",
        username="debug_bot",
        env_wait_ticks=40,
        max_iterations=8,
        action_agent_task_max_retries=2,
    )
    try:
        odyssey_debug.inference_sub_goal(
            task="debug_craft_and_drop_stone_pickaxe",
            sub_goals=["craft stone pickaxe", "drop stone pickaxe"],
            reset_mode="soft",
            reset_env=False,
        )
    finally:
        odyssey_debug.close()


def test_subgoal():
    odyssey_agent = build_odyssey(environment="subgoal")
    test_sub_goals = [
        "craft crafting table",
        "craft wooden pickaxe",
        "craft stone pickaxe",
        "craft iron pickaxe",
        "mine diamond",
    ]
    while True:
        try:
            odyssey_agent.inference_sub_goal(
                task="subgoal_openai",
                sub_goals=test_sub_goals,
            )
        except Exception as e:
            logger.critical(e)
            traceback.print_exc()


def test_combat():
    odyssey_agent = build_odyssey(environment="combat")
    combat_benchmark = [
        "1 skeleton",
        "1 spider",
        "1 zombified_piglin",
        "1 zombie",
        "1 zombie, 1 skeleton",
        "1 zombie, 1 spider",
        "1 zombie, 1 skeleton, 1 spider",
        "3 zombie",
    ]
    multi_round_tasks = ["1 zombie", "1 skeleton", "1 spider"]
    max_retry = 3

    while True:
        _run_combat_tasks(odyssey_agent, combat_benchmark, feedback_rounds=1, max_retry=max_retry)
        _run_combat_tasks(odyssey_agent, multi_round_tasks, feedback_rounds=3, max_retry=max_retry)


def _run_combat_tasks(odyssey_agent, tasks, feedback_rounds, max_retry):
    retry = max_retry
    i = 0
    while i < len(tasks):
        task = tasks[i]
        try:
            odyssey_agent.inference(task=task, reset_env=False, feedback_rounds=feedback_rounds)
            i += 1
            retry = max_retry
        except Exception as e:
            logger.critical(f"{task} failed. retry...")
            logger.critical(e)
            traceback.print_exc()
            if retry > 0:
                retry -= 1
                continue
            i += 1
            retry = max_retry


def explore():
    odyssey_agent = build_odyssey(environment="explore")
    odyssey_agent.learn()


def test_farming():
    odyssey_agent = build_odyssey(environment="farming")
    farming_benchmark = [
        "collect 1 seed (wheat or melon or pumpkin)",
        "hoe a farmland",
        "collect 1 wool by shearing 1 sheep",
        "collect 1 bucket of milk",
        "cook 1 meat (beef or mutton or pork or chicken)",
        "obtain 1 leather",
        "make 1 sugar",
        "collect 1 bucket of water",
    ]
    while True:
        for task in farming_benchmark:
            try:
                odyssey_agent.learn(goals=task, reset_env=False)
            except Exception as e:
                logger.critical(f"{task} failed. retry...")
                logger.critical(e)
                traceback.print_exc()


def test_skill(skill_name):
    odyssey_skill = build_odyssey()
    odyssey_skill.run_raw_skill(f"./skill_library/skill/compositional/{skill_name}", skill_lib="old", reset=True)
    while True:
        odyssey_skill.run_raw_skill(f"./skill_library/skill/compositional/{skill_name}", reset=False)


def test_mc_skill(skill_name, para_list):
    odyssey_mc_skill = build_odyssey(username="bot")
    odyssey_mc_skill.run_raw_skill(
        f"../MC-Comprehensive-Skill-Library/skill/{skill_name}",
        para_list,
        skill_lib="new",
        reset=True,
    )
    while True:
        odyssey_mc_skill.run_raw_skill(
            f"../MC-Comprehensive-Skill-Library/skill/{skill_name}",
            para_list,
            skill_lib="new",
            reset=False,
        )


if __name__ == "__main__":
    debug_lightweight()
