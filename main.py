import argparse
import importlib
from typing import Any

from core.train import load_config, train
from core.visualizer import Visualizer
from envs.base_env import BaseEnv, GymEnv


def build_env(config: dict[str, Any]) -> BaseEnv:
    """Build the environment from config. Uses GymEnv unless env.wrapper specifies a custom class."""
    wrapper = config["env"].get("wrapper")
    if wrapper is None:
        return GymEnv(config)
    module_path, class_name = wrapper.rsplit(".", 1)
    module = importlib.import_module(module_path)
    env_cls: type[BaseEnv] = getattr(module, class_name)
    return env_cls(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="RL Harness")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--vis", action="store_true",
                        help="Enable real-time reward dashboard")
    args = parser.parse_args()

    config = load_config(args.config)

    env = build_env(config)

    visualizer = None
    if args.vis and config["visualizer"]["enabled"]:
        solve_threshold = config["training"]["solve_threshold"]
        solve_window = config["training"].get("solve_window", 50)
        visualizer = Visualizer(solve_threshold=solve_threshold, solve_window=solve_window)
        visualizer.start()

    try:
        train(env, config, visualizer=visualizer)
    finally:
        if visualizer:
            visualizer.stop()


if __name__ == "__main__":
    main()
