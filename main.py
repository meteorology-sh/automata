import argparse
from core.train import load_config, train
from core.visualizer import Visualizer
from envs.base_env import GymEnv


def main():
    parser = argparse.ArgumentParser(description="RL Harness")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--no-vis", action="store_true",
                        help="Disable visualizer")
    args = parser.parse_args()

    config = load_config(args.config)

    env = GymEnv(config)

    visualizer = None
    if not args.no_vis and config["visualizer"]["enabled"]:
        num_actions = config["model"]["num_actions"]
        visualizer = Visualizer(num_actions=num_actions)
        visualizer.start()

    try:
        train(env, config, visualizer=visualizer)
    finally:
        if visualizer:
            visualizer.stop()


if __name__ == "__main__":
    main()
