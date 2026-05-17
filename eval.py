import argparse
import torch
from core.train import load_config, build_model, state_to_tensor
from main import build_env


def evaluate(env, model, config, num_episodes):
    """Run the agent greedily for num_episodes. Returns list of episode rewards."""
    model_type = config["model"]["type"]
    rewards = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = state_to_tensor(state, model_type)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()
            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        print(f"Episode {ep + 1:3d} | reward: {total_reward:.1f}")

    avg = sum(rewards) / len(rewards)
    print(f"\n{'='*40}")
    print(f"Episodes: {num_episodes}")
    print(f"Average reward: {avg:.1f}")
    print(f"Min: {min(rewards):.1f}  Max: {max(rewards):.1f}")

    env.close()
    return rewards


def main():
    parser = argparse.ArgumentParser(description="RL Harness — Evaluate a trained agent")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML (same one used for training)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to run (default: 10)")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable visual rendering")
    args = parser.parse_args()

    config = load_config(args.config)

    if not args.no_render:
        config["env"]["render_mode"] = "human"

    env = build_env(config)

    model = build_model(config)
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    model.eval()

    evaluate(env, model, config, args.episodes)


if __name__ == "__main__":
    main()
