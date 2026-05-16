import torch
import yaml
from collections import deque

from core.agent import DQNAgent
from envs.base_env import BaseEnv


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(config: dict):
    model_type = config["model"]["type"]
    num_actions = config["model"]["num_actions"]

    if model_type == "mlp":
        from models.mlp import MLP
        return MLP(
            state_size=config["model"]["state_size"],
            num_actions=num_actions,
            hidden_size=config["model"]["hidden_size"],
        )
    elif model_type == "cnn":
        from models.cnn import CNN
        return CNN(
            image_size=config["env"]["image_size"],
            num_actions=num_actions,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def state_to_tensor(state, model_type: str) -> torch.Tensor:
    if model_type == "cnn":
        t = torch.tensor(state, dtype=torch.float32)
        return t.permute(2, 0, 1).unsqueeze(0)   # (H,W,3) → (1,3,H,W)
    else:
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)


def train(env: BaseEnv, config: dict, visualizer=None):
    model = build_model(config)
    agent = DQNAgent(model, config)

    episodes = config["training"]["episodes"]
    solve_threshold = config["training"]["solve_threshold"]
    save_every = config["checkpoints"]["save_every"]
    checkpoint_dir = config["checkpoints"]["dir"]
    model_type = config["model"]["type"]
    env_name = config["env"]["name"]

    recent_rewards = deque(maxlen=100)
    best_avg_reward = float("-inf")

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = state_to_tensor(state, model_type)
            action = agent.select_action(state_tensor)
            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated

            agent.remember(state, action, reward, next_state, done)
            loss = agent.train()

            if visualizer:
                q_values = model(state_tensor).detach().squeeze().tolist()
                visualizer.update(state, q_values, reward, agent.epsilon, loss)

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        recent_rewards.append(total_reward)
        avg_reward = sum(recent_rewards) / len(recent_rewards)

        print(
            f"Episode {episode:4d} | reward: {total_reward:6.1f} | avg: {avg_reward:6.1f} | epsilon: {agent.epsilon:.3f}")

        if episode % save_every == 0:
            path = f"{checkpoint_dir}{env_name}_{episode}.pt"
            torch.save(model.state_dict(), path)

        if config["checkpoints"]["keep_best"] and avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            torch.save(model.state_dict(),
                       f"{checkpoint_dir}{env_name}_best.pt")

        if len(recent_rewards) == 100 and avg_reward >= solve_threshold:
            print(
                f"Solved at episode {episode} with avg reward {avg_reward:.1f}")
            break

    env.close()
    return model
