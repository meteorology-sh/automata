import os
from collections import deque
from typing import Any

import torch
import torch.nn as nn
import yaml

from core.agent import DQNAgent
from envs.base_env import BaseEnv


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(config: dict[str, Any]) -> nn.Module:
    model_type = config["model"]["type"]
    num_actions = config["model"]["num_actions"]

    if model_type == "mlp":
        # Dueling head (config-gated) splits Q into V(s) + centred A(s,a). Use it when the
        # action-relative signal is small next to the state value. Default off = plain MLP.
        from models.mlp import MLP, DuelingMLP
        cls = DuelingMLP if config["model"].get("dueling", False) else MLP
        return cls(
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


def state_to_tensor(state: Any, model_type: str) -> torch.Tensor:
    if model_type == "cnn":
        t = torch.tensor(state, dtype=torch.float32)
        return t.permute(2, 0, 1).unsqueeze(0)   # (H,W,3) → (1,3,H,W)
    else:
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)


def train(env: BaseEnv, config: dict[str, Any], visualizer: Any = None) -> nn.Module:
    model = build_model(config)
    agent = DQNAgent(model, config)

    episodes = config["training"]["episodes"]
    solve_threshold = config["training"]["solve_threshold"]
    save_every = config["checkpoints"]["save_every"]
    checkpoint_dir = config["checkpoints"]["dir"]
    model_type = config["model"]["type"]
    env_name = config["env"]["name"]

    os.makedirs(checkpoint_dir, exist_ok=True)

    train_every = config["training"].get("train_every", 1)
    solve_window = config["training"].get("solve_window", 50)
    # Best-checkpoint selection metric. Default "reward" (generic). Set "success" for
    # tasks where average reward is a misleading scorecard — e.g. a policy that lingers in
    # a good state can out-score one that completes the task — in which case the env must
    # expose info["is_success"] (bool) and _best.pt tracks the rolling success rate,
    # tie-broken by average reward. Any OTHER value names a numeric info key (e.g.
    # "coverage_score"): _best.pt then tracks that key's rolling mean, tie-broken by
    # average reward. The loop never learns what the number means — it just maximizes
    # whatever the environment reports, which is how it stays environment-agnostic.
    best_metric = config["checkpoints"].get("best_metric", "reward")
    recent_rewards: deque[float] = deque(maxlen=solve_window)
    recent_success: deque[bool] = deque(maxlen=solve_window)
    recent_metric: deque[float] = deque(maxlen=solve_window)
    best_score: tuple[float, float] = (float("-inf"), float("-inf"))
    step_count = 0

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        info: dict[str, Any] = {}

        while not done:
            state_tensor = state_to_tensor(state, model_type)
            action = agent.select_action(state_tensor)
            next_state, reward, done, truncated, info = env.step(action)

            # Store TERMINATED only. A timeout (truncated) is not a real terminal state —
            # the episode was cut by the clock, not by the MDP — so its target must still
            # bootstrap gamma*Q(s'). Collapsing `done|truncated` into the stored flag
            # zeroes that bootstrap (target = r), which puts a systematic downward bias on
            # exactly the long-horizon states the agent needs to value (Pardo et al. 2018).
            agent.remember(state, action, reward, next_state, done)
            done = done or truncated  # loop control only
            step_count += 1
            if step_count % train_every == 0:
                agent.train()

            state = next_state
            total_reward += reward

        agent.end_episode()   # flush the n-step buffer; no-op for 1-step
        agent.decay_epsilon()
        recent_rewards.append(total_reward)
        recent_success.append(bool(info.get("is_success", False)))
        if best_metric not in ("reward", "success"):
            recent_metric.append(float(info.get(best_metric, 0.0)))
        avg_reward = sum(recent_rewards) / len(recent_rewards)

        if visualizer:
            visualizer.end_episode(total_reward, agent.epsilon)

        # Surface a curriculum difficulty if the env exposes one (generic info field; the
        # loop stays env-agnostic — it just prints whatever number is provided).
        diff_str = f" | difficulty: {float(info['difficulty']):.2f}" if "difficulty" in info else ""
        # Metrics may be counts (1e5) or fractions (0.35) — pick a format that shows both.
        metric_str = ""
        if best_metric not in ("reward", "success") and best_metric in info:
            m = float(info[best_metric])
            mean_m = sum(recent_metric) / len(recent_metric) if recent_metric else 0.0
            fmt = "7.0f" if abs(m) >= 100.0 else "7.3f"
            metric_str = f" | {best_metric}: {m:{fmt}} (avg {mean_m:{fmt}})"
        print(
            f"Episode {episode:4d} | reward: {total_reward:6.1f} | avg: {avg_reward:6.1f} | epsilon: {agent.epsilon:.3f}{diff_str}{metric_str}")

        if episode % save_every == 0:
            path = os.path.join(checkpoint_dir, f"{env_name}_{episode}.pt")
            torch.save(model.state_dict(), path)

        success_rate = sum(recent_success) / len(recent_success) if recent_success else 0.0
        score: tuple[float, float]
        if best_metric == "success":
            # Only select once the window is FULL. Over a partial window an early lucky
            # streak spikes the rate to 1.0 and permanently locks _best.pt to a nearly
            # untrained checkpoint that no later full-window rate can beat. Mirrors the
            # full-window gate on solve detection below.
            eligible = len(recent_success) == solve_window
            score = (success_rate, avg_reward)
        elif best_metric != "reward":
            # Custom numeric info key. Same full-window gate, same reason.
            eligible = len(recent_metric) == solve_window
            metric_mean = sum(recent_metric) / len(recent_metric) if recent_metric else 0.0
            score = (metric_mean, avg_reward)
        else:
            eligible = True
            score = (avg_reward, 0.0)
        if config["checkpoints"]["keep_best"] and eligible and score > best_score:
            best_score = score
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, f"{env_name}_best.pt"))

        if len(recent_rewards) == solve_window and avg_reward >= solve_threshold:
            print(
                f"Solved at episode {episode} with avg reward {avg_reward:.1f}")
            break

    env.close()
    return model
