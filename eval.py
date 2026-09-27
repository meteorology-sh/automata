"""Evaluate a trained checkpoint — and, in the same run, the blind baselines it has to beat.

A learned policy is only interesting relative to what a policy that ignores its
observations can score on the same episodes. The baselines here are deliberately blind:

  random    — a fresh uniform-random action every step
  hold      — random, but each action is held for `--hold` steps (the same temporal
              extension exploration uses). On any task where the reward depends on where
              the agent goes, this is the real floor: coherent random legs score far above
              per-step dithering, so a policy that beats `random` but not `hold` has
              learned nothing.
  straight  — one constant action for the whole episode, redrawn each episode. This is the
              collapse detector: a degenerate policy can post a respectable score without
              using its observations at all.

Every run also reports `Hact`: action entropy computed *within* an episode and then
averaged over episodes, normalized so 1.0 = uses the whole action set and 0.0 = one action
always. Pool across episodes instead and a policy that picks a different constant action
each episode looks varied — exactly the collapse the number exists to catch. A policy with
Hact near zero is not making decisions, however good its reward looks.
"""

import argparse
import math
import random
from collections import Counter
from typing import Any

import torch

from core.train import load_config, build_model, state_to_tensor
from main import build_env

POLICIES = ("checkpoint", "random", "hold", "straight")


class _Policy:
    """Action source with a per-episode reset. Blind baselines ignore the state."""

    def __init__(self, kind: str, num_actions: int, model: Any = None,
                 model_type: str = "mlp", hold: int = 15):
        self.kind = kind
        self.num_actions = num_actions
        self.model = model
        self.model_type = model_type
        self.hold = max(1, hold)
        self._held = 0
        self._steps_left = 0

    def reset(self) -> None:
        self._held = random.randrange(self.num_actions)
        self._steps_left = 0

    def act(self, state: Any) -> int:
        if self.kind == "checkpoint":
            with torch.no_grad():
                return int(self.model(state_to_tensor(state, self.model_type)).argmax().item())
        if self.kind == "random":
            return random.randrange(self.num_actions)
        if self.kind == "straight":
            return self._held           # one constant action all episode
        if self._steps_left <= 0:       # "hold"
            self._held = random.randrange(self.num_actions)
            self._steps_left = self.hold
        self._steps_left -= 1
        return self._held


def _action_entropy(counts: Counter, num_actions: int) -> float:
    """Normalized Shannon entropy of one episode's action histogram (0 = collapsed)."""
    total = sum(counts.values())
    if total == 0 or num_actions < 2:
        return 0.0
    h = -sum((n / total) * math.log(n / total) for n in counts.values() if n)
    return h / math.log(num_actions)


def _run_policy(env, policy: _Policy, num_episodes: int,
                metric: str | None = None) -> dict[str, Any]:
    rewards: list[float] = []
    entropies: list[float] = []
    metrics: list[float] = []

    for ep in range(num_episodes):
        state = env.reset()
        policy.reset()
        done = False
        total_reward = 0.0
        counts: Counter = Counter()
        info: dict[str, Any] = {}

        while not done:
            action = policy.act(state)
            counts[action] += 1
            state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            total_reward += reward

        rewards.append(total_reward)
        entropies.append(_action_entropy(counts, policy.num_actions))
        if metric and metric in info:
            metrics.append(float(info[metric]))
        print(f"  [{policy.kind}] episode {ep + 1:3d} | reward: {total_reward:8.1f}")

    return {
        "policy": policy.kind,
        "rewards": rewards,
        "avg": sum(rewards) / len(rewards),
        "min": min(rewards),
        "max": max(rewards),
        "hact": sum(entropies) / len(entropies),
        "metric": sum(metrics) / len(metrics) if metrics else None,
    }


def evaluate(env, model, config: dict, num_episodes: int,
             policy: str = "checkpoint", hold: int = 15,
             metric: str | None = None) -> list[float]:
    """Run one policy for num_episodes. Returns the list of episode rewards."""
    p = _Policy(policy, config["model"]["num_actions"], model=model,
                model_type=config["model"]["type"], hold=hold)
    stats = _run_policy(env, p, num_episodes, metric=metric)
    print(f"\n{'=' * 40}")
    print(f"Policy: {policy}   Episodes: {num_episodes}")
    print(f"Average reward: {stats['avg']:.1f}")
    print(f"Min: {stats['min']:.1f}  Max: {stats['max']:.1f}")
    print(f"Action entropy (within-episode, averaged): {stats['hact']:.2f}")
    if stats["metric"] is not None:
        print(f"{metric}: {stats['metric']:.4g}")
    env.close()
    return stats["rewards"]


def main():
    parser = argparse.ArgumentParser(
        description="RL Harness — score a checkpoint against the blind baselines")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML (same one used for training)")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to .pt checkpoint (required for the 'checkpoint' policy)")
    parser.add_argument("--policy", nargs="+", default=["checkpoint"], choices=POLICIES,
                        help="Policies to score, in one table (default: checkpoint)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Episodes per policy (default: 10)")
    parser.add_argument("--seed", type=int,
                        help="Fix the episode sequence so policies face identical episodes. "
                             "Select on one seed, confirm on a disjoint one.")
    parser.add_argument("--hold", type=int, default=15,
                        help="Steps to hold each action in the 'hold' baseline (default: 15)")
    parser.add_argument("--metric", type=str,
                        help="Numeric info key to report per policy (e.g. the task's own "
                             "scorecard metric, not reward)")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable visual rendering")
    args = parser.parse_args()

    config = load_config(args.config)
    if not args.no_render:
        config["env"]["render_mode"] = "human"
    if args.seed is not None:
        config["env"]["seed"] = args.seed

    model = None
    if "checkpoint" in args.policy:
        if not args.checkpoint:
            parser.error("--checkpoint is required when scoring the 'checkpoint' policy")
        model = build_model(config)
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
        model.eval()

    rows = []
    for kind in args.policy:
        if args.seed is not None:
            random.seed(args.seed)      # same baseline draws for every policy
        env = build_env(config)         # rebuild so each policy faces episode 0 onward
        p = _Policy(kind, config["model"]["num_actions"], model=model,
                    model_type=config["model"]["type"], hold=args.hold)
        rows.append(_run_policy(env, p, args.episodes, metric=args.metric))
        env.close()

    seed_str = "unseeded" if args.seed is None else f"seed {args.seed}"
    print(f"\n{'=' * 64}")
    print(f"Scorecard — {args.episodes} episodes per policy, greedy, {seed_str}")
    header = f"{'policy':<12}{'avg reward':>12}{'min':>10}{'max':>10}{'Hact':>8}"
    if args.metric:
        header += f"{args.metric:>16}"
    print(header)
    for r in rows:
        line = (f"{r['policy']:<12}{r['avg']:>12.1f}{r['min']:>10.1f}"
                f"{r['max']:>10.1f}{r['hact']:>8.2f}")
        if args.metric:
            line += f"{(r['metric'] if r['metric'] is not None else float('nan')):>16.4g}"
        print(line)
    if any(r["policy"] == "checkpoint" and r["hact"] < 0.05 for r in rows):
        print("\nWARNING: the checkpoint's within-episode action entropy is ~0 — it is "
              "playing one action and ignoring its observations. Do not ship it.")


if __name__ == "__main__":
    main()
