---
paths:
  - "envs/*.py"
  - "envs/**/*.py"
---

# Environment Rules

## Render mode passthrough
All environment wrappers that call `gym.make()` must pass `config["env"].get("render_mode")` to enable evaluation rendering. It is `None` during training, so there is no overhead, and `"human"` during eval. If you create a new `BaseEnv` subclass that wraps a gymnasium env, include this line:

```python
render_mode = config["env"].get("render_mode")
self.env = gym.make("EnvName", render_mode=render_mode)
```

## Reward functions belong in the environment
Never compute or shape rewards in the training loop. All reward logic goes in the environment's `step()` method. This keeps the training loop environment-agnostic.

## Crashes must be penalized in the reward
Every environment's `_get_reward()` must return a negative reward, typically -1.0, when the agent has crashed: a ground collision, a boundary violation, excessive tilt. Without this, crash transitions enter the replay buffer with small positive rewards, and the agent has no gradient to learn "don't crash." This is especially damaging in early training when most episodes are random exploration ending in crashes — the buffer fills with indistinguishable low-positive crash transitions and the agent never learns basic stability.

## Subclassing BaseEnv
Every new environment must subclass `BaseEnv` from `envs/base_env.py`. Implement `reset()`, `step()`, and `get_state()`. Use the `env.wrapper` config key to select custom subclasses without modifying `main.py`.

## Prefer environments with built-in rendering
When an environment is not available in standard Gymnasium, look for a third-party Gymnasium-compatible package — PyFlyt for drones, gym-pybullet-drones, the MuJoCo envs — before building custom physics. Wrap the third-party env in a `BaseEnv` subclass to handle reward shaping or action-space discretization, but let the underlying simulator handle rendering via the `render_mode` passthrough. Only build custom physics and rendering when no suitable simulator exists for the domain.

## The info dict is the environment's reporting channel

The training loop is environment-agnostic, so anything it should track must arrive through
`info` from `step()`. Three generic keys are understood, and the loop never learns what they
mean:

- `info["is_success"]`, a bool — set it when the episode achieved the task, for
  `checkpoints.best_metric: "success"`.
- any numeric key such as `info["coverage_score"]` — name it in `checkpoints.best_metric` and
  `_best.pt` tracks that key's rolling mean instead of average reward. Use this whenever
  average reward is a misleading scorecard.
- `info["difficulty"]`, a float — a curriculum's current setting, printed per episode.

Curricula and rehearsal live here, in the environment, never in the loop.

## Reproducible episodes

`BaseEnv` reads `env.seed` from config and exposes `self.rng` plus `_begin_episode()`. Call
`_begin_episode()` at the top of `reset()` and draw **every** per-episode randomization from
`self.rng`, so a seeded run is repeatable and two policies can be compared on identical
episodes. Selecting a checkpoint and reporting it on the same seed is winner's curse — see
`.claude/rules/evaluation.md`.

## Reward design has its own rule

`.claude/rules/reward-design.md` covers what to count, one-time versus loiterable payouts,
potential-based shaping, and the per-step arithmetic check. Read it before writing or changing
a `_get_reward()`.
