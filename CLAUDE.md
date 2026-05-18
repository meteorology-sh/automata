# CLAUDE.md — RL Harness

## Purpose

This is a generic reinforcement learning harness. It provides reusable infrastructure for training DQN agents in any Gymnasium-compatible environment. Users extend it by subclassing `BaseEnv` and providing a config.

---

## Rules

These must be followed at all times:

- Never hardcode environment names, model sizes, or hyperparameters. All configuration lives in `configs/`.
- All environments must subclass `BaseEnv` in `envs/base_env.py`. Never interact with a simulator directly from the training loop.
- Never apply softmax to Q-value outputs. The decision model outputs raw logits.
- The training loop in `core/train.py` must remain environment-agnostic. No drone, vacuum, or CartPole-specific logic belongs there.
- Always save checkpoints to `checkpoints/` using the pattern `{env_name}_{episode}.pt`. Never overwrite the best checkpoint.
- Reward functions belong in the environment subclass, not in the training loop.
- All models must accept a `num_actions` argument. Never hardcode action counts.

---

## Architecture

```
core/
  agent.py            ← DQN agent (epsilon-greedy, replay, soft target updates)
  memory.py           ← replay buffer
  train.py            ← training loop (environment-agnostic)
  visualizer.py       ← real-time reward dashboard

envs/
  base_env.py         ← abstract base class + GymEnv wrapper
  mujoco_env.py       ← base class for MuJoCo-backed environments
  quadrotor_hover.py  ← hover at target altitude
  quadrotor_follow.py ← follow a moving target

models/
  mlp.py              ← MLP for vector-based states
  cnn.py              ← CNN for image-based states
  mjcf/               ← MuJoCo robot definitions (MJCF XML)
    quadrotor.xml

configs/
  default.yaml            ← LunarLander-v3 (shipped default)
  quadrotor_hover.yaml
  quadrotor_follow.yaml

tests/
  test_smoke.py               ← framework tests (agent, buffer, models)
  test_quadrotor_hover.py     ← hover env tests
  test_quadrotor_follow.py    ← follow env tests

main.py               ← training entry point
eval.py               ← evaluate a trained checkpoint
checkpoints/           ← saved model weights (local)
```

---

## CLI

```bash
# Training (headless by default)
python main.py --config configs/default.yaml
python main.py --config configs/default.yaml --vis    # enable reward dashboard

# Evaluation (renders by default)
python eval.py --config configs/quadrotor_hover.yaml --checkpoint checkpoints/QuadrotorHover_best.pt
python eval.py --config configs/default.yaml --checkpoint checkpoints/LunarLander-v3_best.pt --no-render
```

---

## Skills

### Adding a new environment

For Gymnasium environments, use `GymEnv` wrapper. For custom physics, subclass `MujocoEnv`:

1. Create `envs/your_env.py`, subclass `MujocoEnv` (or `BaseEnv` for non-MuJoCo)
2. Implement `_get_obs()`, `_get_reward()`, `_is_done()`, `_apply_action()`
3. Override `_reset_state()` for initial randomization
4. Override `_configure_camera()` to set the viewer viewport
5. Add a config in `configs/your_env.yaml` with `wrapper: "envs.your_env.YourEnv"`
6. Create `tests/test_your_env.py` (never add env tests to `test_smoke.py`)

```python
from envs.mujoco_env import MujocoEnv

class YourEnv(MujocoEnv):
    def _apply_action(self, action: int):
        # map discrete action to self.data.ctrl

    def _get_obs(self) -> np.ndarray:
        # return observation vector

    def _get_reward(self) -> float:
        if self._is_crashed():
            return -1.0  # crash penalty is mandatory
        # compute reward components
        return float(np.clip(reward, -1.0, 1.0))

    def _is_done(self) -> bool:
        return self._is_crashed()

    def _reset_state(self):
        # randomize initial qpos/qvel

    def _configure_camera(self):
        # set cam.lookat, cam.distance, cam.elevation from env params
```

### Choosing a model

- Use `models/mlp.py` when the state is a numerical vector (e.g. position, velocity)
- Use `models/cnn.py` when the state is an image (e.g. camera frame)
- Both accept `(state_size_or_input_shape, num_actions)` and output raw Q-values

### Reward function design

Design rewards inside `_get_reward()`. Follow these principles:

- Always return -1.0 on crash — without this, the replay buffer fills with indistinguishable low-positive crash transitions and the agent never learns stability
- Reward the actual goal, not a proxy for it
- No single reward component should dominate
- Think adversarially: how would an optimizer game this function?
- Clip total reward to [-1, 1] for training stability
- Only reward progress when preconditions are safe (e.g. only reward forward velocity below obstacle proximity threshold)

### Hyperparameters

All hyperparameters are read from config YAML files. Canonical baseline:

```yaml
epsilon:
  start: 0.9
  min: 0.01
  decay: 0.995
training:
  gamma: 0.99
  batch_size: 128
  lr: 0.0001
  memory_size: 10000    # scale with max_episode_steps
  train_every: 1
  tau: 0.005
  solve_window: 50
  solve_threshold: 200  # set 10-15% below expected peak
```

Override per-environment by creating `configs/your_env.yaml`. Key tuning rules:
- Scale `memory_size` with `max_episode_steps` — long episodes need larger buffers to resist the death spiral (short crash episodes overwrite good data)
- Set `solve_threshold` conservatively below expected peak — if unreachable, training continues past the cliff
- Epsilon should reach `epsilon_min` at ~80% of the episode budget
- See `.claude/rules/hyperparameters.md` for detailed interaction rules

### Stopping criteria

The training loop stops when average reward over the last `solve_window` episodes (default 50) exceeds `solve_threshold`, or when the episode budget is exhausted. Always checkpoint the best-performing model.

---

## Visualizer

The visualizer is a single-chart dashboard that runs in a background thread during training. It displays:

- Episode reward over time (rolling chart with solve threshold line)
- Rolling average matching the `solve_window` (so the chart predicts when training stops)
- Current epsilon as a decimal value in the title

The training loop calls `visualizer.end_episode(total_reward, epsilon)` once per episode. The visualizer must never add per-step computation to the training loop.

---

## DQN Training Loop (reference)

```
for each episode:
    state = env.reset()
    while not done:
        action = agent.select_action(state, epsilon)
        next_state, reward, done, truncated, info = env.step(action)
        memory.append((state, action, reward, next_state, done or truncated))
        agent.train(memory)
        state = next_state
    decay epsilon
    checkpoint if best reward
```

---

## Maintenance

When adding new files, environments, configs, or concepts to the project, update this file to reflect them. Specifically:
- New files or directories → update the Architecture tree
- New environments → add to the tree and ensure the Skills section covers any new patterns
- New CLI flags → update the CLI section
- New rules or lessons learned → add to `.claude/rules/` and summarize here if they affect how environments are built or trained

CLAUDE.md is loaded at the start of every session. If it is stale, the assistant wastes tokens re-exploring the codebase. Keep it current.

---

## Key Concepts (for context)

- **Q-value**: expected future reward for taking an action in a state
- **Bellman update**: `Q(s,a) = reward + gamma * max(Q(s',a'))`
- **Replay buffer**: stores past experiences, sampled randomly to break temporal correlation
- **Epsilon-greedy**: random action with probability epsilon, greedy otherwise
- **Sim-to-real gap**: policies trained in simulation may not transfer cleanly to physical hardware — address with domain randomization and noise injection
