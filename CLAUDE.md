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
  agent.py        ← epsilon-greedy action selection
  memory.py       ← replay buffer
  train.py        ← training loop (environment-agnostic)
  visualizer.py   ← real-time dashboard

envs/
  base_env.py     ← abstract base class, all envs subclass this

models/
  cnn.py          ← vision model for image-based states
  mlp.py          ← MLP for vector-based states

configs/
  default.yaml    ← all hyperparameters live here

main.py           ← training entry point
eval.py           ← evaluate a trained checkpoint with rendering
```

---

## Skills

### Adding a new environment

1. Create `envs/your_env.py`
2. Subclass `BaseEnv`
3. Implement `reset()`, `step()`, and `get_state()`
4. Define the reward function inside `step()`
5. Add a config entry in `configs/`

```python
from envs.base_env import BaseEnv

class YourEnv(BaseEnv):
    def reset(self):
        # return initial state

    def step(self, action):
        # apply action
        # compute reward here
        # return (next_state, reward, done, truncated, info)

    def get_state(self):
        # return current state as numpy array or image
```

### Choosing a model

- Use `models/mlp.py` when the state is a numerical vector (e.g. position, velocity)
- Use `models/cnn.py` when the state is an image (e.g. camera frame)
- Both accept `(state_size_or_input_shape, num_actions)` and output raw Q-values

### Reward function design

Design rewards inside `step()`. Follow these principles:

- Reward the actual goal, not a proxy for it
- No single reward component should dominate
- Think adversarially: how would an optimizer game this function?
- Clip total reward to [-1, 1] for training stability
- Only reward progress when preconditions are safe (e.g. only reward forward velocity below obstacle proximity threshold)

### Hyperparameters

All hyperparameters are read from `configs/default.yaml`:

```yaml
epsilon_start: 0.9
epsilon_min: 0.01
epsilon_decay: 0.995
gamma: 0.99
batch_size: 128
lr: 0.0001
memory_size: 10000
train_every: 1
tau: 0.005
solve_window: 50
episodes: 1000
image_size: 224
```

Override per-project by creating `configs/your_env.yaml`. See `SKILLS.md` for lessons on how these values interact.

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

## Key Concepts (for context)

- **Q-value**: expected future reward for taking an action in a state
- **Bellman update**: `Q(s,a) = reward + gamma * max(Q(s',a'))`
- **Replay buffer**: stores past experiences, sampled randomly to break temporal correlation
- **Epsilon-greedy**: random action with probability epsilon, greedy otherwise
- **Sim-to-real gap**: policies trained in simulation may not transfer cleanly to physical hardware — address with domain randomization and noise injection
