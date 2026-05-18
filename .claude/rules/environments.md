---
paths:
  - "envs/*.py"
  - "envs/**/*.py"
---

# Environment Rules

## Render mode passthrough
All environment wrappers that call `gym.make()` must pass `config["env"].get("render_mode")` to enable evaluation rendering. This is `None` during training (no overhead) and `"human"` during eval. If you create a new `BaseEnv` subclass that wraps a gymnasium env, include this line:

```python
render_mode = config["env"].get("render_mode")
self.env = gym.make("EnvName", render_mode=render_mode)
```

## Reward functions belong in the environment
Never compute or shape rewards in the training loop. All reward logic goes in the environment's `step()` method. This keeps the training loop environment-agnostic.

## Crashes must be penalized in the reward
Every environment's `_get_reward()` must return a negative reward (typically -1.0) when the agent has crashed (ground collision, boundary violation, excessive tilt, etc.). Without this, crash transitions enter the replay buffer with small positive rewards, and the agent has no gradient to learn "don't crash." This is especially damaging in early training when most episodes are random exploration ending in crashes — the buffer fills with indistinguishable low-positive crash transitions and the agent never learns basic stability.

## Subclassing BaseEnv
Every new environment must subclass `BaseEnv` from `envs/base_env.py`. Implement `reset()`, `step()`, and `get_state()`. Use the `env.wrapper` config key to select custom subclasses without modifying `main.py`.

## Prefer environments with built-in rendering
When an environment is not available in standard Gymnasium, look for a third-party Gymnasium-compatible package (e.g. PyFlyt for drones, gym-pybullet-drones, MuJoCo envs) before building custom physics. Wrap the third-party env in a `BaseEnv` subclass to handle reward shaping or action-space discretization, but let the underlying simulator handle rendering via the `render_mode` passthrough. Only build custom physics and rendering when no suitable simulator exists for the domain.
