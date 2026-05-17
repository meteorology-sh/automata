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

## Subclassing BaseEnv
Every new environment must subclass `BaseEnv` from `envs/base_env.py`. Implement `reset()`, `step()`, and `get_state()`. Use the `env.wrapper` config key to select custom subclasses without modifying `main.py`.
