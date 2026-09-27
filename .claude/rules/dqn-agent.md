---
paths:
  - "core/agent.py"
---

# DQN Agent Rules

## Loss function
Always use `nn.SmoothL1Loss()`, the Huber loss, for DQN. Never MSE. MSE squares large TD errors, producing gradient explosions that cause reward crashes. Huber loss is linear for large errors, quadratic for small ones — stable without sacrificing gradient signal.

## Gradient clipping
Use `nn.utils.clip_grad_value_(params, clip_value=100)` to clip individual gradient elements. Never use `clip_grad_norm_` with a small `max_norm` — for a network with thousands of parameters, the natural gradient norm is much larger than 1.0, so `clip_grad_norm_(max_norm=1.0)` silently scales every update to a fraction of the configured learning rate. Q-values for different actions will never differentiate.

## Target network
Use Polyak averaging — a soft update — with `tau` from config, 0.005 by default. Every training step, blend online weights into the target: `target = (1 - tau) * target + tau * online`. Never use hard copies that copy all weights every N steps — hard copies create discontinuities in target Q-values that destabilize learning. Soft updates are also simpler to tune since they don't interact with `train_every`.

## Numpy-to-tensor conversion
When converting a list of numpy arrays to a tensor, always stack with `np.array()` first: `torch.tensor(np.array(states), dtype=torch.float32)`. Passing a list of numpy arrays directly to `torch.tensor()` is extremely slow and produces a warning.
