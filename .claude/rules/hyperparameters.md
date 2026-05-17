---
paths:
  - "configs/*.yaml"
  - "configs/**/*.yaml"
---

# Hyperparameter Rules

## Canonical baseline
Start from these values for any new environment. They are validated against the PyTorch DQN tutorial and solve CartPole reliably:

```yaml
training:
  batch_size: 128
  lr: 0.0001
  gamma: 0.99
  memory_size: 10000
  train_every: 1
  tau: 0.005
  solve_window: 50
epsilon:
  start: 0.9
  min: 0.01
  decay: 0.995
```

Do not change `lr`, `tau`, loss function, or gradient clipping without a specific reason. Adjust `epsilon_decay` and `solve_threshold` per environment.

## Learning rate and train_every are coupled
If using `train_every > 1` for speed, reduce `lr` proportionally. `train_every: 4` with `lr: 0.001` is 4x too aggressive — each of the fewer updates overshoots. The original DQN paper used `train_every: 4` with `lr: 0.00025`. The product `lr * (steps_per_episode / train_every)` should stay roughly constant.

## Epsilon min must be compatible with the solve threshold
With `epsilon_min: 0.05`, 5% of actions are random. Over a 500-step episode, that's ~25 random actions — enough to frequently kill episodes and prevent the rolling average from reaching a high threshold. Calculate: if `epsilon_min * max_episode_steps` exceeds ~5, the noise floor may be too high. Lower `epsilon_min` or lower `solve_threshold`.

## Solve window should match the environment's timescale
A 100-episode window takes 100+ episodes past convergence to reflect that the agent has solved the environment. Default to 50 episodes — statistically meaningful but responsive. Harder environments with high variance may need a larger window, but never so large that training runs for hundreds of episodes past convergence.

## Epsilon decay timing
Epsilon should reach `epsilon_min` at roughly 80% of the episode budget. For 1000 episodes with `epsilon_decay: 0.995`: `0.9 * 0.995^800 = 0.016`. Verify this for each new config.
