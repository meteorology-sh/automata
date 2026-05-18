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

Do not change `lr`, `tau`, loss function, or gradient clipping without a specific reason. Adjust `epsilon_decay`, `solve_threshold`, and `memory_size` per environment. Scale `memory_size` with `max_episode_steps` — the baseline 10,000 assumes short episodes (~500 steps).

## Learning rate and train_every are coupled
If using `train_every > 1` for speed, reduce `lr` proportionally. `train_every: 4` with `lr: 0.001` is 4x too aggressive — each of the fewer updates overshoots. The original DQN paper used `train_every: 4` with `lr: 0.00025`. The product `lr * (steps_per_episode / train_every)` should stay roughly constant.

## Epsilon min is a tradeoff between noise floor and recovery
Too high: random actions tank episode scores. With `epsilon_min: 0.05` and 500-step episodes, 25 random actions add enough noise to prevent the rolling average from reaching a high threshold. Rule of thumb: if `epsilon_min * max_episode_steps` exceeds ~5, the noise floor may suppress convergence.

Too low: the agent can't recover from policy collapse. With `epsilon_min: 0.01` and long episodes (1000+ steps), a degraded policy crashes early, short crash episodes fill the replay buffer orders of magnitude faster than long successful ones, and the agent enters a death spiral — the buffer fills with crash data, which trains a crashing policy, which generates more crash data. Low epsilon provides no exploration to escape.

For long episodes, compensate with a larger replay buffer (`memory_size`) so good experiences survive temporary collapses. Scale `memory_size` proportionally to `max_episode_steps` — a 1500-step environment needs at least 4x the buffer of a 500-step one.

## Solve window should match the environment's timescale
A 100-episode window takes 100+ episodes past convergence to reflect that the agent has solved the environment. Default to 50 episodes — statistically meaningful but responsive. Harder environments with high variance may need a larger window, but never so large that training runs for hundreds of episodes past convergence.

## Solve threshold must be reachable
Set `solve_threshold` conservatively below the expected peak performance. If the threshold is higher than what the agent can sustain, training continues past the peak and the policy degrades — potentially triggering a death spiral. The best checkpoint is saved regardless, but wasted episodes after the peak burn compute and can corrupt the replay buffer. When in doubt, set the threshold 10-15% below expected peak and rely on `keep_best: true` to capture the actual best model.

## Epsilon decay timing
Epsilon should reach `epsilon_min` at roughly 80% of the episode budget. For 1000 episodes with `epsilon_decay: 0.995`: `0.9 * 0.995^800 = 0.016`. Verify this for each new config.
