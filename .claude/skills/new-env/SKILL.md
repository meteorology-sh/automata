---
description: Scaffold a new environment config with canonical DQN hyperparameters. Use when adding a new Gymnasium environment to the harness.
argument-hint: <env-name> <state-size> <num-actions>
arguments: [env_name, state_size, num_actions]
---

Create a new config file at `configs/$env_name.yaml` for the Gymnasium environment.

Use the canonical DQN hyperparameters as the baseline. Before writing the config:

1. Verify the environment exists: `python -c "import gymnasium as gym; env = gym.make('$env_name'); print('State:', env.observation_space); print('Actions:', env.action_space); env.close()"`
2. Confirm the state size ($state_size) and action count ($num_actions) match the environment
3. Look up a reasonable solve threshold for this environment

Write the config with these canonical values:

```yaml
env:
  name: "$env_name"
  image_size: 224
  max_episode_steps: 500
model:
  type: "mlp"
  hidden_size: 128
  num_actions: $num_actions
  state_size: $state_size
training:
  episodes: 1000
  batch_size: 128
  lr: 0.0001
  gamma: 0.99
  memory_size: 10000
  train_every: 1
  tau: 0.005
  solve_window: 50
  solve_threshold: <look up appropriate value>
epsilon:
  start: 0.9
  min: 0.01
  decay: 0.995
checkpoints:
  dir: "checkpoints/"
  save_every: 100
  keep_best: true
visualizer:
  enabled: true
```

Then verify epsilon_min is compatible with the solve threshold: if `epsilon_min * max_episode_steps > 5`, warn that random exploration noise may prevent solving.

Verify epsilon reaches min at ~80% of episode budget: `epsilon_start * epsilon_decay^(episodes * 0.8)` should be near `epsilon_min`.
