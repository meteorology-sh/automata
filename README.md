# Automata

This is an experimental workspace for developing a harness for reinforcement learning in autonomous robotics applications.

## About

Automata is a generic reinforcement learning harness for training DQN agents in any Gymnasium-compatible environment. Provides reusable training infrastructure, model architectures, and a real-time visualization dashboard.

## Quick Start

```bash
pip install -r requirements.txt
python main.py --config configs/default.yaml
```

This trains a DQN agent on CartPole-v1 with a live dashboard. Pass `--no-vis` for headless training.

## Real-Time Dashboard

The visualizer is a 2x2 matplotlib dashboard that runs in a background thread during training, updating every 100ms without blocking the training loop.

| Panel               | What it shows                                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Reward per step** | Rolling line chart of step rewards — shows whether the agent is improving                                             |
| **Epsilon**         | Exploration rate as it decays from 1.0 toward the minimum — shows the explore/exploit transition                      |
| **Q-values**        | Bar chart of Q-values for each action at the current state; the greedy choice is highlighted red                      |
| **Agent view**      | Adapts to state type: raw image frame for vision-based envs, or a bar chart of state dimensions for vector-based envs |

The dashboard is thread-safe and runs as a daemon thread, so it shuts down automatically when training ends. On Windows, if the window doesn't render, you may need to set the matplotlib backend to `TkAgg`.

## Evaluating a Trained Agent

After training, use `eval.py` to load a checkpoint and watch the agent perform:

```bash
python eval.py --config configs/default.yaml --checkpoint checkpoints/CartPole-v1_best.pt
```

This opens a rendering window and runs 10 episodes with greedy action selection (no exploration noise). Pass `--episodes N` to change the count, or `--no-render` for headless evaluation with reward stats only.

## The Pipeline

The harness is designed around a simulation-to-hardware pipeline:

```
Define env --> Train policy --> Evaluate in sim --> Export model --> Deploy to hardware
```

Today, the first three stages are supported. The trained `.pt` checkpoint is a set of neural network weights — the "brain" that turns sensor readings into actions. See [`PLAN.md`](PLAN.md) for the full roadmap toward model export and hardware deployment.

## Documentation

- [`CLAUDE.md`](CLAUDE.md) — Architecture rules, project conventions, and how to extend the harness (add environments, choose models, design reward functions)
- [`docs/README.md`](docs/README.md) — Detailed reference on DQN concepts, state representation, model selection, reward shaping, and sim-to-real considerations
- [`PLAN.md`](PLAN.md) — Pipeline roadmap: from simulation to hardware deployment
