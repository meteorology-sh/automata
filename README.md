# Automata

Automata is a generic reinforcement learning harness for training DQN agents on simulated robots. Define a robot in MJCF XML, write a reward function, point it at a config, and train. The harness handles the training loop, replay buffer, epsilon scheduling, checkpointing, and visualization.

MuJoCo provides the physics and 3D rendering, while the harness runs training and policy definitions.

## Quick Start

Automata ships with a default scenario in the form of [Gymnasium's Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/). Follow these instructions to install dependencies, train the model, and evaluate its performance.

First, install [Python 3.12](https://www.python.org/downloads/release/python-31213/)

Then, in the root directory, instantiate a virtual environment:

```bash
python3.12 -m venv venv
source ./venv/bin/activate
```

Install dependencies, train the default agent, and evaluate its performance:

```bash
pip install -r requirements.txt

# Train a DQN agent on LunarLander
python main.py --config configs/default.yaml

# Evaluate the trained agent with rendering
python eval.py --config configs/default.yaml --checkpoint checkpoints/LunarLander-v3_best.pt
```

Pass `--no-vis` for headless training, `--no-render` for headless evaluation.

## How It Works

A trained policy is produced from three independent inputs:

| Input                  | Defines                                                                | Example                        |
| ---------------------- | ---------------------------------------------------------------------- | ------------------------------ |
| **MJCF XML**           | What the robot _is_ — geometry, mass, joints, actuators                | `models/mjcf/quadrotor.xml`    |
| **MujocoEnv subclass** | What it _should learn_ — reward function, observations, termination    | `envs/quadrotor_hover.py`      |
| **Config YAML**        | How to _train_ — hyperparameters, model architecture, epsilon schedule | `configs/quadrotor_hover.yaml` |

The same robot can back different tasks (hover, land, track waypoints) by swapping the env subclass. The training loop never changes.

## Shipped Environment

The default config trains on **LunarLander-v3** — land a spacecraft between the flags using discrete thrust controls. Solved when average reward exceeds 200 over 50 episodes.

For custom robots, subclass `MujocoEnv` and provide an MJCF XML file (see [Adding a New Robot](#adding-a-new-robot)).

## Adding a New Robot

1. Write an MJCF XML file describing the robot (`models/mjcf/your_robot.xml`)
2. Subclass `MujocoEnv`
3. Implement `_get_obs()`, `_get_reward()`, `_is_done()`, `_apply_action()`
4. Create a config in `configs/`
5. Train: `python main.py --config configs/your_robot.yaml`

See [`CLAUDE.md`](CLAUDE.md) for full conventions and reward design principles.

## Project Structure

```
core/
  agent.py        -- DQN agent (epsilon-greedy, replay, soft target updates)
  memory.py       -- replay buffer
  train.py        -- training loop (environment-agnostic)
  visualizer.py   -- real-time reward dashboard

envs/
  base_env.py     -- abstract base class + GymEnv wrapper
  mujoco_env.py   -- base class for MuJoCo-backed environments

models/
  mlp.py          -- MLP for vector-based states
  cnn.py          -- CNN for image-based states
  mjcf/           -- MuJoCo robot definitions (MJCF XML, local dev)

configs/
  default.yaml    -- LunarLander-v3 (shipped)

checkpoints/      -- saved model weights (local)
```

## Documentation

- [`CLAUDE.md`](CLAUDE.md): AI instructions on architecture rules, project conventions, and how to extend the harness
- [`PLAN.md`](PLAN.md): A roadmap for MuJoCo integration, model export, hardware deployment, domain randomization
- [`docs/README.md`](docs/README.md): Human readible documentation for DQN concepts, reward shaping, and sim-to-real considerations
