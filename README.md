# Automata

Automata is a generic reinforcement learning harness for training DQN agents on simulated robots. Define a robot in MJCF XML, write a reward function, point it at a config, and train. The harness handles the training loop, replay buffer, epsilon scheduling, checkpointing, and visualization.

MuJoCo provides the physics and 3D rendering, while the harness runs training and policy definitions.

## Quick Start

Automata ships with a default scenario in the form of [Gymnasium's Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/). Follow these instructions to install dependencies, train the model, and evaluate its performance.

First, install [Python 3.12](https://www.python.org/downloads/release/python-31213/)

Then, in the root directory, instantiate a virtual environment:

```bash
python3.12 -m venv venv
```

```bash
source ./venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Training

```bash
# Train headless
python main.py --config configs/default.yaml
```

```bash
# Train with the real-time reward dashboard
python main.py --config configs/default.yaml --vis
```

### Evaluation

```bash
# Evaluate with 3D rendering
python eval.py --config configs/default.yaml --checkpoint checkpoints/LunarLander-v3_best.pt
```

```bash
# Evaluate headless
python eval.py --config configs/default.yaml --checkpoint checkpoints/LunarLander-v3_best.pt --no-render
```

```bash
# Score it against the blind baselines on fixed seeds — the honest scorecard
python eval.py --config configs/default.yaml --checkpoint checkpoints/LunarLander-v3_best.pt \
    --policy checkpoint random hold straight --episodes 40 --seed 7000 --no-render
```

`random` acts at random every step. `hold` holds each random action for a few steps and is the real
floor — coherent random legs score far above per-step dithering. `straight` plays one constant action
per episode, which is the collapse detector. The table also reports `Hact`, action entropy measured
within an episode: a policy near 0 is playing one action and ignoring its observations, however good
its reward looks. Select a checkpoint on one seed, confirm it on a disjoint one.

### Checks

```bash
# Tests and the strict type check; both should be clean
python -m pytest tests/ -q
python -m pyright --pythonpath venv/bin/python
```

## How It Works

A trained policy is produced from three independent inputs:

| Input                  | Defines                                                                | Example                        |
| ---------------------- | ---------------------------------------------------------------------- | ------------------------------ |
| **MJCF XML**           | What the robot _is_ — geometry, mass, joints, actuators                | `models/mjcf/quadrotor.xml`    |
| **MujocoEnv subclass** | What it _should learn_ — reward function, observations, termination    | `envs/quadrotor_hover.py`      |
| **Config YAML**        | How to _train_ — hyperparameters, model architecture, epsilon schedule | `configs/quadrotor_hover.yaml` |

The same robot can back different tasks — hover, land, track waypoints — by swapping the env subclass. The training loop never changes.

## Shipped Environment

The default config trains on **LunarLander-v3** — land a spacecraft between the flags using discrete thrust controls. Solved when average reward exceeds 200 over 50 episodes.

For custom robots, subclass `MujocoEnv` and provide an MJCF XML file — see [Adding a New Robot](#adding-a-new-robot).

## Adding a New Robot

1. Write an MJCF XML file describing the robot in `models/mjcf/your_robot.xml`
2. Subclass `MujocoEnv`
3. Implement `_get_obs()`, `_get_reward()`, `_is_done()`, `_apply_action()`
4. Create a config in `configs/`
5. Train: `python main.py --config configs/your_robot.yaml`

See [`CLAUDE.md`](CLAUDE.md) for full conventions and reward design principles.

## Project Structure

```
core/
  agent.py        -- DQN agent: epsilon-greedy + held exploration, replay, soft target,
                     Double DQN, n-step returns
  memory.py       -- replay buffer with a per-sample bootstrap discount
  train.py        -- training loop; environment-agnostic, checkpoint selection by any metric
  visualizer.py   -- real-time reward dashboard

envs/
  base_env.py     -- abstract base class + GymEnv wrapper + reproducible episode seeding
  mujoco_env.py   -- base class for MuJoCo-backed environments

models/
  mlp.py          -- MLP and DuelingMLP for vector-based states
  cnn.py          -- CNN for image-based states
  mjcf/           -- MuJoCo robot definitions in MJCF XML, local dev

configs/
  default.yaml    -- LunarLander-v3, shipped

tests/            -- test_smoke.py for framework wiring, test_agent.py for agent mechanisms,
                     plus per-environment test files, local

data/             -- the experiment record: one file per finding + INDEX.md

eval.py           -- scorecard: a checkpoint against the blind baselines, with action entropy

pyrightconfig.json -- pyright strict over core, envs, models, tests, main.py, eval.py

checkpoints/      -- saved model weights, local
```

## Documentation

- [`CLAUDE.md`](CLAUDE.md): AI instructions on architecture rules, project conventions, and how to extend the harness
- [`docs/README.md`](docs/README.md): Human readable documentation for DQN concepts, what the harness adds beyond vanilla DQN, reward shaping, scoring a policy honestly, and sim-to-real considerations
- [`.claude/rules/`](.claude/rules): the enforced conventions — reward design, exploration and credit assignment, how to judge a policy, how to run an experiment
- [`data/INDEX.md`](data/INDEX.md): the experiment record — research question, parameters, results, conclusion
