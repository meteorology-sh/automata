# CLAUDE.md — RL Harness

## Purpose

This is a generic reinforcement learning harness. It provides reusable infrastructure for training DQN agents in any Gymnasium-compatible environment. Users extend it by subclassing `BaseEnv` and providing a config.

The harness stays domain-agnostic: everything task-specific — physics, observation, reward, curriculum — lives in the environment subclass and its config, and **the training loop never knows what it is training.**

---

## Rules

These must be followed at all times:

- Never hardcode environment names, model sizes, or hyperparameters. All configuration lives in `configs/`.
- All environments must subclass `BaseEnv` in `envs/base_env.py`. Never interact with a simulator directly from the training loop.
- Never apply softmax to Q-value outputs. The decision model outputs raw logits.
- The training loop in `core/train.py` must remain environment-agnostic. No drone, vacuum, or CartPole-specific logic belongs there. `checkpoints.best_metric` may name any numeric `info` key, and the loop maximizes it without knowing what it means.
- Always save checkpoints to `checkpoints/` using the pattern `{env_name}_{episode}.pt`. Never overwrite the best checkpoint.
- Reward functions belong in the environment subclass, not in the training loop.
- All models must accept a `num_actions` argument. Never hardcode action counts.
- Strict-typed under pyright strict via `pyrightconfig.json`, zero errors. Run it as `venv/bin/python -m pyright --pythonpath venv/bin/python`, or phantom import errors appear.
- Never judge a policy by average reward, and never on the seeds you selected it on. See `.claude/rules/evaluation.md`.
- Never store a timeout as a terminal transition. Truncation is the clock, not the MDP.
- Falsify a design change cheaply before spending a training run on it, and change one lever per run. See `.claude/rules/experiment-method.md`.
- Decide what is learned and what is ordinary code before training, and check the task fits the platform's budget. See `.claude/rules/task-decomposition.md`.

### Rule files in `.claude/rules/`

| File | Covers |
|---|---|
| `dqn-agent.md` | loss, gradient clipping, target network, tensor conversion |
| `environments.md` | subclassing, render passthrough, the `info` reporting keys, episode seeding |
| `task-decomposition.md` | what to learn versus what to write as code, layer metrics, budgeting the task |
| `observation-design.md` | what the policy can see, memory instead of an oracle, the sensor's resolution floor |
| `platform-model.md` | honest vehicle models: real component data, which limit binds, disturbances, randomization |
| `reward-design.md` | what to count, one-time vs loiterable payouts, shaping, the per-step arithmetic check |
| `exploration-and-credit.md` | held exploration, n-step, Double DQN, truncation, curricula |
| `evaluation.md` | the scorecard: blind floors, action entropy, held-out seeds, hand-coded ceiling |
| `experiment-method.md` | falsify cheaply, one lever per arm, reproduce the baseline |
| `hyperparameters.md` | canonical baseline and the interaction rules between keys |
| `training-loop.md` | loop and visualizer invariants |
| `mujoco.md`, `testing.md`, `venv.md` | simulator, test layout, interpreter |
| `typing.md` | pyright strict at zero errors, and how to type the untyped boundaries |
| `long-running-jobs.md` | detached launches, ETAs, never idling the session |
| `docs-consistency.md` | docs describe the system as it is, stay consistent, and are not changelogs |
| `version-control.md` | read history freely, never commit unless asked, what stays local |
| `reporting.md` | run cadence, table-first reports, plain language |
| `research-record.md` | how findings are recorded in `data/` |

---

## Architecture

```
core/
  agent.py            ← DQN agent: epsilon-greedy + held exploration, replay, soft target,
                        Double DQN, n-step returns
  memory.py           ← replay buffer; transitions carry a per-sample bootstrap discount
  train.py            ← training loop; environment-agnostic, terminated-only bootstrap,
                        checkpoint selection by any info metric
  visualizer.py       ← real-time reward dashboard

envs/
  base_env.py         ← abstract base class + GymEnv wrapper + reproducible episode seeding
  mujoco_env.py       ← base class for MuJoCo-backed environments;
                        task environments are local, not shipped

models/
  mlp.py              ← MLP and DuelingMLP for vector-based states
  cnn.py              ← CNN for image-based states
  mjcf/               ← MuJoCo robot definitions in MJCF XML, local

configs/
  default.yaml        ← LunarLander-v3, the shipped default; task configs are local

tests/
  test_smoke.py       ← framework wiring: buffer, agent, models, env wrapper, eval
  test_agent.py       ← agent mechanisms: n-step arithmetic, held exploration, dueling head

data/                 ← the experiment record: one file per finding + INDEX.md
docs/README.md        ← RL and DQN fundamentals, and what the harness adds
main.py               ← training entry point
eval.py               ← scorecard: a checkpoint against the blind baselines, with action entropy
pyrightconfig.json    ← pyright strict over core, envs, models, tests, main.py, eval.py
checkpoints/          ← saved model weights, local
```

---

## CLI

```bash
# Training, headless by default
python main.py --config configs/default.yaml
python main.py --config configs/default.yaml --vis    # enable reward dashboard

# Evaluate one checkpoint; renders by default
python eval.py --config configs/default.yaml --checkpoint checkpoints/LunarLander-v3_best.pt

# Checks — both must be clean before handing work back
python -m pytest tests/ -q
python -m pyright --pythonpath venv/bin/python

# Score it against the blind baselines on fixed seeds — the honest scorecard
python eval.py --config configs/default.yaml --checkpoint checkpoints/LunarLander-v3_best.pt \
    --policy checkpoint random hold straight --episodes 40 --seed 7000 --no-render
# add --metric <info key> to report the task's own metric instead of reward
```

Select a checkpoint on seed 7000 and confirm it on the disjoint seed 3000.

---

## Skills

Slash-invocable workflows in `.claude/skills/`:

| Skill | Use it when |
|---|---|
| `frame-task` | a physical task is described and nothing exists yet: what to learn, the horizon, the observation, the reward, the ceiling |
| `new-env` | scaffolding a config for a new environment |
| `validate-config` | before a run, or when training is unstable or not converging |
| `solvability-ceiling` | starting a task: measure the hand-coded ceiling and blind floors first |
| `diagnose-training` | a trained policy underperforms, hovers, collapses, or ignores its sensor |
| `run-lever` | taking one proposed change from hypothesis to verdict |
| `write-finding` | recording a settled result in `data/` |

### Adding a new environment

Starting from a described task rather than a known environment — a vehicle, a sensor, a goal — run
`frame-task` first: it settles what is learned versus coded, whether the task fits the episode budget, and
what the observation and scorecard have to be, before any code exists.

For Gymnasium environments, use `GymEnv` wrapper. For custom physics, subclass `MujocoEnv`:

1. Create `envs/your_env.py` and subclass `MujocoEnv`, or `BaseEnv` for non-MuJoCo
2. Implement `_get_obs()`, `_get_reward()`, `_is_done()`, `_apply_action()`
3. Override `_reset_state()` for initial randomization — draw from `self.rng` after `_begin_episode()`
4. Override `_configure_camera()` to set the viewer viewport
5. Add a config in `configs/your_env.yaml` with `wrapper: "envs.your_env.YourEnv"`
6. Report what the loop should track through `info` — see below
7. Create `tests/test_your_env.py`; never add env tests to `test_smoke.py`
8. Write the hand-coded controller and measure the floors **before** training, with the `solvability-ceiling` skill

```python
from envs.mujoco_env import MujocoEnv

class YourEnv(MujocoEnv):
    def _apply_action(self, action: int):
        # map discrete action to self.data.ctrl

    def _get_obs(self) -> np.ndarray:
        # return observation vector

    def _get_reward(self) -> float:
        if self._is_crashed():
            return -1.0  # crash penalty is mandatory
        # compute reward components
        return float(np.clip(reward, -1.0, 1.0))

    def _is_done(self) -> bool:
        return self._is_crashed()

    def _reset_state(self):
        # randomize initial qpos/qvel from self.rng

    def _configure_camera(self):
        # set cam.lookat, cam.distance, cam.elevation from env params
```

### The `info` dict is the environment's reporting channel

The loop is environment-agnostic, so anything it should track arrives through `info` from `step()`:

- `info["is_success"]`, a bool — for `checkpoints.best_metric: "success"`
- any numeric key, e.g. `info["coverage_score"]` — name it in `best_metric` and `_best.pt` tracks its rolling mean
- `info["difficulty"]`, a float — a curriculum's current setting, printed per episode

Curricula and rehearsal live in the environment, never in the loop.

### Choosing a model

- Use `models/mlp.py` when the state is a numerical vector of position, velocity and sensor readings
- Use `models/cnn.py` when the state is an image such as a camera frame
- `DuelingMLP`, enabled with `model.dueling: true`, splits Q into a state value plus a mean-centred advantage — reach for it only when a measurement that can see the behaviour says the action preference is swamped by the value baseline
- All output raw Q-values and accept `(state_size_or_input_shape, num_actions)` — never softmax

### Reward function design

Design rewards inside `_get_reward()`. Full rule: `.claude/rules/reward-design.md`.

- Always return -1.0 on crash — without this, the replay buffer fills with indistinguishable low-positive crash transitions and the agent never learns stability
- **Reward the deliverable, not a proxy.** What you count decides the geometry the optimizer converges to: counting distance travelled in a good region selects a back-and-forth shuttle; counting distinct ground first visited selects a sweep
- **Pay once for fresh progress**, and never for a state the agent can sit in — any loiterable term will be loitered on
- Read the payout from ground truth; let the noisy version appear only in the observation
- **Check the arithmetic**: what the progress term pays per step versus what the per-step cost charges. If progress is net-negative, refusing to act is optimal
- **Payout frequency is its own lever** — the same total reward split into finer increments can turn a policy that stops early into one that keeps working
- Keep shaping potential-based and **undiscounted**, `Φ(s') − Φ(s)`; the discounted form's standing `−(1−γ)Φ` term can make success net-negative over a long episode, and then crashing beats flying
- No single component should dominate; clip the total to [-1, 1]
- Think adversarially: how would an optimizer game this function without doing the job?

### Judging a policy

Never by average reward, and never on the seeds used to select it. Score the checkpoint against
`random`, the `hold` floor and the `straight` collapse detector on fixed seeds, report
within-episode action entropy, and compare against a hand-coded ceiling that sees only what the
policy sees. Ship the checkpoint that evaluates best, not the latest. `.claude/rules/evaluation.md`.

### Running an experiment

A training run is the most expensive possible test of a hypothesis. Check it in closed form, then
with a scripted rollout that needs no learning, then with a short smoke train, and only then spend
the run — one lever per arm, against a baseline reproduced under the identical protocol. Launch
detached with an announced ETA, and write the verdict into `data/`, a null result included.
`.claude/rules/experiment-method.md`, `.claude/rules/long-running-jobs.md`.

### Hyperparameters

All hyperparameters are read from config YAML files. Canonical baseline:

```yaml
epsilon:
  start: 0.9
  min: 0.01
  decay: 0.995
training:
  gamma: 0.99
  batch_size: 128
  lr: 0.0001
  memory_size: 10000    # scale with max_episode_steps
  train_every: 1
  tau: 0.005
  double_dqn: true      # keep on; vanilla DQN overestimates and decays after its peak
  n_step: 1             # >1 when the payoff is hundreds of steps after the action
  explore_hold: 1       # >1 for coherent exploratory manoeuvres — the big long-horizon lever
  solve_window: 50
  solve_threshold: 200  # set 10-15% below expected peak
checkpoints:
  best_metric: "reward" # or "success", or any numeric info key
```

Override per-environment by creating `configs/your_env.yaml`. Key tuning rules:
- Scale `memory_size` with `max_episode_steps` — long episodes need larger buffers to resist the death spiral, because short crash episodes overwrite good data
- Set `solve_threshold` conservatively below expected peak — if unreachable, training continues past the cliff. When judging by a custom `best_metric`, disable it instead by setting it out of reach
- Epsilon should reach `epsilon_min` at ~80% of the episode budget
- For a long-horizon task: `gamma: 0.995`, `n_step: 3`, `explore_hold: 15`
- See `.claude/rules/hyperparameters.md` and `.claude/rules/exploration-and-credit.md` for detailed interaction rules

### Stopping criteria

The training loop stops when average reward over the last `solve_window` episodes, 50 by default, exceeds `solve_threshold`, or when the episode budget is exhausted. Always checkpoint the best-performing model — and note that "best" is `checkpoints.best_metric`, which need not be reward.

---

## Visualizer

The visualizer is a single-chart dashboard that runs in a background thread during training. It displays:

- Episode reward over time, as a rolling chart with the solve threshold drawn on it
- Rolling average matching the `solve_window`, so the chart predicts when training stops
- Current epsilon as a decimal value in the title

The training loop calls `visualizer.end_episode(total_reward, epsilon)` once per episode. The visualizer must never add per-step computation to the training loop. It shows reward, which is the optimization target — judge the policy with `eval.py`, not with the chart.

---

## DQN Training Loop Reference

```
for each episode:
    state = env.reset()
    while not done:
        action = agent.select_action(state)          # epsilon-greedy, possibly holding
        next_state, reward, done, truncated, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)   # TERMINATED only
        done = done or truncated                     # loop control only
        agent.train()                                # every train_every steps
        state = next_state
    agent.end_episode()                              # flush n-step window, clear hold
    decay epsilon
    checkpoint if best, by checkpoints.best_metric
```

---

## Maintenance

When adding new files, environments, configs, or concepts to the project, update this file to reflect them. Specifically:
- New files or directories → update the Architecture tree
- New environments → add to the tree and ensure the Skills section covers any new patterns
- New CLI flags → update the CLI section
- New rules or lessons learned → add to `.claude/rules/` and summarize here if they affect how environments are built or trained
- New experiment results → a file in `data/` plus its row in `data/INDEX.md`; if the result changes how the harness should be used, update the matching rule in the same pass

CLAUDE.md is loaded at the start of every session. If it is stale, the assistant wastes tokens re-exploring the codebase. Keep it current.

---

## Terminology

The vocabulary used throughout these rules — Q-value, replay buffer, held exploration, n-step
return, blind floor, action entropy, hand-coded ceiling — is defined in the glossary at the end of
[`docs/README.md`](docs/README.md), which also explains the mechanisms behind each one.
