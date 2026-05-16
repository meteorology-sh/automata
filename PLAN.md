# PLAN — RL Harness: First Look to Working Demo

## Feasibility Assessment

**Verdict: Feasible.** The harness is a complete, functional DQN training system with real-time visualization. Every core component is implemented and follows the architectural rules in CLAUDE.md. A working demo can be produced without writing any new infrastructure — only configuration and a thin environment wrapper.

---

## What Already Works

| Component | File | Status |
|---|---|---|
| DQN agent (epsilon-greedy, replay, Bellman) | `core/agent.py` | Complete |
| Replay buffer | `core/memory.py` | Complete |
| Training loop (environment-agnostic) | `core/train.py` | Complete |
| Real-time 4-panel dashboard | `core/visualizer.py` | Complete |
| Abstract environment interface | `envs/base_env.py` | Complete |
| Gymnasium wrapper (`GymEnv`) | `envs/base_env.py` | Complete |
| MLP model (vector states) | `models/mlp.py` | Complete |
| CNN model (image states) | `models/cnn.py` | Complete |
| YAML config system | `configs/default.yaml` | Complete |
| Entry point with CLI args | `main.py` | Complete |

The default config targets **CartPole-v1** with an MLP, 1000 episodes, and a solve threshold of 475. Running `python main.py` should train an agent end-to-end with a live dashboard.

---

## What's Missing (for a demo)

Nothing structural. The gaps are operational:

1. ~~**Dependencies not verified**~~ — Resolved. Dependencies install and train.py runs to completion.
2. ~~**No smoke test**~~ — CartPole-v1 trained successfully through 700+ episodes with checkpoints saved.
3. **No custom environment** — only the generic `GymEnv` wrapper exists. A domain-specific subclass would prove the harness is truly extensible.
4. **No tests** — no pytest infrastructure. `requirements.txt` includes pytest but no test files exist yet.

---

## Plan

### Phase 1 — Validate the foundation ✓

> Goal: Confirm the harness runs, trains, and visualizes without errors.

- [x] **1.1** Install dependencies from `requirements.txt` into a clean venv. Fix any version conflicts.
- [x] **1.2** Run `python main.py --config configs/default.yaml` on CartPole-v1. Confirm:
  - Training loop starts and episodes accumulate.
  - Visualizer window opens with all 4 panels updating.
  - Epsilon decays on schedule.
  - Checkpoints are written to `checkpoints/`.
- [x] **1.3** Let it run to convergence (or at least 200 episodes). Verify:
  - Reward trend climbs in the dashboard.
  - Best checkpoint is saved separately from periodic checkpoints.
  - Training stops early if solve threshold (475) is reached.
- [x] **1.4** Document any bugs or friction found during the run.

**Phase 1 findings:**

- Training ran through 700+ episodes. Periodic checkpoints saved every 100 episodes (`CartPole-v1_0.pt` through `CartPole-v1_700.pt`). Best checkpoint tracked separately as `CartPole-v1_best.pt`.
- **Bug found:** `core/train.py` lines 89 and 95 build checkpoint paths via string concatenation (`f"{checkpoint_dir}{env_name}_{episode}.pt"`) instead of `os.path.join()`. This works only because `default.yaml` sets `dir: checkpoints/` with a trailing slash. A config without the trailing slash would write files to the wrong location. Needs fix in Phase 2.

### Phase 2 — Fix what's broken ✓

> Goal: Patch anything Phase 1 surfaces so the harness is reliable.

- [x] **2.1** Fix checkpoint path construction in `core/train.py` — replaced string concatenation with `os.path.join()` on both checkpoint save paths.
- [x] **2.2** Verify visualizer threading on Windows. Default backend is `tkagg` — correct for Windows. No explicit backend override needed; visualizer ran successfully in Phase 1.
- [x] **2.3** Confirm `--no-vis` flag works for headless training — ran 3-episode headless session, no errors. Also fixed a `UserWarning` in `core/agent.py` where `torch.tensor()` was called on a list of numpy arrays instead of a pre-stacked `np.array()`.
- [x] **2.4** Confirm checkpoint filenames follow the `{env_name}_{episode}.pt` pattern — verified after the `os.path.join()` fix. Headless run produced correct paths.

### Phase 3 — Prove extensibility with a second environment ✓

> Goal: Demonstrate the harness is truly generic by adding a non-trivial environment.

- [x] **3.1** Target: **Acrobot-v1** — 6-dimensional vector state, 3 discrete actions. (Original plan targeted LunarLander-v3, but Box2D and pygame don't build on Python 3.14. Acrobot is included in base Gymnasium and still exercises different `state_size`/`num_actions`.)
- [x] **3.2** Created `configs/acrobot.yaml` with tuned hyperparameters (1500 episodes, batch 64, lr 0.0005, epsilon decay 0.997).
- [x] **3.3** Ran training headless — training loop and checkpoint saving worked with zero code changes. Checkpoints saved as `Acrobot-v1_0.pt`, `Acrobot-v1_best.pt`.
- [x] **3.4** Created `envs/shaped_acrobot.py` — a `ShapedAcrobot` subclass of `BaseEnv` with a dense height-based reward bonus. Added config-driven env dispatch to `main.py` via an `env.wrapper` key (e.g., `wrapper: "envs.shaped_acrobot.ShapedAcrobot"`). Existing configs without the key still use `GymEnv`. Tested all three paths (CartPole/GymEnv, Acrobot/GymEnv, Acrobot/ShapedAcrobot) — all work.

**Phase 3 findings:**

- The env dispatch in `main.py` (`build_env()`) uses `importlib` to load a class from the `env.wrapper` config key. This keeps the training loop environment-agnostic.
- ShapedAcrobot rewards (~-458) differ from vanilla Acrobot (-500) even early on, confirming the height bonus is active.
- No new pip dependencies required — Acrobot ships with base Gymnasium.

### Phase 4 — Polish the demo

> Goal: Make the result presentable and reproducible.

- [ ] **4.1** Add a basic pytest smoke test (`tests/test_smoke.py`):
  - Instantiate `ReplayBuffer`, push and sample transitions.
  - Instantiate `DQNAgent` with a small MLP, call `select_action` and `train`.
  - Instantiate `GymEnv("CartPole-v1")`, run 1 episode to completion, assert no crashes.
  - Verify `state_to_tensor` produces correct shapes for both MLP and CNN paths.
  - Test `build_env()` with and without a `wrapper` key.
- [ ] **4.2** Clean up any dead code or unused config keys found during Phases 1-3.
- [ ] **4.3** Add action label support to the visualizer. Environments should optionally provide action names (e.g., `["Left", "Right"]` for CartPole, `["Torque -1", "None", "Torque +1"]` for Acrobot) so the Q-value bar chart displays readable labels instead of `Action 0`, `Action 1`.
- [ ] **4.4** Commit a working state with all configs and fixes.

---

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| ~~Matplotlib backend issues on Windows~~ | ~~Medium~~ | Resolved — `tkagg` backend works, no override needed |
| ~~CartPole doesn't converge with default hyperparams~~ | ~~Low~~ | Resolved — CartPole trained successfully |
| ~~Box2D dependency doesn't install cleanly on Windows~~ | ~~Medium~~ | Resolved — pivoted to Acrobot-v1, no extra deps |
| CNN path untested (no image-based env in demo) | Low | Defer to future phase; MLP path is sufficient for demo |

---

## Success Criteria

The demo is "working" when:

1. ~~`python main.py` trains a DQN agent on CartPole-v1 and shows a live dashboard.~~ ✓
2. ~~The agent's reward visibly improves over training.~~ ✓
3. ~~A second environment trains with only a config change.~~ ✓ (Acrobot-v1)
4. ~~Checkpoints are saved correctly.~~ ✓
5. The whole thing is reproducible from a clean `pip install -r requirements.txt`.

---

## Current Status (2026-05-16)

**Phases 1-3 complete. Phase 4 is next.**

The harness is validated, patched, and proven generic. Two environments train successfully (CartPole, Acrobot) with zero training-loop changes. A custom `BaseEnv` subclass (`ShapedAcrobot`) demonstrates that reward shaping lives in the environment. Config-driven env dispatch added to `main.py`. Next step is polish: smoke tests, action labels, cleanup.

---

## What Comes After the Demo

These are out of scope for this plan but worth noting:

- **Custom environments** — subclass `BaseEnv` for a domain-specific simulator (robotics, game, control system).
- **Image-based environment** — test the CNN path with an Atari or pixel-observation env.
- **Advanced algorithms** — Double DQN, Dueling DQN, prioritized experience replay.
- **Offline analysis** — TensorBoard or Weights & Biases integration alongside the live visualizer.
- **Hyperparameter sweeps** — Optuna or grid search over config values.
- **Test coverage** — full pytest suite for agent, memory, models, and training loop.
