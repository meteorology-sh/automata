---
paths:
  - "envs/*.py"
  - "envs/**/*.py"
---

# Rule: Observation design

A policy can only act on what is in its state. Most "the agent can't learn this" problems are really
"the agent cannot see this", and the rest are "the agent is being asked for precision its sensor does
not have". Decide both before training.

## Must follow

- **Give memory, not an oracle.** When a single instantaneous reading is ambiguous — one scalar with no
  direction, a partial view, anything where the useful quantity is a *change* — the agent has to move to
  sense. Support that honestly by stacking the last several time-steps of readings and positions, and by
  adding a little distilled memory such as the best value seen so far and the displacement back to where
  it was seen. Do not hand the policy a computed gradient or a privileged target vector it could not
  measure in deployment; that trains a policy that cannot be shipped.

- **Include a coarse hint when the task is otherwise unsearchable, and keep its weight low.** A rough
  prior about where to start turns an impossible search into a feasible one. Expose it as a displacement
  in the observation, never as something the reward pays for approaching, or the agent learns to fly to
  the guess and stop.

- **Measure the observation's resolution floor before choosing a success tolerance.** Compute the change
  in the signal across the tolerance you want and compare it with the noise. If the difference is at or
  below the noise, the task is impossible for *any* controller, learned or hand-coded, and no amount of
  training will change it — the lever is a second channel, a longer baseline, or a looser tolerance.
  Choose tolerances from the physics of the deliverable, never for difficulty: an over-tight criterion
  put one whole campaign below the sensor floor and made every experiment in it uninterpretable.

- **Treat the uninformative range as a control signal, not a training problem.** Beyond some distance or
  below some level the reading carries no usable information. That is the natural gate for handing
  control to a systematic pattern, per `.claude/rules/task-decomposition.md`, rather than something to
  train the policy through.

- **More observation is not better.** A finer grid or an extra channel makes the input harder to learn
  from within the same episode budget — a finer visitation map measurably did not pay for itself. Add a
  channel only when you can name the decision it changes, and keep the vector as small as the task
  allows.

- **Normalize, bound, and keep the layout stable.** Scale components to comparable magnitudes. The
  observation layout is part of the checkpoint: when it changes, every earlier checkpoint is silently
  invalid. Assert the expected `state_size` when loading a checkpoint and fail loudly rather than
  running a model on a vector it was never trained on.

- **The observation is the noisy channel; the reward reads ground truth.** Sample the sensor for
  `_get_obs()` and the true state for `_get_reward()`, so the agent cannot farm sensor noise. See
  `.claude/rules/reward-design.md`.
