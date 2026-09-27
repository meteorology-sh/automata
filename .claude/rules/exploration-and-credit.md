---
paths:
  - "core/agent.py"
  - "configs/*.yaml"
---

# Rule: Exploration and credit assignment

On long-horizon tasks the binding constraint is usually not the reward or the network — it is
that the agent never performs the behaviour by chance, so nothing ever pays it. These are the
mechanisms that address that, in the order they matter. Each is config-gated and a no-op at
its default.

## Must follow

- **Extend exploration in time before touching anything else**, with `training.explore_hold`.
  Per-step epsilon-greedy re-draws a random action every step, and independent nudges average
  out to roughly zero net displacement: it explores jitter, not trajectories. Holding one
  random action for ~10–20 steps produces coherent exploratory manoeuvres, which is the only
  way a policy discovers behaviour that only pays off after sustained motion. On a
  travel-and-search task this was the single largest lever, and it also raises the blind floor
  the policy must beat; see `.claude/rules/evaluation.md`.

- **Use n-step returns when the payoff is far from the action that earned it**, via
  `training.n_step`, typically 3. With `gamma = 0.995` and a reward that arrives hundreds of
  steps later, 1-step bootstrapping propagates credit too slowly to shape early decisions.
  n-step is generic and environment-agnostic; the training loop must call
  `agent.end_episode()` once per episode so partial windows flush and no return spans a reset.

- **Keep Double DQN on**: `training.double_dqn: true` is the default. With vanilla DQN the same
  noisy network both picks and scores the next action, which systematically overestimates Q;
  the symptom is performance that peaks and then collapses.

- **Never store a timeout as a terminal.** Truncation is the clock ending the episode, not the
  MDP ending it, so its target must still bootstrap `gamma·Q(s')`. Collapsing
  `done or truncated` into the stored flag zeroes that bootstrap and biases exactly the
  long-horizon states you need valued. The loop stores `done` only and uses
  `done or truncated` for loop control.

- **Put curricula and rehearsal in the environment, never in the training loop.** When a task
  is unlearnable from the full difficulty, because no successful episode ever happens by chance, have
  the environment start easy and widen on a rolling success rate, mixing in a fraction of
  easier episodes as rehearsal so the earlier skill is not forgotten. Expose the current
  setting as `info["difficulty"]` — the loop prints it without knowing what it means. Expect a
  curriculum to turn 0% into learning but to **stall short of the full task**: calibrate the
  advance threshold against the *epsilon-suppressed* score, and cap difficulty where the
  hand-coded ceiling itself collapses. Training for a difficulty the platform cannot service
  within its horizon is wasted compute.

- **Mind the epsilon floor at both ends.** Too high and random actions tank the score of a
  competent policy; too low and a degraded policy cannot escape a collapse, because short
  failing episodes fill the buffer faster than long good ones. Scale `memory_size` with
  `max_episode_steps` so good experience survives a bad patch.

- **Prefer an exploration or credit-assignment fix to a reward rewrite.** When a policy
  refuses to do the task, the reward is the *first* thing to check arithmetically, per
  `.claude/rules/reward-design.md`, but usually the *last* thing that needs redesigning. The
  fix is normally that the behaviour is never sampled, or that its payoff never propagates
  back to the decision that caused it.
