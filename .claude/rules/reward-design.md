---
paths:
  - "envs/*.py"
  - "envs/**/*.py"
---

# Rule: Reward design

The reward is where a task breaks most easily, because the learner optimizes exactly what is
written, not what was meant. Design adversarially: for any reward, ask "how would an
optimizer farm this without doing the job?" Rewards live in the environment's
`_get_reward()` / `step()`, never in the training loop.

## Must follow

- **Reward the deliverable, not a proxy that correlates with it in easy cases.** What you
  count determines the geometry the optimizer converges to — the learner just finds the
  optimum of the counting rule. Counting *distance travelled* in a good region selects a
  back-and-forth shuttle down one lane; counting *distinct ground first visited* selects a
  space-filling sweep. Same physics, same network, same learner: only the counting rule
  changed.

- **Pay once for fresh progress, and never for a state you can sit in.** A one-time payout
  per newly achieved thing is camping-proof, since a hover claims one unit then nothing, and
  shuttle-proof, since re-covering pays nothing. Any term you can loiter on will be loitered on.

- **Read the objective from ground truth, the observation from the sensor.** Compute the
  payout from the true, noiseless state, and let the noisy or partial version appear only in
  the observation. Paying out on the noisy reading trains the agent to farm sensor noise.

- **Crash, or any hard failure, is always a clear negative and terminal.** Without it the buffer
  fills with indistinguishable low-positive "alive but idle" transitions and the agent never
  learns to survive before it learns to perform. Forfeiting the rest of the episode is the
  real deterrent.

- **Check the arithmetic of each term against the per-step penalty floor.** Write down what
  the progress term pays per step in the typical case and compare it with the per-step cost of
  battery, time and control penalties. If progress pays +0.00023/step against a 0.0006/step
  cost, doing the task is net-negative and the agent will correctly refuse to do it. This
  one-minute decomposition catches more failures than a training run.

- **Payout frequency is a lever distinct from magnitude and geometry.** Splitting the *same*
  total payout into more, smaller pulses — finer progress increments — can convert a policy
  that banks a little and then stops into one that keeps working — under a high discount,
  sparse payouts under-credit the act of *continuing*. Change frequency alone, hold total
  reward per unit of progress constant, and measure.

- **Keep shaping terms potential-based, and keep them undiscounted.** A distance-to-goal
  bonus written as `Φ(s') − Φ(s)` telescopes over any round trip to exactly zero, so it
  cannot be farmed by going back and forth, and lingering nets nothing. The theoretically
  cleaner discounted form `γΦ(s') − Φ(s)` adds a standing `−(1−γ)Φ` every step: over a long
  episode that integrates into a penalty big enough to make success net-negative, and the
  agent learns that **terminating early beats continuing**. Shape `Φ` so its maximum is the
  true goal, never an intermediate hint the agent could fly to and stop at.

- **Keep components balanced, bounded, and few.** No term should dominate; clip the total to
  a sane range for stability. Every extra term is another surface to farm.
