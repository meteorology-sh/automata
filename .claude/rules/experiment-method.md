# Rule: Falsify a change cheaply before you spend a training run on it

A training run is the most expensive possible test of a design hypothesis: hours of compute
for a delayed, noisy verdict. Three overnight arms were once lost because one bundled change
rested on an analytic error that a ten-second check would have caught.

## Must follow

- **Climb the cost ladder, and stop at the first rung that fails.** In order: a closed-form
  check — does the shaping telescope, what does the term pay per step → a scripted rollout
  through the environment with **no learning**, which shows whether the reward really changes the
  incentive on a fixed trajectory the way you claim → a 100-episode smoke train → the full run. A
  reward or incentive change is measurable with zero learning, so there is never an excuse to
  first *learn* whether a reward does what you think.

- **One lever per arm, measured against the current best.** Each arm changes exactly one thing
  versus the shipped recipe. Bundle two and a losing arm is uninterpretable — worse, one bad
  lever silently poisons every arm carrying it. If you must test a combination, also run each
  component alone.

- **Reproduce the baseline under the identical protocol before claiming a win or a
  regression.** Same seed, same episode count, same difficulty, same config — not a
  remembered number from a different protocol.

- **Distrust a local argument about a global quantity.** Marginal, per-step reasoning is
  locally right and globally wrong exactly when the reward has structure: potentials that
  telescope, one-time payouts, sums over a horizon. Check it empirically instead of on paper.

- **A change that contradicts a documented decision must refute that decision's stated reason
  on its own terms.** The rules and comments record *why* the current design is what it is.
  When your diagnosis implies one is wrong, engage its actual argument and show where it
  breaks. Overriding a lock with a fresh argument that never mentions it is how a correct,
  load-bearing design gets deleted.

- **Negative results are results — write them down.** "No lever beat the incumbent" and "this
  natural hypothesis is false" are worth as much as a win and save the next run. Record them
  with the evidence; see `.claude/rules/research-record.md`.

- **Expect the diagnosis, not just the result, to be wrong sometimes.** Several times the
  number was right and the story about it was not: the loss was never re-traversal, it was work
  not attempted. When a lever built on a diagnosis fails, re-test the diagnosis before
  building the next lever on it.
