---
paths:
  - "eval.py"
---

# Rule: How to judge a policy

Most wrong conclusions in an RL project come from a **measurement** mistake, not a training
mistake. Before redesigning the learner, make sure the scorecard can actually see the
behaviour you care about.

## Must follow

- **Never judge a policy by average reward.** Reward is the optimization target, not the
  report card. In any task where lingering in a good state is possible, a policy that camps
  banks more return than one that does the job efficiently. Pick a scorecard metric that
  *is* the deliverable — task completions, distinct ground covered, time-to-goal — and report
  reward separately, if at all.

- **Score against the blind baselines, and use the right floor.** `eval.py --policy
  checkpoint random hold straight` scores them on identical episodes. `hold`, which holds each
  random action for k steps, is the honest floor rather than `random`: coherent random legs score
  roughly twice per-step dithering on any task where the reward depends on where the agent goes. A
  policy that beats `random` but not `hold` has learned nothing.

- **Gate on within-episode action entropy, `Hact`.** Compute the entropy of the action
  histogram *inside* an episode, then average over episodes. Pooling across episodes makes a
  policy that plays a different constant action each episode look varied at ≈0.6, while the
  within-episode figure is 0.0 — the exact collapse the gate exists to catch. A policy with
  `Hact` ≈ 0 is not making
  decisions and must not be shipped, however good its score. Keep the `straight` baseline in
  the table permanently as proof the shipped policy is not that.

- **Evaluate on fixed, held-out episode seeds.** Set `env.seed`, or `eval.py --seed`, so
  every policy faces the same episodes. **Select** a checkpoint on one seed and **confirm**
  it on a disjoint one; reporting on the seeds you selected on is winner's curse. Seed-to-seed
  spread alone can move a score by tens of percent, so a single-cell comparison is close to
  noise.

- **Ship the checkpoint that evaluates best, not the latest.** Long training with a floored
  epsilon, and curricula, degrade late checkpoints. `_best.pt` is selected by
  `checkpoints.best_metric`, but the final word is an evaluation on the confirm seed.

- **Match the evaluation setup to the behaviour under test.** A metric measured in a
  situation where the behaviour cannot occur reads as noise regardless of policy quality — if
  you want to know whether the agent travels to a target, do not start it on the target. Ask
  "could this run distinguish a good policy from a bad one?" before trusting its number.

- **Establish the hand-coded ceiling before blaming the learner.** Write a scripted
  controller that sees only what the policy sees and score it on the same protocol. If it
  solves the task, any gap is learnability; if it cannot, no amount of training will help and
  the task or the horizon is the problem. See `.claude/skills/solvability-ceiling`.

- **A large Q-value spread is not a usable action preference.** What matters is whether the
  argmax points the right way, not how big the gap is. Do not change architecture on the
  strength of a value-spread statistic.
