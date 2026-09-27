---
description: Triage a policy that trains but behaves badly — measurement errors first, then incentives, then learning. Use when a trained checkpoint underperforms, ignores its sensor, hovers, or collapses.
argument-hint: <config-path> <checkpoint-path>
arguments: [config_path, checkpoint_path]
---

Work these in order and stop at the first one that explains the behaviour. Most apparent
learning failures are measurement failures, and the order below is cheapest-first.

## 1. Can the scorecard see the behaviour at all?

- Is the metric the **deliverable**, or average reward? A policy that lingers in a good state
  out-scores one that does the job — switch to the task's own metric.
- Does the evaluation setup let the behaviour happen? A homing metric measured from a start
  already at the target reads as noise regardless of policy quality.
- Are you scoring the checkpoint that evaluates best, or the latest one written? Check
  `_best.pt` against the last few numbered checkpoints on a fixed seed.
- Is the number from the same protocol as the baseline you are comparing against — same seed,
  episode count and difficulty? Seed spread alone moves scores by tens of percent.

## 2. Is the policy collapsed or blind?

```bash
venv/bin/python eval.py --config $config_path --checkpoint $checkpoint_path \
    --policy checkpoint random hold straight --episodes 40 --seed 7000 --no-render
```

- `Hact` ≈ 0 → the policy plays one action and ignores its observations. Reject it.
- Beats `random` but not `hold` → it has learned nothing; `hold` is the floor.
- Beats `hold` by a clear margin with `Hact` ~0.5+ → it is a real controller, and the problem is
  the size of the gap, not the existence of learning.

## 3. Does the reward actually pay for the behaviour you want?

Decompose it arithmetically on a typical trajectory, without training:

- What does the progress term pay per step? What does the per-step cost charge? If progress is
  net-negative, refusing to act is optimal and the agent is right.
- Can the agent farm anything by sitting still or by retracing its path? Any loiterable term
  will be loitered on.
- Are payouts so sparse that, under this `gamma`, continuing to work is under-credited? Try
  splitting the same total payout into finer increments.
- Is a shaping term's maximum at an intermediate hint rather than the true goal?

Test these with a scripted rollout through the environment — no learning needed.

## 4. Is the behaviour ever sampled, and does its credit propagate?

- `explore_hold` at 1 explores jitter, not trajectories. Raise it to ~10–20.
- Payoff hundreds of steps after the action → `n_step: 3`.
- Performance peaked then collapsed → confirm `double_dqn: true`.
- Buffer too small for the episode length → good experience is overwritten by failures.
- No successful episode ever occurs at this difficulty → add an in-environment curriculum with
  rehearsal, and cap it where the hand-coded ceiling collapses.

## 5. Only then, change the learner or the architecture

A small Q-gap between actions is not by itself a defect — what matters is whether the argmax
points the right way. Confirm on a measurement that can see the behaviour before reaching for a
dueling head, a bigger network, or a new algorithm.

Report the finding as: what the number was, what explained it, and which single change you
propose next. See `.claude/rules/reporting.md`.
