---
description: Write a hand-coded controller for a task and measure the ceiling and the blind floors before training anything. Use when starting a new environment, or before blaming the learner for a bad score.
argument-hint: <config-path>
arguments: [config_path]
---

Before a single episode is trained on `$config_path`, establish the two numbers that make every
later result interpretable: what a scripted controller can achieve, and what a policy that
ignores its observations already scores.

## 1. Write the hand-coded controller

Create `probe_<task>.py` — a scripted controller that solves the task **using only the
information the policy will have**: the same observation, the same action set, the same episode
limit. No privileged access to hidden state. Keep it simple and geometric: go to the hint,
sweep a pattern, climb a gradient.

Score it on the task's own metric rather than reward, greedy, on a fixed seed, over enough episodes to
be stable.

This number is the ceiling. Report it as a table row.

## 2. Measure the blind floors

```bash
venv/bin/python eval.py --config $config_path --policy random hold straight \
    --episodes 40 --seed 7000 --no-render --metric <the task's metric key>
```

`hold` is the honest floor, not `random`. Note the `straight` score too — it is what a collapsed
policy can bank without using its observations.

## 3. Decide before training

| Outcome | What it means | What to do |
|---|---|---|
| Ceiling solves the task comfortably | The task is feasible in the horizon | Train; any gap is learnability |
| Ceiling barely solves it | The horizon or the geometry is the binding constraint | Shrink the task — shorter distances, an easier start — or lengthen the episode before training |
| Ceiling cannot solve it | No amount of training will | Fix the task, the sensing, or the horizon |
| `straight` ≈ ceiling | The metric cannot distinguish searching from luck at this difficulty | Cap difficulty where the two separate |

State these numbers in your first report on the task, and keep them in the scorecard from then
on. If the hand-coded controller beats every policy you train, that is a legitimate result worth
writing down, per `.claude/rules/research-record.md` — and a sign the task may not need RL at all.
