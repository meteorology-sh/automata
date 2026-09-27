---
description: Turn a described physical task — a vehicle, a sensor, a goal — into a harness plan: what to learn versus code, the horizon, the observation, the reward, the metric and the ceiling. Use at the start of a new project, before writing an environment.
argument-hint: "<the task in one or two sentences>"
arguments: [task]
---

Work through these in order for `$task` and write the answers down before any code. Each step has a rule
behind it; read that rule when the answer is not obvious.

## 1. Split the job: what is learned, what is code

List the roles the system performs end to end. For each, ask whether ordinary code already does it:

- travelling to a known coordinate → waypoint navigation, not RL
- sweeping a region systematically → a geometric pattern, not RL
- stabilizing → a classical controller, not RL

What remains is the candidate learned objective: the part where a fixed pattern structurally cannot win.
One policy per role, never one policy for two roles. `.claude/rules/task-decomposition.md`.

## 2. Budget the episode against the task

Compute what one episode physically buys — distance, time, energy — and what the hardest setting demands.
If the demand exceeds the budget, shrink the task now; that gap is geometry and no learner closes it.
Pick the operating point from the deliverable, then check the platform can reach it.

## 3. Model the platform honestly

A pure, unit-tested module for actuator curves and consumption, from real component data. Determine which
limit binds — peak capability or consumable budget — by measuring, not reasoning. Disturbances as explicit
external forces; per-episode randomization of what you are unsure about.
`.claude/rules/platform-model.md`.

## 4. Design the observation

What must the policy see to act? If the key quantity is a change rather than a level, give it stacked
recent frames plus a little distilled memory, not a computed gradient it could not measure for real.
Measure the sensor's resolution floor and set tolerances from physics, not from difficulty. Keep the
vector small. `.claude/rules/observation-design.md`.

## 5. Write the reward for the deliverable

Count the thing you actually want, one-time for fresh progress, with nothing loiterable and a clear
terminal negative for hard failure. Read the payout from ground truth. Check the per-step arithmetic
against the per-step cost before you believe it. `.claude/rules/reward-design.md`.

## 6. Pick the scorecard, then measure the ceiling and the floors

Choose the metric that is the deliverable and report it through `checkpoints.best_metric`, not average
reward. Write the hand-coded controller that uses only the policy's observation, and run
`eval.py --policy random hold straight`. Those two numbers make every later result interpretable.
`.claude/skills/solvability-ceiling`, `.claude/rules/evaluation.md`.

## 7. Configure, then train the smallest thing that can work

Config from `.claude/skills/new-env`, checked with `.claude/skills/validate-config`. For a long horizon:
`gamma: 0.995`, `n_step: 3`, `explore_hold: 15`, a buffer scaled to the episode length, and a curriculum
in the environment if no episode ever succeeds by chance. One lever per run afterwards
`.claude/rules/experiment-method.md`.

## Report

Give the plan as a short table — role, learned or coded, metric — plus the budget number, the chosen
operating point, and the two reference scores. Name anything that came back infeasible; that is the most
valuable output of this exercise, and it costs no compute.
