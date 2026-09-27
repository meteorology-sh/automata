---
description: Take one proposed change from hypothesis to verdict — cheap falsification, a single-lever training arm, a seeded scorecard, and a written finding. Use when about to spend a training run on an idea.
argument-hint: <config-path> "<the one change>"
arguments: [config_path, change]
---

One arm tests one lever: `$change` against the current best recipe in `$config_path`. Do not
bundle a second change into it — a losing bundle is uninterpretable and one bad component
poisons every arm that carries it.

## 1. Falsify it for free — before anything else

- **Closed form.** What does the change pay or cost per step? Does a shaping term telescope over
  a round trip? Does the incentive survive a whole episode, not just one leg?
- **Contradiction check.** Does `$change` cut against something in `.claude/rules/`? If so, state
  that rule's reason and show where it breaks, on its terms. If you cannot, stop here.
- **Scripted rollout, no learning.** Drive a fixed trajectory through the environment and print
  the per-step reward with and without the change. Confirm the incentive moves the way you claim.

If any of these fails, you are done — write it up as a negative result and move on. This rung
costs minutes; the training run costs hours.

## 2. Pin the baseline under the identical protocol

```bash
venv/bin/python eval.py --config $config_path --checkpoint checkpoints/<incumbent>.pt \
    --policy checkpoint hold straight --episodes 40 --seed 7000 --no-render --metric <metric>
```

Record the incumbent's number *now*, on this protocol. Never compare against a remembered figure.

## 3. Smoke, then run

Short run first, about 100 episodes, to confirm it imports, episode 0 prints, rewards are finite, and
the metric is being reported. Then launch the full arm **detached** and announce the clock:

```
HH:MM — launched <arm>, N episodes, ETA ~HH:MM
```

Use `run_in_background`, never `setsid` or `nohup`, write logs under the job directory, hand the
shell back, and chain the evaluation onto the same command — `train && eval` — so one tracked event
covers both. See `.claude/rules/long-running-jobs.md`.

## 4. Judge it

Score the arm and the incumbent on the **select** seed, then confirm the winner on a **disjoint**
seed. Report `Hact` alongside the metric — an arm that wins with a collapsed policy has not won.
A one-cell improvement inside seed spread is not a result.

## 5. Write it down either way

Add the finding to `data/` in the four-part form — question → parameters → results table →
conclusion — and update `data/INDEX.md`; see `.claude/skills/write-finding`. If the verdict
changes how the harness should be used, update the relevant `.claude/rules/` file in the same
pass. A null result is a result: record it so the lever is not re-tested.
