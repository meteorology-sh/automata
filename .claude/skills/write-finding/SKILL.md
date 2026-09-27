---
description: Record a settled experiment result in data/ in the fixed four-part form and update the index. Use after an experiment changes what we believe.
argument-hint: <slug>
arguments: [slug]
---

Write `data/NN-$slug.md`, where `NN` is the next number in `data/INDEX.md`. Findings are settled
results, not status updates: write one when a lever won, a hypothesis was falsified, a ceiling
was measured, or a measurement error was found. Use this template exactly.

```markdown
# NN — <the question, as a question>

**Task version:** <which definition of the task this ran under>. **Data status:** complete |
partial | regenerate — <one line on what is or isn't preserved>.

## Research question

One or two sentences. What did we not know, and why did it matter?

## Input parameters

| Parameter | Value |
|---|---|
| Policy / checkpoint | |
| Config | |
| Metric | |
| Protocol | greedy, N episodes, select seed / confirm seed, horizon |
| The single change under test | |

Everything needed to re-run it. If two arms are compared, one row per arm stating the one
difference.

## Results

A table of measured numbers. Mark anything not preserved `—not recorded—`; never estimate a
number into a results table. Include the blind floor and the hand-coded ceiling where they exist,
and `Hact` for any learned policy.

## Conclusion

What is now settled, in plain language. Then, explicitly, what generalizes beyond this task — the
part a future project would want. Note anything this did *not* fix, and any earlier conclusion
this overturns.
```

Then:

1. Add the row to the table in `data/INDEX.md`: number, file, question in one line, data status.
2. If the finding changes how the harness should be used, update the matching `.claude/rules/`
   file in the same pass — the `data/` file is the evidence, the rule is the instruction.
3. If it overturns an earlier finding, edit that file to point forward. Do not leave two files
   asserting opposite things.
