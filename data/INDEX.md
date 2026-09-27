# Experiment record in `data/`

One file per **settled finding**, each in the same four-part form: **research question → input
parameters → results → conclusion**. This is the evidence base for the design decisions encoded in
`.claude/rules/` — the rule is the instruction, the file here is why. It is a record of
experiments, not a work log: write a file when a result changes what we believe, not after every
run. Conventions and the template: `.claude/rules/research-record.md` and
`.claude/skills/write-finding`.

Every file carries a **Data status** line: `complete` when the numbers are as measured; `partial`
when some cells are prose verdicts or come from a superseded protocol; `regenerate` when the finding
holds but the table needs a fresh seeded run. Missing cells are written `—not recorded—`, never
guessed.

## The arc

| # | File | Question in one line | Data status |
|---|---|---|---|
| — | — | *no findings recorded yet in this repo* | — |

## Conventions

- **Seeds** — select on one seed, confirm on a disjoint one. A table that mixes them says so.
- **Greedy** — every evaluation number is greedy, with no exploration, unless stated.
- **Floors and ceiling** — report the blind `hold` floor and the hand-coded ceiling next to any
  learned score, and `Hact`, the within-episode action entropy, for any learned policy.
- **Task versions** — when the task definition changes, numbers across that boundary are not
  comparable; each file states which version it ran under.
