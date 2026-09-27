# Rule: Cadence, reporting, and plain language

How to communicate while running experiments. These are interaction rules, not code rules.

## Must follow

- **Lead with the result.** A short prose summary of a few sentences, the numbers as a **table**,
  then an explicit recommendation or a small set of options. No essays, no lab-notebook
  narration of what you did. The user is deciding, not reading a diary.

- **Report on results, not on every step.** Surface a report when there is a finding worth a
  decision. Between meaningful results, stay quiet or give a one-line status. A multi-step
  sweep gets ONE summary at the end, not a play-by-play.

- **Always state which checkpoint, config, seed, episode count and difficulty produced a
  number**, so any table is reproducible and comparable with the last one. Numbers from
  different protocols are not comparable and must not share a table without saying so.

- **Timestamp and ETA every train/eval pass** — see `.claude/rules/long-running-jobs.md`.

- **Write for an average software engineer, not an RL specialist.** Ordinary maths and CS
  vocabulary is fine — gradient, variance, entropy, discount — but jargon is not when a plain phrase
  works. Prefer "the gap between the hint and the true target" to "residual", "the bonuses along
  a round trip cancel out" to "the shaping telescopes", "% of the hand-coded best-possible
  score" to "%ceil". If a specialized term is genuinely needed, open with a two- or three-row
  term/definition table, then use it freely.

- **Never make the reader decode.** Spell the term out in place rather than leaning on a label
  from an earlier message. When in doubt, plainer wins.

- **Fix trivial problems instead of reporting them.** Broken links, typos, stale paths: just fix
  them. Never end a report with a list of things you noticed and did not act on — either do it,
  or ask the one question whose answer changes the work.
