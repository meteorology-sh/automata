# Rule: Keep an experiment record, not a work log

Training runs are expensive and their verdicts are easy to lose. Without a record, the same
lever gets re-tested, a superseded number gets quoted as current, and the reasoning behind a
design is reconstructed from memory. `data/` is that record; `data/INDEX.md` is its table of
contents.

## Must follow

- **One file per milestone finding**, named `NN-short-slug.md`, in this fixed four-part form:
  **research question → input parameters → results as a table → conclusion**. The question is
  one line. The parameters table must contain everything needed to re-run it. The conclusion
  states what is now settled, and — when it generalizes beyond this task — says so explicitly.

- **A finding is a settled result, not a status update.** Write a file when an experiment
  changes what we believe: a lever won, a natural hypothesis was falsified, a ceiling was
  measured, a measurement error was found. Do not write files for routine runs, and do not turn
  a file into a diary of attempts.

- **Record a data status and never invent a number.** Mark each file `complete`; `partial` when
  some cells are prose verdicts or come from a superseded protocol; or `regenerate` when the
  finding holds but the table needs a fresh seeded run. Missing cells are written
  `—not recorded—`. A guessed number is worse than an absent one.

- **State the protocol and the task era.** Both seeds, select and confirm, episode count, difficulty,
  greedy or not, and which version of the task it ran under. When the task definition changes,
  numbers across that boundary are not comparable and each file must say which side it is on.

- **Negative and null results get a file too.** "None of these four levers beat the incumbent"
  is one of the most valuable entries you can write.

- **Update the index and the affected rules in the same pass.** A finding that changes how the
  harness should be used belongs in `.claude/rules/` as well; the file in `data/` is the
  evidence, the rule is the instruction. Keep them consistent.

- **`data/` is a workspace, not documentation.** Revise it freely. Reference documentation
  — `docs/`, `README.md`, `CLAUDE.md`, `.claude/rules/` — describes the system as it *is*. Keep
  the history of how we got here in `data/` and git, not scattered through the docs.
