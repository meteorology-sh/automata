# Rule: Docs describe the system as it is, and stay consistent

Docs here means anything under a `docs/` directory, plus `CLAUDE.md`, `README.md` and the files in
`.claude/rules/`. They are the settled account of the system, and they are only useful while they are
true.

## Must follow

- **Ask before creating, moving, renaming or deleting a doc.** Read them freely. Editing one to correct
  it alongside the change that made it wrong is normal; restructuring the set is a decision to confirm
  first.

- **The whole set reads as one account.** No doc may contradict another, or contradict the code, configs
  and shipped checkpoint. When a code change makes a doc wrong, either fix the doc in the same pass or
  say plainly which sentence is now false — never leave it to drift, and never leave two docs asserting
  opposite things.

- **A doc is not a changelog.** Describe the current design and why it is what it is. Do not accumulate
  "previously we did X", retired-model comparisons, or dated progress narratives. The record of how we
  got here lives in git history and, for research results, in `data/`. Prefer replacing superseded
  content over stacking a new layer beside it.

- **Findings go in `data/`, instructions go in `.claude/rules/`.** The experiment file is the evidence;
  the rule is what to do about it. When a result changes practice, update both in the same pass.
