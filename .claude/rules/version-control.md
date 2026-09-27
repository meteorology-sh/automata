# Rule: Version control

- **Read git history freely; never commit or push unless asked.** Inspect log, blame and diffs whenever
  it helps. Leave finished work in the working tree and say what changed, so the human decides what
  becomes a commit.

- **Never work in another repository without explicit permission.**

- **Checkpoints and local task files stay out of the repo.** `.gitignore` ships only the base classes,
  the default config and the framework tests; task environments, task configs, MJCF files and weights
  are local. When a new file genuinely belongs in the repo, add its exception explicitly rather than
  loosening a whole directory.
