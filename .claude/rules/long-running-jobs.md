# Rule: Long-running jobs never block the shell, and never leave the session idle

Training runs and evaluation sweeps take minutes to hours. A foreground call that waits on one
holds the shell open, streams logs nobody reads, blocks the user's next message behind the tool
call, and — when the tool's timeout fires — kills the job's whole process group.

## Must follow

- **Launch every train/eval job with the Bash tool's `run_in_background`, and return
  immediately.** It is non-blocking *and* harness-tracked: when the job exits, the agent is
  re-invoked automatically. Do **not** use `setsid`/`nohup`/`disown` — those orphan the job, so
  no completion event ever fires and the session sits idle until a human comes back. That mistake
  once cost five hours of an authorized overnight session: the runs were fine, the continuity was
  not. Never `sleep`, poll, or `wait` inside the launching call.

- **Announce wall-clock start time and an ETA, before the run, every time.** Format: `"00:14 —
  launched arm_b, 1500 episodes, ETA ~03:10"`. This is the user's only handle on a hardware hang or
  a stalled job without reading logs; a run announced without an ETA is unmonitorable. Say it
  up front, not afterwards, for training runs and evaluation passes alike.

- **Check for early failure once, then stop looking.** One short read of the first lines of the
  log — does it import, does episode 0 print, is the reward finite — is correct and cheap. After
  that, do not tail logs on a timer — poll when the ETA has elapsed or when you have a specific
  question the log can answer.

- **Never read a whole training log.** They are thousands of near-identical lines. Extract what
  you need: `tail -1`, an awk over the metric column, a checkpoint listing. Reporting is driven
  by the evaluation scorecard, not by watching reward tick up.

- **Never end a turn with work outstanding and no wake-up path.** One of these must be true,
  and say which: the job is harness-tracked and will re-invoke you on exit, or a wakeup is
  scheduled at or shortly after the ETA, or something is waiting on a concrete condition. If none
  holds, you have parked the session, not yielded it.

- **Queue the next step before you sleep, not after you wake.** Where the next stage needs no
  judgment, such as the evaluation that always follows a train, launch it as one backgrounded
  command —
  `train && eval` — so completion is a single tracked event. Reserve wakeups for stages that
  genuinely need a decision in between.

- **Keep the shell free while a job runs.** Between launching and the ETA, do work that does not
  block: write the next config, prepare the eval commands, read code. If there is nothing to do,
  say so and yield the turn.

- **A killed job, and an idle stretch, are both reportable events.** Say what died or sat idle,
  at what wall-clock time, and what was lost — then relaunch. Do not quietly restart and let the
  user believe the original ETA still holds.
