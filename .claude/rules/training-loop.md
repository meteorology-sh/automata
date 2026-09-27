---
paths:
  - "core/train.py"
  - "core/visualizer.py"
---

# Training Loop and Visualizer Rules

## Visualizer must not slow training
The visualizer must never add per-step computation to the training loop. No forward passes for Q-value display, no per-step data collection. All data the visualizer displays should be per-episode only: total reward and epsilon. Render frequency should be 1 fps or slower.

## Chart must match the solve condition
The rolling average displayed on the chart must use the same window size as the solve condition, `solve_window` from config. If the chart shows a 20-episode average but the solve condition uses 50, the user sees the average cross the threshold long before training actually stops. What the user sees must directly predict when training will stop.

## Visualization is opt-in
Training runs headless by default. Pass `--vis` to enable the real-time reward dashboard. Evaluation renders by default with `render_mode="human"`; pass `--no-render` for headless eval.

## Path construction
Always use `os.path.join()` for file paths. Never concatenate directory and filename strings — `f"{dir}{name}.pt"` breaks when the directory doesn't have a trailing slash.

## Truncation is not termination

Store the terminated flag `done` only in the replay buffer, and use `done or truncated` for loop
control. A timeout is the clock ending the episode, not the MDP: its target must still
bootstrap `gamma*Q(s')`. Collapsing the two zeroes that bootstrap and puts a systematic
downward bias on exactly the long-horizon states the agent needs to value.

## Flush the agent's episode state

Call `agent.end_episode()` once per episode, after the inner loop. It drains partial n-step
windows, so a return never spans a reset, and clears any in-progress exploratory hold. It is a
no-op for 1-step configs.

## Checkpoint selection stays environment-agnostic

`checkpoints.best_metric` may be `"reward"`, `"success"`, or the name of any numeric `info`
key. The loop maximizes whatever the environment reports without knowing what it means — that
is what keeps it generic. Two invariants: selection on a non-reward metric only becomes
eligible once the rolling window is **full** — otherwise an early lucky streak locks `_best.pt`
to a nearly untrained checkpoint that no later full window can beat — and ties are broken by
average reward.
