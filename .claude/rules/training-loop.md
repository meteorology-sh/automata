---
paths:
  - "core/train.py"
  - "core/visualizer.py"
---

# Training Loop and Visualizer Rules

## Visualizer must not slow training
The visualizer must never add per-step computation to the training loop. No forward passes for Q-value display, no per-step data collection. All data the visualizer displays should be per-episode only (total reward, epsilon). Render frequency should be 1 fps or slower.

## Chart must match the solve condition
The rolling average displayed on the chart must use the same window size as the solve condition (`solve_window` from config). If the chart shows a 20-episode average but the solve condition uses 50, the user sees the average cross the threshold long before training actually stops. What the user sees must directly predict when training will stop.

## Visualization is opt-in
Training runs headless by default. Pass `--vis` to enable the real-time reward dashboard. Evaluation renders by default (`render_mode="human"`); pass `--no-render` for headless eval.

## Path construction
Always use `os.path.join()` for file paths. Never concatenate directory and filename strings — `f"{dir}{name}.pt"` breaks when the directory doesn't have a trailing slash.
