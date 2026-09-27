---
description: Check a training config for common DQN pitfalls before running. Use when training is failing, unstable, or not converging.
argument-hint: <config-path>
arguments: [config_path]
---

Read the config at `$config_path` and check for these known failure modes:

## 1. Learning rate / train_every coupling
If `train_every > 1`, check that `lr` is proportionally reduced. The product `lr * (1 / train_every)` should be around 0.0001. If `train_every: 4` and `lr: 0.001`, flag this — it's 4x too aggressive and will cause Q-value oscillation.

## 2. Epsilon min vs solve threshold
Calculate the expected random failure rate: `epsilon_min * max_episode_steps` is the expected number of random actions per episode. If this exceeds 5, warn that random exploration noise may prevent the rolling average from reaching the solve threshold. Suggest lowering `epsilon_min`.

## 3. Epsilon decay timing
Calculate when epsilon reaches min: find the episode where `epsilon_start * epsilon_decay^episode ≈ epsilon_min`. This should be around 80% of the episode budget. If epsilon bottoms out before 50% of episodes, the agent stops exploring before it has learned enough. If it bottoms out after 95%, there isn't enough greedy exploitation time.

## 4. Solve window size
If `solve_window` is not set, warn that it defaults to 50. If it's set above 100, warn that training may run for hundreds of episodes past convergence. If it's below 20, warn that the solve may trigger on a lucky streak.

## 5. Batch size vs memory size
If `batch_size > memory_size / 10`, warn that the replay buffer won't have enough diversity — the same transitions will be sampled repeatedly.

## 6. Missing keys
Check that these keys exist and flag any that are missing:
- `training.tau` — needed for soft target updates
- `training.train_every` — defaults to 1 if missing, but should be explicit
- `training.solve_window` — defaults to 50 if missing

## 7. Horizon-scaled keys

- `memory_size` should scale with `max_episode_steps`. Flag anything below roughly
  `50 * max_episode_steps` — with short crash episodes overwriting long good ones, a small buffer
  drives a death spiral.
- `gamma` should match the horizon: 0.99 for ~500-step episodes, 0.995 for thousands. Flag a
  0.99 on a multi-thousand-step task, where the payoff at the end is invisible from the start.

## 8. Scorecard consistency

- If `checkpoints.best_metric` is set to something other than `"reward"`, `solve_threshold` must
  be unreachable, such as `1000000`. Otherwise training stops on an average reward that does not
  measure the task. FAIL if both are active.
- If `best_metric` is `"success"`, the environment must set `info["is_success"]`; if it names a
  numeric key, the environment must report that key in `info`. Grep the env for it and FAIL if it
  is absent — the loop silently reads 0.0 for a missing key and `_best.pt` never moves.

## 9. Exploration for long-horizon tasks

- If `max_episode_steps` is in the thousands and `explore_hold` is 1 or missing, WARN: per-step
  epsilon explores jitter, not trajectories.
- If the reward arrives only at the end of a long episode and `n_step` is 1, WARN: credit will
  propagate too slowly. Suggest `n_step: 3`.
- `double_dqn` should be present and true. WARN if explicitly false without a stated reason.

## 10. Reproducibility

- WARN if `env.seed` is unset in a config intended for evaluation — comparisons then run on
  different episodes. Remind the user to select on one seed and confirm on a disjoint one.

Print a summary of findings with PASS/WARN/FAIL for each check.
