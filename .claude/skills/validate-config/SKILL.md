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
Calculate when epsilon reaches min: find the episode where `epsilon_start * epsilon_decay^episode ≈ epsilon_min`. This should be around 80% of the episode budget. If epsilon bottoms out too early (< 50% of episodes), the agent stops exploring before it has learned enough. If too late (> 95%), there isn't enough greedy exploitation time.

## 4. Solve window size
If `solve_window` is not set, warn that it defaults to 50. If it's set above 100, warn that training may run for hundreds of episodes past convergence. If it's below 20, warn that the solve may trigger on a lucky streak.

## 5. Batch size vs memory size
If `batch_size > memory_size / 10`, warn that the replay buffer won't have enough diversity — the same transitions will be sampled repeatedly.

## 6. Missing keys
Check that these keys exist and flag any that are missing:
- `training.tau` (needed for soft target updates)
- `training.train_every` (defaults to 1 if missing, but should be explicit)
- `training.solve_window` (defaults to 50 if missing)

Print a summary of findings with PASS/WARN/FAIL for each check.
