# Testing Rules

## Separate test files per environment
Each environment gets its own test file (e.g. `tests/test_quadrotor_follow.py`). Never append environment-specific tests to `tests/test_smoke.py`. The smoke tests cover framework-level concerns only (replay buffer, agent, models, training utilities). Environment tests live in `tests/test_{env_name}.py`.

## Canonical example is LunarLander
The smoke tests in `tests/test_smoke.py` use LunarLander-v3 (state_size=8, num_actions=4) as the default environment. Use this for any new framework-level tests that need a concrete env.
