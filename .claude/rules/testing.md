# Testing Rules

## Separate test files per environment
Each environment gets its own test file (e.g. `tests/test_quadrotor_follow.py`). Never append environment-specific tests to `tests/test_smoke.py`. The smoke tests cover framework-level concerns only (replay buffer, agent, models, training utilities). Environment tests live in `tests/test_{env_name}.py`.
