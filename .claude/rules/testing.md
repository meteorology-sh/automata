# Testing Rules

## Separate test files per environment
Each environment gets its own test file, such as `tests/test_quadrotor_follow.py`. Never append environment-specific tests to `tests/test_smoke.py`. The smoke tests cover framework-level concerns only: replay buffer, agent, models, training utilities. Environment tests live in `tests/test_{env_name}.py`.

## Canonical example is LunarLander
The smoke tests in `tests/test_smoke.py` use LunarLander-v3 with `state_size=8` and `num_actions=4` as the default environment. Use this for any new framework-level tests that need a concrete env.

## Framework mechanism tests live in tests/test_agent.py

`tests/test_smoke.py` covers wiring: buffer, model shapes, env wrapper, eval path.
`tests/test_agent.py` covers the agent mechanisms whose correctness is arithmetic rather than
behavioural: n-step return accumulation and discounts, terminals cutting the window, no
cross-episode leakage, exploratory holds, the dueling head. Any new learning mechanism gets a
test of the same kind — these are the cheapest possible check that a lever does what its
config key claims, and they run in seconds instead of hours. See
`.claude/rules/experiment-method.md`.
