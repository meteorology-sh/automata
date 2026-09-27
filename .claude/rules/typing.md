# Rule: The harness is strict-typed, at zero errors

`pyrightconfig.json` runs pyright in strict mode over `core`, `envs`, `models`, `tests`, `main.py` and
`eval.py`. Zero errors is the standard, not an aspiration: a harness that silently passes a float where a
tensor belongs, or a `dict` whose shape nobody has written down, fails hours into a run instead of at
edit time.

## Must follow

- **Run it from the project root, with the venv interpreter named explicitly:**

  ```bash
  venv/bin/python -m pyright --pythonpath venv/bin/python
  ```

  Without `--pythonpath` pyright resolves a different interpreter and invents dozens of phantom import
  errors. Run it before handing work back, the same way you run the tests.

- **Config dicts are `dict[str, Any]`.** A YAML config is genuinely dynamic, so type it honestly at the
  boundary and read keys with the same names the configs use. Never spread `Any` further than the
  boundary that earns it.

- **Type the environment contract once and reuse it.** `envs/base_env.py` defines
  `StepResult = tuple[npt.NDArray[Any], float, bool, bool, dict[str, Any]]`. Every `step()` returns that
  alias, so a subclass that returns a bare tuple or forgets the info dict fails the check rather than the
  run. Observations are `npt.NDArray[Any]`.

- **Contain an untyped third-party boundary in one place.** MuJoCo ships no stubs, so `envs/mujoco_env.py`
  aliases the module as `_mj: Any` at the top and uses that alias throughout; Gymnasium's loosely typed
  returns are `cast(...)` once, at the wrapper. Prefer a single alias or cast at the seam over scattering
  ignores through the file, and keep `# type: ignore` to the rare attribute a library really does not
  declare, with the specific rule named.

- **Fix the code, not the checker.** Do not loosen `pyrightconfig.json`, delete an annotation, or reach
  for a blanket ignore to make an error go away. The suppressions already in the config are limited to
  unknown *member*, *variable* and *argument* types from untyped libraries; anything beyond that needs a
  stated reason.

- **Annotate new code as you write it**, including tests: parameters, return types, and the element type
  of every container such as `deque[float]` or `Counter[int]`. A helper that builds a config returns
  `dict[str, Any]`, or every later assignment into it fails on an inferred literal type.
