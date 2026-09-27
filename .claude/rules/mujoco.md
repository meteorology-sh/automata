# MuJoCo Environment Rules

## Numpy bool casting
MuJoCo `data.qpos` comparisons return `np.bool_`, not Python `bool`. Always wrap termination conditions with `bool()` so the training loop and tests receive a native Python bool.

## MJCF file location
MJCF XML robot definitions live in `models/mjcf/`. The env config references them via `env.mjcf` key.

## Actuator convention
Quadrotor uses `general` actuators with `site` target and `gear="0 0 1 0 0 0"` to apply force in the body-frame z-direction. Control values map directly to force in Newtons.

## Hover thrust computation
Read mass from the loaded model via `model.body_mass[body_id]` rather than hardcoding. This keeps the env self-consistent with whatever MJCF is loaded:
```python
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
hover_thrust = model.body_mass[body_id] * abs(model.opt.gravity[2]) / 4.0
```

## Viewer lifecycle
Create the viewer lazily on first `reset()` when `render_mode="human"`. Call `viewer.sync()` after each `step()`. Close in `close()`. Never create a viewer during training, when `render_mode` is None.

## Camera configuration
Override `_configure_camera()` in each MujocoEnv subclass to set the initial viewport from the environment's parameters: target height, trajectory radius, operating bounds. The base class default uses `model.stat.center` and `model.stat.extent` which are dominated by the ground plane and rarely frame the action well. Compute `cam.lookat`, `cam.distance`, and `cam.elevation` dynamically so the viewport scales with config changes.
