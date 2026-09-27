import time
from typing import Any

import mujoco
import mujoco.viewer
import numpy.typing as npt

from envs.base_env import BaseEnv, StepResult

# mujoco ships no type stubs; alias as Any so its C-extension symbols type-check.
_mj: Any = mujoco
_mj_viewer: Any = mujoco.viewer


class MujocoEnv(BaseEnv):
    """
    Base class for environments using MuJoCo as the physics backend.

    Subclasses provide an MJCF XML file (via config["env"]["mjcf"]) and implement:
        _get_obs()            — observation vector from simulator state
        _get_reward()         — scalar reward
        _is_done()            — termination condition
        _apply_action(action) — map discrete action to self.data.ctrl

    Optionally override:
        _reset_state()        — randomize initial qpos/qvel
        _get_info()           — extra info dict
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        mjcf_path = config["env"]["mjcf"]
        self.model: Any = _mj.MjModel.from_xml_path(mjcf_path)
        self.data: Any = _mj.MjData(self.model)
        self.render_mode = config["env"].get("render_mode")
        self.n_substeps = config["env"].get("n_substeps", 10)
        self._viewer: Any = None

    def reset(self) -> npt.NDArray[Any]:
        self.episode_step = 0
        self._begin_episode()
        _mj.mj_resetData(self.model, self.data)
        self._reset_state()
        _mj.mj_forward(self.model, self.data)
        if self.render_mode == "human" and self._viewer is None:
            self._viewer = _mj_viewer.launch_passive(self.model, self.data)
            self._configure_camera()
        return self._get_obs()

    def step(self, action: int) -> StepResult:
        self.episode_step += 1
        self._apply_action(action)
        for _ in range(self.n_substeps):
            _mj.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self._get_reward()
        done = self._is_done()
        truncated = self._check_truncated()
        info = self._get_info()
        if self.render_mode == "human" and self._viewer is not None:
            self._viewer.sync()
            # Pace to real time so the viewer is watchable
            time.sleep(self.model.opt.timestep * self.n_substeps)
        return obs, reward, done, truncated, info

    def get_state(self) -> npt.NDArray[Any]:
        return self._get_obs()

    def _configure_camera(self) -> None:
        """Set initial camera to frame the full scene. Override for custom views."""
        if self._viewer is None:
            return
        cam = self._viewer.cam
        cam.lookat[:] = self.model.stat.center
        cam.distance = self.model.stat.extent * 1.5
        cam.elevation = -20
        cam.azimuth = 135

    def _reset_state(self) -> None:
        """Override to set initial qpos/qvel with randomization, drawn from self.rng."""
        pass

    def _apply_action(self, action: int) -> None:
        raise NotImplementedError

    def _get_obs(self) -> npt.NDArray[Any]:
        raise NotImplementedError

    def _get_reward(self) -> float:
        raise NotImplementedError

    def _is_done(self) -> bool:
        raise NotImplementedError

    def _get_info(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
