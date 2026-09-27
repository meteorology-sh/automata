from abc import ABC, abstractmethod
from typing import Any, cast

import gymnasium as gym
import numpy as np
import numpy.typing as npt

# Return type of an environment step: (obs, reward, done, truncated, info)
StepResult = tuple[npt.NDArray[Any], float, bool, bool, dict[str, Any]]


class BaseEnv(ABC):
    """
    Abstract base class for all environments in the RL harness.
    Subclass this to add a new environment — implement reset(), step(), and get_state().
    The reward function belongs inside step(), not in the training loop.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.episode_step = 0
        self.max_steps = config["env"]["max_episode_steps"]
        # Optional reproducible episode seeding. `env.seed` in config fixes the sequence of
        # episodes, so an evaluation run is repeatable and two policies can be compared on
        # the *same* episodes. Always select on one seed and confirm on a disjoint one —
        # reporting on the seeds a checkpoint was chosen on is winner's curse.
        self.seed = config["env"].get("seed")
        self.episode_index = -1
        self.rng = np.random.default_rng(self.seed)

    def _begin_episode(self) -> int | None:
        """Call at the top of reset(). Advances the episode counter, reseeds self.rng, and
        returns this episode's seed (None when env.seed is unset). Subclasses should draw
        every per-episode randomization from self.rng so the episode is reproducible."""
        self.episode_index += 1
        episode_seed = None if self.seed is None else int(self.seed) + self.episode_index
        self.rng = np.random.default_rng(episode_seed)
        return episode_seed

    @abstractmethod
    def reset(self) -> npt.NDArray[Any]:
        """
        Reset the environment to an initial state.
        Returns the initial state as a numpy array.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: int) -> StepResult:
        """
        Apply an action to the environment.
        Returns (next_state, reward, done, truncated, info).
        Compute and return the reward here.
        """
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> npt.NDArray[Any]:
        """
        Return the current state as a numpy array.
        For image-based envs, return shape (H, W, 3).
        For vector-based envs, return shape (state_size,).
        """
        raise NotImplementedError

    def close(self) -> None:
        """Optional cleanup. Override if your simulator needs explicit teardown."""
        pass

    def _check_truncated(self) -> bool:
        """Returns True if the episode has exceeded max_steps."""
        return self.episode_step >= self.max_steps


class GymEnv(BaseEnv):
    """
    Thin wrapper around any standard Gymnasium environment.
    Use this for built-in envs like CartPole — no subclassing needed.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        render_mode = config["env"].get("render_mode")
        self.env = gym.make(config["env"]["name"], render_mode=render_mode)

    def reset(self) -> npt.NDArray[Any]:
        self.episode_step = 0
        state, _ = self.env.reset(seed=self._begin_episode())
        return cast(npt.NDArray[Any], state)

    def step(self, action: int) -> StepResult:
        self.episode_step += 1
        next_state, reward, done, truncated, info = self.env.step(action)
        truncated = truncated or self._check_truncated()
        return (cast(npt.NDArray[Any], next_state), float(reward),
                bool(done), bool(truncated), info)

    def get_state(self) -> npt.NDArray[Any]:
        return cast(npt.NDArray[Any], self.env.unwrapped.state)  # type: ignore[attr-defined]

    def close(self) -> None:
        self.env.close()
