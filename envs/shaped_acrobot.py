import numpy as np
import gymnasium as gym
from envs.base_env import BaseEnv


class ShapedAcrobot(BaseEnv):
    """
    Acrobot-v1 with a shaped reward function.

    The default Acrobot reward is -1 per step (sparse signal — the agent only
    learns that shorter episodes are better). This subclass adds a dense reward
    component based on the height of the tip of the lower link, giving the
    agent a gradient to follow before it discovers the swing-up solution.

    Reward = base_reward + height_bonus
      - base_reward: -1 per step (unchanged from Gymnasium)
      - height_bonus: tip_height normalized to [0, 1], so total reward per
        step is in [-1, 0]. Clipped to [-1, 1] per CLAUDE.md guidelines.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        render_mode = config["env"].get("render_mode")
        self.env = gym.make("Acrobot-v1", render_mode=render_mode)
        self._state = None

    def reset(self) -> np.ndarray:
        self.episode_step = 0
        state, _ = self.env.reset()
        self._state = state
        return state

    def step(self, action: int) -> tuple:
        self.episode_step += 1
        next_state, reward, done, truncated, info = self.env.step(action)
        truncated = truncated or self._check_truncated()

        # Shape the reward: encourage height of the tip of the lower link.
        # State: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot]
        # Tip height = -cos(θ1) - cos(θ1 + θ2)  (ranges from -2 to +2)
        cos_theta1 = next_state[0]
        sin_theta1 = next_state[1]
        cos_theta2 = next_state[2]
        sin_theta2 = next_state[3]
        # cos(θ1+θ2) = cos(θ1)cos(θ2) - sin(θ1)sin(θ2)
        cos_sum = cos_theta1 * cos_theta2 - sin_theta1 * sin_theta2
        tip_height = -cos_theta1 - cos_sum  # range [-2, 2]

        # Normalize to [0, 1] and add as bonus
        height_bonus = (tip_height + 2.0) / 4.0
        shaped_reward = reward + height_bonus

        # Clip to [-1, 1] per project guidelines
        shaped_reward = float(np.clip(shaped_reward, -1.0, 1.0))

        self._state = next_state
        return next_state, shaped_reward, done, truncated, info

    def get_state(self) -> np.ndarray:
        return self._state

    def close(self):
        self.env.close()
