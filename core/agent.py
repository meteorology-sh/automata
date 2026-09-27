import copy
import random
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.memory import ReplayBuffer

# In-flight n-step accumulator entry: (state, action, reward, next_state, done).
_NStepEntry = tuple[Any, int, float, Any, bool]


class DQNAgent:
    """
    Epsilon-greedy DQN agent with a target network.

    Uses a slowly-updated copy of the model (target network, Polyak-averaged via
    ``tau``) to supply the next-state value in the Bellman target, which keeps the
    bootstrap target from jumping every gradient step.

    Three mechanisms beyond vanilla DQN are available, each read from config and each
    a no-op at its default, so an existing config keeps its behaviour:

    - ``training.double_dqn`` (default ``true``) — the online network selects the greedy
      next action and the target network scores it. Vanilla DQN takes ``max`` over the
      target network's own Q-values, using one noisy estimate to both pick and score the
      action, which systematically overestimates Q: values inflate and average reward
      peaks then degrades. Set ``false`` for vanilla DQN.
    - ``training.n_step`` (default ``1``) — accumulate the next ``n`` rewards into one
      transition so a distant payoff propagates back ``n`` times faster than 1-step
      bootstrapping. Needed whenever the reward arrives hundreds of steps after the
      actions that earned it. Requires the training loop to call ``end_episode()``.
    - ``training.explore_hold`` (default ``1``) — when exploring, hold the random action
      for this many steps instead of re-drawing every step. Per-step epsilon-dithering
      averages out to near-zero net motion, so it explores *jitter*, not *trajectories*;
      holding produces coherent exploratory manoeuvres. In a task whose reward depends
      on where the agent goes, this is usually the single largest lever.
    """

    def __init__(self, model: nn.Module, config: dict[str, Any]):
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.target_model.eval()
        self.config = config
        self.num_actions = config["model"]["num_actions"]

        self.epsilon = config["epsilon"]["start"]
        self.epsilon_min = config["epsilon"]["min"]
        self.epsilon_decay = config["epsilon"]["decay"]

        self.gamma = config["training"]["gamma"]
        self.batch_size = config["training"]["batch_size"]
        self.tau = config["training"].get("tau", 0.005)
        self.double_dqn = config["training"].get("double_dqn", True)

        # n-step returns. n_step=1 is plain 1-step DQN, identical to before.
        self.n_step = int(config["training"].get("n_step", 1))
        self._nstep_buf: deque[_NStepEntry] = deque()

        # Temporally-extended exploration. 1 = plain per-step epsilon-greedy.
        self.explore_hold = int(config["training"].get("explore_hold", 1))
        self._held_action = 0
        self._hold_steps = 0

        self.memory = ReplayBuffer(config["training"]["memory_size"])
        self.optimizer = optim.Adam(
            model.parameters(), lr=config["training"]["lr"])
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state_tensor: torch.Tensor) -> int:
        # Continue an in-progress exploratory hold — a coherent random manoeuvre.
        if self._hold_steps > 0:
            self._hold_steps -= 1
            return self._held_action
        if random.random() < self.epsilon:
            self._held_action = random.randint(0, self.num_actions - 1)
            self._hold_steps = self.explore_hold - 1  # hold for the remaining steps
            return self._held_action
        with torch.no_grad():
            return self.model(state_tensor).argmax().item()

    def remember(self, state: Any, action: int, reward: float,
                 next_state: Any, done: bool) -> None:
        if self.n_step <= 1:
            self.memory.append(state, action, reward, next_state, done, self.gamma)
            return
        # Accumulate; emit one n-step transition once the window is full, then slide.
        # Partial windows (the last n_step-1 of an episode) are flushed by end_episode().
        self._nstep_buf.append((state, action, reward, next_state, done))
        if len(self._nstep_buf) >= self.n_step:
            self._push_nstep()

    def _push_nstep(self) -> None:
        """Emit the n-step transition starting at the oldest buffered step, then drop it.

        Accumulates the discounted reward over the buffered window, stopping early if a
        terminal is reached, and records gamma^k as the bootstrap discount for the
        landing state (k = number of steps included). Bootstraps through non-terminal
        endings (done=False), matching the truncation convention in the training loop."""
        buf = self._nstep_buf
        s0, a0 = buf[0][0], buf[0][1]
        total_r = 0.0
        discount = 1.0
        next_state = buf[0][3]
        done_flag = False
        for (_s, _a, r, s2, d) in buf:
            total_r += discount * r
            discount *= self.gamma
            next_state, done_flag = s2, d
            if d:
                break
        self.memory.append(s0, a0, total_r, next_state, done_flag, discount)
        buf.popleft()

    def end_episode(self) -> None:
        """Flush any partial n-step windows at episode end (terminal or truncated).

        No-op for 1-step. Draining every episode prevents an n-step return from spanning
        a reset. Must be called once per episode by the training loop."""
        while self._nstep_buf:
            self._push_nstep()
        self._hold_steps = 0  # don't carry an exploratory hold across a reset

    def train(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones, discounts = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        # Per-transition bootstrap discount: gamma for 1-step, gamma^k for n-step.
        discounts = torch.tensor(discounts, dtype=torch.float32)

        current_q = self.model(states).gather(
            1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            if self.double_dqn:
                # Online net selects the greedy next action; target net scores it.
                next_actions = self.model(next_states).argmax(1, keepdim=True)
                next_q = self.target_model(next_states).gather(
                    1, next_actions).squeeze(1)
            else:
                # Vanilla DQN: target net both selects and evaluates via max.
                next_q = self.target_model(next_states).max(1).values
        target_q = rewards + discounts * next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.model.parameters(), clip_value=100)
        self.optimizer.step()

        # Soft update: blend online weights into target network
        for target_p, online_p in zip(self.target_model.parameters(), self.model.parameters()):
            target_p.data.mul_(1 - self.tau).add_(online_p.data * self.tau)

        return loss.item()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
