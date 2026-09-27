"""Framework tests for the DQN agent mechanisms beyond vanilla DQN.

Covers n-step return accumulation (correct discounting, terminals cutting the window, no
leakage across a reset), temporally-extended exploration, and the dueling head. Each
mechanism must be a no-op at its default so existing configs are unaffected.
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from core.agent import DQNAgent


def _make_agent(n_step: int, gamma: float = 0.9) -> DQNAgent:
    config: dict[str, Any] = {
        "model": {"num_actions": 3},
        "epsilon": {"start": 0.0, "min": 0.0, "decay": 1.0},
        "training": {
            "gamma": gamma, "batch_size": 8, "lr": 1e-3, "memory_size": 1000,
            "tau": 0.005, "double_dqn": True, "n_step": n_step,
        },
    }
    model = nn.Linear(4, 3)  # 4-D state → 3 actions
    return DQNAgent(model, config)


def _s(i: int) -> np.ndarray:
    return np.full(4, float(i), dtype=np.float32)


def test_one_step_is_unchanged() -> None:
    """n_step=1 stores each transition immediately with discount == gamma."""
    agent = _make_agent(n_step=1, gamma=0.9)
    agent.remember(_s(0), 1, 0.5, _s(1), False)
    agent.end_episode()
    assert len(agent.memory) == 1
    _st, a, r, _ns, done, disc = agent.memory.buffer[0]
    assert a == 1 and r == 0.5 and done is False and disc == 0.9


def test_three_step_return_and_discount() -> None:
    """A full 3-step window: R = r0 + g*r1 + g^2*r2, bootstrap discount g^3, lands on s3."""
    g = 0.9
    agent = _make_agent(n_step=3, gamma=g)
    for i, r in enumerate([1.0, 2.0, 3.0, 4.0]):
        agent.remember(_s(i), 0, r, _s(i + 1), False)
    # After 3 appends the first window emits; after the 4th, the second emits.
    assert len(agent.memory) == 2
    _st, _a, R0, ns0, done0, disc0 = agent.memory.buffer[0]
    assert np.isclose(R0, 1.0 + g * 2.0 + g * g * 3.0)
    assert np.isclose(disc0, g ** 3) and done0 is False
    assert np.array_equal(ns0, _s(3))  # bootstraps from s3 (3 steps ahead of s0)


def test_terminal_cuts_window_and_flushes() -> None:
    """A terminal inside the window truncates the return and bootstraps as terminal."""
    g = 0.9
    agent = _make_agent(n_step=3, gamma=g)
    agent.remember(_s(0), 0, 1.0, _s(1), False)
    agent.remember(_s(1), 0, 2.0, _s(2), True)   # terminal on step 2
    agent.end_episode()
    assert len(agent.memory) == 2                # two start states, both windows cut
    _st0, _a0, R0, ns0, done0, disc0 = agent.memory.buffer[0]
    assert np.isclose(R0, 1.0 + g * 2.0) and done0 is True
    assert np.isclose(disc0, g ** 2) and np.array_equal(ns0, _s(2))
    _st1, _a1, R1, _ns1, done1, disc1 = agent.memory.buffer[1]
    assert np.isclose(R1, 2.0) and done1 is True and np.isclose(disc1, g)


def test_no_cross_episode_leak() -> None:
    """end_episode() drains the accumulator so a return never spans a reset."""
    agent = _make_agent(n_step=5, gamma=0.9)
    for i in range(3):  # 3 < n_step, so nothing emits until the flush
        agent.remember(_s(i), 0, 1.0, _s(i + 1), False)
    assert len(agent.memory) == 0
    agent.end_episode()
    assert len(agent.memory) == 3          # all three partial windows flushed
    assert len(agent._nstep_buf) == 0      # accumulator drained
    agent.remember(_s(0), 0, 1.0, _s(1), False)
    agent.end_episode()
    assert len(agent.memory) == 4          # next episode starts clean


def test_explore_hold_holds_action() -> None:
    """With explore_hold>1 and epsilon=1, an exploratory action is held for k steps."""
    config: dict[str, Any] = {
        "model": {"num_actions": 7},
        "epsilon": {"start": 1.0, "min": 1.0, "decay": 1.0},  # always explore
        "training": {
            "gamma": 0.9, "batch_size": 8, "lr": 1e-3, "memory_size": 100,
            "tau": 0.005, "double_dqn": True, "explore_hold": 5,
        },
    }
    agent = DQNAgent(nn.Linear(4, 7), config)
    st = torch.zeros(1, 4)
    actions = [agent.select_action(st) for _ in range(5)]
    assert len(set(actions)) == 1   # all five steps hold the same random action
    agent.end_episode()
    assert agent._hold_steps == 0   # hold cleared at the episode boundary


def test_explore_hold_default_is_per_step() -> None:
    """explore_hold defaults to 1 → no hold state accumulates (unchanged behaviour)."""
    agent = _make_agent(n_step=1)
    assert agent.explore_hold == 1
    agent.epsilon = 1.0
    agent.select_action(torch.zeros(1, 4))
    assert agent._hold_steps == 0   # never holds


def test_double_dqn_is_default_and_optional() -> None:
    """Double DQN is on by default and can be switched off from config."""
    assert _make_agent(n_step=1).double_dqn is True
    config: dict[str, Any] = {
        "model": {"num_actions": 3},
        "epsilon": {"start": 0.0, "min": 0.0, "decay": 1.0},
        "training": {"gamma": 0.9, "batch_size": 2, "lr": 1e-3, "memory_size": 10,
                     "tau": 0.005, "double_dqn": False},
    }
    agent = DQNAgent(nn.Linear(4, 3), config)
    assert agent.double_dqn is False
    for i in range(4):
        agent.remember(_s(i), 0, 1.0, _s(i + 1), False)
    assert isinstance(agent.train(), float)   # vanilla path still trains


def test_dueling_head_shape_and_routing() -> None:
    """The dueling MLP outputs one raw Q per action (mean-centred advantage), and
    build_model routes on model.dueling."""
    from models.mlp import MLP, DuelingMLP
    from core.train import build_model

    net = DuelingMLP(state_size=10, num_actions=7)
    q = net(torch.randn(4, 10))
    assert q.shape == (4, 7)                 # (batch, num_actions), raw Q-values
    # Q = V + (A - mean A); with equal advantages Q collapses to V across actions.
    with torch.no_grad():
        net.advantage_head.weight.zero_()
        net.advantage_head.bias.zero_()
        q2 = net(torch.randn(1, 10))
    assert torch.allclose(q2, q2[:, :1].expand_as(q2), atol=1e-6)

    base = {"type": "mlp", "state_size": 10, "num_actions": 7, "hidden_size": 16}
    assert isinstance(build_model({"model": {**base, "dueling": True}}), DuelingMLP)
    assert isinstance(build_model({"model": base}), MLP)  # default off = plain MLP
