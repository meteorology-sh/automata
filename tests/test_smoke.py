import numpy as np
import torch
import yaml

from eval import evaluate
from core.memory import ReplayBuffer
from core.agent import DQNAgent
from core.train import state_to_tensor
from envs.base_env import GymEnv
from main import build_env
from models.mlp import MLP
from models.cnn import CNN


def _cartpole_config():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)
    config["training"]["episodes"] = 1
    config["training"]["solve_threshold"] = 99999
    config["visualizer"]["enabled"] = False
    return config


# --- ReplayBuffer ---

def test_replay_buffer_append_and_len():
    buf = ReplayBuffer(capacity=100)
    assert len(buf) == 0
    buf.append([1, 2], 0, 1.0, [3, 4], False)
    assert len(buf) == 1


def test_replay_buffer_sample():
    buf = ReplayBuffer(capacity=100)
    for i in range(10):
        buf.append([i], 0, 1.0, [i + 1], False)
    batch = buf.sample(5)
    assert len(batch) == 5
    assert all(len(t) == 5 for t in batch)


def test_replay_buffer_capacity():
    buf = ReplayBuffer(capacity=5)
    for i in range(10):
        buf.append([i], 0, 1.0, [i + 1], False)
    assert len(buf) == 5


# --- DQNAgent ---

def test_agent_select_action():
    config = _cartpole_config()
    model = MLP(state_size=4, num_actions=2, hidden_size=32)
    agent = DQNAgent(model, config)
    state = torch.randn(1, 4)
    action = agent.select_action(state)
    assert action in (0, 1)


def test_agent_remember_and_train():
    config = _cartpole_config()
    config["training"]["batch_size"] = 4
    model = MLP(state_size=4, num_actions=2, hidden_size=32)
    agent = DQNAgent(model, config)

    # Not enough samples yet — train returns None
    agent.remember(np.zeros(4), 0, 1.0, np.zeros(4), False)
    assert agent.train() is None

    # Fill buffer past batch_size
    for _ in range(10):
        agent.remember(np.random.randn(4), 0, 1.0, np.random.randn(4), False)
    loss = agent.train()
    assert isinstance(loss, float)


def test_agent_epsilon_decay():
    config = _cartpole_config()
    model = MLP(state_size=4, num_actions=2, hidden_size=32)
    agent = DQNAgent(model, config)
    initial = agent.epsilon
    agent.decay_epsilon()
    assert agent.epsilon < initial


# --- state_to_tensor ---

def test_state_to_tensor_mlp():
    state = np.array([1.0, 2.0, 3.0, 4.0])
    t = state_to_tensor(state, "mlp")
    assert t.shape == (1, 4)


def test_state_to_tensor_cnn():
    state = np.random.rand(84, 84, 3)
    t = state_to_tensor(state, "cnn")
    assert t.shape == (1, 3, 84, 84)


# --- Models ---

def test_mlp_output_shape():
    model = MLP(state_size=4, num_actions=2, hidden_size=32)
    out = model(torch.randn(1, 4))
    assert out.shape == (1, 2)


def test_cnn_output_shape():
    model = CNN(image_size=84, num_actions=4)
    out = model(torch.randn(1, 3, 84, 84))
    assert out.shape == (1, 4)


# --- build_env dispatch ---

def test_build_env_default():
    config = _cartpole_config()
    env = build_env(config)
    assert isinstance(env, GymEnv)
    env.close()


def test_build_env_with_wrapper():
    config = _cartpole_config()
    # Point at the shaped acrobot subclass
    config["env"]["name"] = "Acrobot-v1"
    config["env"]["wrapper"] = "envs.shaped_acrobot.ShapedAcrobot"
    config["model"]["state_size"] = 6
    config["model"]["num_actions"] = 3
    env = build_env(config)
    from envs.shaped_acrobot import ShapedAcrobot
    assert isinstance(env, ShapedAcrobot)
    env.close()


# --- GymEnv full episode ---

def test_gym_env_full_episode():
    config = _cartpole_config()
    env = GymEnv(config)
    state = env.reset()
    assert isinstance(state, np.ndarray)
    assert state.shape == (4,)

    done = False
    steps = 0
    while not done:
        action = 0
        next_state, reward, done, truncated, info = env.step(action)
        done = done or truncated
        steps += 1
        assert isinstance(next_state, np.ndarray)

    assert steps > 0
    env.close()


# --- Eval ---

def test_eval_headless():
    config = _cartpole_config()
    env = GymEnv(config)
    model = MLP(state_size=4, num_actions=2, hidden_size=128)
    rewards = evaluate(env, model, config, num_episodes=2)
    assert len(rewards) == 2
    assert all(isinstance(r, (int, float)) for r in rewards)


def test_gym_env_render_mode_from_config():
    config = _cartpole_config()
    env = GymEnv(config)
    assert env.env.render_mode is None
    env.close()

    config["env"]["render_mode"] = "rgb_array"
    env = GymEnv(config)
    assert env.env.render_mode == "rgb_array"
    env.close()
