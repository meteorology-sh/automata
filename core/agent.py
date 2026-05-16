import random
import torch
import torch.nn as nn
import torch.optim as optim

from core.memory import ReplayBuffer


class DQNAgent:
    """
    Epsilon-greedy DQN agent. Selects actions and learns from experience.
    Works with any model that accepts a state tensor and outputs Q-values.
    """

    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.config = config
        self.num_actions = config["model"]["num_actions"]

        self.epsilon = config["epsilon"]["start"]
        self.epsilon_min = config["epsilon"]["min"]
        self.epsilon_decay = config["epsilon"]["decay"]

        self.gamma = config["training"]["gamma"]
        self.batch_size = config["training"]["batch_size"]

        self.memory = ReplayBuffer(config["training"]["memory_size"])
        self.optimizer = optim.Adam(
            model.parameters(), lr=config["training"]["lr"])
        self.loss_fn = nn.MSELoss()

    def select_action(self, state_tensor: torch.Tensor) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        with torch.no_grad():
            return self.model(state_tensor).argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)

    def train(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        current_q = self.model(states).gather(
            1, actions.unsqueeze(1)).squeeze()
        next_q = self.model(next_states).max(1).values
        target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
