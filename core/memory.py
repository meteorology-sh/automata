import random
from collections import deque


class ReplayBuffer:
    """
    Stores past experiences and samples random batches for training.
    Random sampling breaks temporal correlation between consecutive experiences.
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def append(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
