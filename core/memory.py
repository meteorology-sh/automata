import random
from collections import deque
from typing import Any

# (state, action, reward, next_state, done, discount). `reward` is the (possibly n-step)
# accumulated return and `discount` is the factor to apply to the bootstrap value of
# `next_state`: gamma for a 1-step transition, gamma^n for a full n-step transition,
# gamma^k for a partial window flushed at the end of an episode. Storing the discount
# per transition is what lets 1-step and n-step samples share one buffer.
Transition = tuple[Any, int, float, Any, bool, float]


class ReplayBuffer:
    """
    Stores past experiences and samples random batches for training.
    Random sampling breaks temporal correlation between consecutive experiences.
    """

    def __init__(self, capacity: int):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def append(self, state: Any, action: int, reward: float, next_state: Any,
               done: bool, discount: float = 1.0) -> None:
        self.buffer.append((state, action, reward, next_state, done, discount))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
