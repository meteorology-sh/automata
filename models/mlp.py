import torch.nn as nn


class MLP(nn.Module):
    """
    Decision model for vector-based states (e.g. position, velocity, sensor readings).
    Outputs one raw Q-value per action — no softmax.
    """

    def __init__(self, state_size: int, num_actions: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, x):
        return self.net(x)
