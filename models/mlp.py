import torch
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingMLP(nn.Module):
    """Dueling MLP for vector states — splits Q into a state value and an advantage.

    A shared trunk feeds two heads: V(s) (one scalar) and A(s,a) (per-action), combined
    as ``Q = V + (A - mean_a A)``. Mean-centring the advantage makes the decomposition
    identifiable and gives the *action-relative* signal its own pathway instead of leaving
    it as a small residual on top of a large common value (Wang et al. 2016).

    Reach for it (``model.dueling: true``) when the Q-values for different actions in a
    state are nearly identical while the state value itself varies a lot — i.e. the action
    preference is being swamped by the value baseline. Note that a small action gap is not
    by itself a defect: what matters is whether the argmax points the right way. Confirm on
    a measurement that can see the behaviour before changing architecture. Outputs raw
    Q-values (no softmax).
    """

    def __init__(self, state_size: int, num_actions: int, hidden_size: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_size, 1)
        self.advantage_head = nn.Linear(hidden_size, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
