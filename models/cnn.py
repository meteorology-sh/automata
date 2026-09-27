import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Perception + decision model for image-based states (e.g. camera frames).
    Expects input shape (batch, 3, H, W), normalized to [0, 1].
    Outputs one raw Q-value per action — no softmax.
    """

    def __init__(self, image_size: int, num_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # → (32, H/2, W/2)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # → (64, H/4, W/4)
        )
        reduced = image_size // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * reduced * reduced, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
