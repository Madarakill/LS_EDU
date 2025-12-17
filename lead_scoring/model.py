from __future__ import annotations

import torch
from torch import nn

# Модель нейронной сети
class LeadMLP(nn.Module):
    def __init__(self, input_layer: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_layer, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)

