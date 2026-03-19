from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class ClassificationHead(nn.Module):
    """Standard classification head used in fine-tuning."""

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = activation or nn.Tanh()
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(features)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.out_proj(x)


class RegressionHead(nn.Module):
    """Regression head for single or multi-target outputs."""

    def __init__(
        self,
        hidden_size: int,
        num_targets: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.out_proj = nn.Linear(hidden_size, num_targets)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(features)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.out_proj(x)
