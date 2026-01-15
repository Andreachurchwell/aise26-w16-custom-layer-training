import torch
import torch.nn as nn


class LearnedAffine(nn.Module):
    """
    Custom layer: y = x * gamma + beta

    gamma and beta are learnable parameters of shape (dim,).
    Intended input shape: (batch_size, dim)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma + self.beta