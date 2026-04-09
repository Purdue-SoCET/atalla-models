"""Tiny MLP + LayerNorm for graph / validate bring-up."""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNormSmoke(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.ln(x)
