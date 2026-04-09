"""Minimal ViT-style stack: LayerNorm, single-head attention (matmul+softmax), GELU FFN.

Shapes are chosen so LayerNorm uses the AtallaC path (``D % 32 == 0``): ``dim=32``,
``n_tokens=4``. Example input: ``(1, 4, 32)`` — treat as patch tokens after embedding.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTMicro(nn.Module):
    def __init__(self, dim: int = 32, n_tokens: int = 4):
        super().__init__()
        if dim % 32 != 0:
            raise ValueError("ViTMicro uses dim divisible by 32 for hardware LayerNorm.")
        self.dim = dim
        self.n_tokens = n_tokens
        self.patch_embed = nn.Linear(dim, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, dim))
        self.ln1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn_proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        hidden = dim * 2
        self.ff1 = nn.Linear(dim, hidden)
        self.ff2 = nn.Linear(hidden, dim)
        self.scale = 1.0 / math.sqrt(float(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        x = self.patch_embed(x) + self.pos_embed
        xa = self.ln1(x)
        q = self.q_proj(xa)
        k = self.k_proj(xa)
        v = self.v_proj(xa)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        mixed = torch.matmul(attn, v)
        x = x + self.attn_proj(mixed)
        xb = self.ln2(x)
        h = self.ff1(xb)
        h = F.gelu(h, approximate="tanh")
        x = x + self.ff2(h)
        return x
