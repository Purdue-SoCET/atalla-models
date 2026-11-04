"""GPT-2 transformer blocks sized for the 32-wide Atalla tile (issue #66).

Three ``nn.Module``s, all operating on a 2-D ``(N, D)`` activation (batch squeezed)
so that every FX node lowers to an existing graph kernel:

* ``GPT2MLPBlock``  : ``x + proj(gelu(fc(ln(x))))``
* ``GPT2AttnBlock`` : ``x + proj(causal_attention(ln(x)))``
* ``GPT2Block``     : attention block followed by MLP block (pre-LN GPT-2 layout).

Default geometry: ``N=32`` tokens, ``D=32`` model width, one head, MLP hidden ``4*D=128``.

Attention is written **explicitly** (matmul -> scalar mul -> add causal mask
buffer -> softmax -> matmul) instead of ``F.scaled_dot_product_attention(is_causal=True)``
because the existing ``atalla_sdpa`` lowering (``kernels/flash_sdpa_n32d32.c``) is the
non-causal flash kernel.  The causal mask is a registered buffer ``(N, N)`` of
``0`` / ``-1e4`` (bf16-representable; ``-inf`` would produce NaN in the bf16 add path),
so it shows up in the FX graph as a ``get_attr`` constant like a bias.

Ops used and how the pipeline treats them (see docs/gpt2_operator_coverage.md):
  nn.LayerNorm -> F.layer_norm  (emulator layernorm kernel, D % 32 == 0)
  nn.Linear    -> transpose(W) + torch.matmul + operator.add(bias)   (GEMM + add kernels)
  k.transpose  -> call_method transpose                                (passthrough / view)
  s * scale    -> operator.mul(tensor, float)                          (mul kernel)
  s + mask     -> operator.add(tensor, get_attr buffer)                (add kernel)
  F.softmax    -> softmax kernel (row_len == 32)
  F.gelu       -> emit_gelu (PyTorch bf16 fallback unless wired to kernels/gelu.c)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("GPT2MLPBlock", "GPT2AttnBlock", "GPT2Block", "causal_mask")

MASK_NEG = -1.0e4  # bf16-exact (-10000 == 0xC61C), avoids -inf NaN propagation on hardware


def causal_mask(n_tokens: int) -> torch.Tensor:
    """(N, N) additive mask: 0 on/below the diagonal, MASK_NEG above."""
    m = torch.full((n_tokens, n_tokens), MASK_NEG)
    return torch.triu(m, diagonal=1)


class GPT2MLPBlock(nn.Module):
    """x + c_proj(gelu(c_fc(ln_2(x)))) on a 2-D (N, D) activation."""

    def __init__(self, n_tokens: int = 32, dim: int = 32, hidden: int | None = None,
                 gelu_approximate: str = "tanh"):
        super().__init__()
        if dim % 32 != 0:
            raise ValueError("GPT2MLPBlock uses dim divisible by 32 for the hardware LayerNorm.")
        hidden = hidden if hidden is not None else 4 * dim
        self.n_tokens = n_tokens
        self.dim = dim
        self.hidden = hidden
        self.ln_2 = nn.LayerNorm(dim)
        self.c_fc = nn.Linear(dim, hidden)
        self.c_proj = nn.Linear(hidden, dim)
        self.gelu_approximate = gelu_approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_2(x)
        h = self.c_fc(h)
        h = F.gelu(h, approximate=self.gelu_approximate)
        h = self.c_proj(h)
        return x + h


class GPT2AttnBlock(nn.Module):
    """x + c_proj(softmax(q k^T * 1/sqrt(D) + causal_mask) v) with q,k,v = Linear(ln_1(x)).

    Single head; q/k/v are three separate Linear layers (GPT-2's fused c_attn is
    split so each GEMM is a plain (N,D)x(D,D) tile product).
    """

    def __init__(self, n_tokens: int = 32, dim: int = 32):
        super().__init__()
        if dim % 32 != 0:
            raise ValueError("GPT2AttnBlock uses dim divisible by 32 for the hardware LayerNorm.")
        self.n_tokens = n_tokens
        self.dim = dim
        self.scale = 1.0 / math.sqrt(float(dim))
        self.ln_1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.c_proj = nn.Linear(dim, dim)
        self.register_buffer("causal_mask", causal_mask(n_tokens), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_1(x)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        s = torch.matmul(q, k.transpose(-1, -2))   # (N, N) scores
        s = s * self.scale                          # scalar mul
        s = s + self.causal_mask                    # additive causal mask (get_attr)
        p = F.softmax(s, dim=-1)                    # row softmax, row_len = N
        a = torch.matmul(p, v)                      # (N, D)
        a = self.c_proj(a)
        return x + a


class GPT2Block(nn.Module):
    """Full pre-LN GPT-2 block: attention sub-block then MLP sub-block."""

    def __init__(self, n_tokens: int = 32, dim: int = 32, hidden: int | None = None):
        super().__init__()
        self.attn = GPT2AttnBlock(n_tokens=n_tokens, dim=dim)
        self.mlp = GPT2MLPBlock(n_tokens=n_tokens, dim=dim, hidden=hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.mlp(x)
        return x


if __name__ == "__main__":
    # Sanity check against F.scaled_dot_product_attention(is_causal=True).
    torch.manual_seed(0)
    blk = GPT2AttnBlock()
    x = torch.randn(32, 32)
    h = blk.ln_1(x)
    q, k, v = blk.q_proj(h), blk.k_proj(h), blk.v_proj(h)
    ref = F.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=True)[0]
    s = torch.matmul(q, k.transpose(-1, -2)) * blk.scale + blk.causal_mask
    mine = torch.matmul(F.softmax(s, dim=-1), v)
    print("max|explicit - sdpa(is_causal)| =", (mine - ref).abs().max().item())
    print(GPT2Block()(x).shape)
