"""
models/mamba_ref.py
====================
Pure-PyTorch reference implementation of the Mamba SSM block.
Used as CPU fallback when mamba-ssm (CUDA) is not available.

This is a simplified, non-hardware-optimised version for
compatibility and testing. For full performance, install:
    pip install mamba-ssm --no-build-isolation

Reference:
    Gu & Dao, "Mamba: Linear-Time Sequence Modeling with
    Selective State Spaces," arXiv:2312.00752, 2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MambaRef(nn.Module):
    """Simplified selective SSM (Mamba) — pure PyTorch, no CUDA kernels.

    Implements the S6 selective scan with a sequential loop.
    Slower than the official CUDA implementation but functionally equivalent
    for small batch sizes and short sequences.

    Args:
        d_model : input/output token dimension
        d_state : SSM latent state dimension (default 16)
        d_conv  : local depthwise conv kernel size (default 4)
        expand  : inner dimension = d_model * expand (default 2)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv:  int = 4,
        expand:  int = 2,
    ):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_inner  = d_model * expand

        # ── Input projection ──────────────────────────────────────────────
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # ── Local depthwise conv ──────────────────────────────────────────
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # ── SSM parameters ────────────────────────────────────────────────
        # A: fixed log-diagonal initialisation (HiPPO-inspired)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))   # (d_inner, d_state)

        self.D = nn.Parameter(torch.ones(self.d_inner))  # skip connection

        # Selective parameters: Δ, B, C depend on the input
        self.x_proj = nn.Linear(self.d_inner,
                                 d_state + d_state + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Initialise dt_proj bias so softplus(bias) ≈ dt_init
        dt_init = 0.001
        inv_dt  = math.log(math.expm1(dt_init))
        nn.init.constant_(self.dt_proj.bias, inv_dt)

        # ── Output projection ─────────────────────────────────────────────
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            y: (B, L, d_model)
        """
        B, L, _ = x.shape

        # 1. Input projection + split into (z, x_inner)
        xz = self.in_proj(x)                    # (B, L, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)        # each (B, L, d_inner)

        # 2. Local depthwise conv (causal: trim right padding)
        x_conv = x_inner.transpose(1, 2)        # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[..., :L]   # (B, d_inner, L)
        x_conv = F.silu(x_conv).transpose(1, 2) # (B, L, d_inner)

        # 3. Compute selective SSM parameters from input
        delta_BC = self.x_proj(x_conv)          # (B, L, d_state*2 + 1)
        delta = delta_BC[..., :1]               # (B, L, 1)
        B_sel = delta_BC[..., 1:1+self.d_state] # (B, L, d_state)
        C_sel = delta_BC[..., 1+self.d_state:]  # (B, L, d_state)

        delta = F.softplus(self.dt_proj(delta)) # (B, L, d_inner)

        # 4. Discretise A
        A = -torch.exp(self.A_log.float())      # (d_inner, d_state)
        # dA: (B, L, d_inner, d_state)
        dA = torch.exp(delta.unsqueeze(-1) * A)

        # 5. Selective scan (sequential loop — numerically stable)
        h = torch.zeros(B, self.d_inner, self.d_state,
                        device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            # dB_t: (B, d_inner, d_state)
            dB_t = (delta[:, t, :].unsqueeze(-1) *
                    B_sel[:, t, :].unsqueeze(-2))
            h = dA[:, t, :, :] * h + dB_t * x_conv[:, t, :].unsqueeze(-1)
            # y_t: (B, d_inner)
            y_t = (h * C_sel[:, t, :].unsqueeze(-2)).sum(dim=-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)              # (B, L, d_inner)

        # 6. D skip connection + gate
        y = y + self.D * x_conv
        y = y * F.silu(z)

        # 7. Output projection
        return self.out_proj(y)                 # (B, L, d_model)
