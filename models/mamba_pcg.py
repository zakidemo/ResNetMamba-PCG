"""
mamba_pcg.py — Mamba-Only Baseline for PCG Classification
==========================================================
Ablation model: Raw PCG → Linear Projection → Stacked BiMamba → Classifier

No wavelet, no ResNet — pure SSM on raw PCG.
This is the first application of Mamba/SSM to PCG signals.

Architecture:
    Raw PCG (B, 1, 16000)
            │
    Patch Embedding (Conv1d stem, temporal downsampling)
            │
    Linear Projection → d_model tokens
            │
    BiMamba Block × n_mamba
            │
    Global Average Pooling
            │
    Classifier Head
"""

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    from .mamba_ref import MambaRef as Mamba
    print("[INFO] mamba_ssm not found — using MambaRef fallback")


# ─── Bidirectional Mamba block (shared with WaveResNetMamba) ──────────────────

class BiMambaBlock(nn.Module):
    """Bidirectional Mamba with pre-norm and residual."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm      = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state,
                               d_conv=d_conv, expand=expand)

    def forward(self, x):
        x_norm = self.norm(x)
        fwd    = self.mamba_fwd(x_norm)
        bwd    = torch.flip(
            self.mamba_bwd(torch.flip(x_norm, dims=[1])), dims=[1]
        )
        return x + fwd + bwd


# ─── Patch Embedding Stem ─────────────────────────────────────────────────────

class PatchEmbedding1D(nn.Module):
    """
    Convert raw PCG waveform into a sequence of tokens via conv stem.
    Downsamples temporally to produce ~500-1000 tokens for Mamba.

    Raw PCG (B, 1, 16000)
        → Conv1d(1, d_model, k=7, s=2) → BN → GELU    (B, d_model, 8000)
        → Conv1d(d_model, d_model, k=3, s=2) → BN → GELU  (B, d_model, 4000)
        → Conv1d(d_model, d_model, k=3, s=2) → BN → GELU  (B, d_model, 2000)
        → Conv1d(d_model, d_model, k=3, s=2) → BN → GELU  (B, d_model, 1000)
    """
    def __init__(self, d_model=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

    def forward(self, x):
        # x: (B, 1, T) → (B, d_model, T')
        return self.stem(x)


# ─── Main Model ──────────────────────────────────────────────────────────────

class MambaPCG(nn.Module):
    """
    Mamba-Only PCG Classifier.

    Parameters
    ----------
    num_classes : int   — 2 for normal/abnormal
    d_model     : int   — Mamba model dimension (default 128)
    n_mamba     : int   — number of stacked BiMamba blocks (default 2)
    """
    def __init__(self, num_classes=2, d_model=128, n_mamba=2):
        super().__init__()

        # Patch embedding: raw PCG → token sequence
        self.patch_embed = PatchEmbedding1D(d_model=d_model)

        # Stacked BiMamba
        self.mamba_blocks = nn.ModuleList([
            BiMambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_mamba)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, 16000)

        # Patch embedding → (B, d_model, T')
        tokens = self.patch_embed(x)

        # Transpose to sequence: (B, T', d_model)
        tokens = tokens.transpose(1, 2)

        # Stacked BiMamba
        for block in self.mamba_blocks:
            tokens = block(tokens)

        tokens = self.final_norm(tokens)

        # Global average pooling
        pooled = tokens.mean(dim=1)   # (B, d_model)

        return self.classifier(pooled)


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Initializing MambaPCG (Mamba-Only Ablation)...")
    model = MambaPCG(num_classes=2, d_model=128, n_mamba=2).cuda()

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")

    dummy = torch.randn(4, 1, 16000).cuda()
    out   = model(dummy)
    print(f"Output shape: {out.shape}  ✓")
    print("MambaPCG — READY ✓")
