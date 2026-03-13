"""
resnet_mamba.py — ResNet + Mamba (No Wavelet) for PCG Classification
=====================================================================
Ablation model: Raw PCG → Single ResNet (temporal-preserving) → BiMamba → Classifier

No wavelet decomposition — this proves whether the wavelet front-end
adds value on top of the ResNet+Mamba combination.

Architecture:
    Raw PCG (B, 1, 16000)
            │
    Single ResNet Stream (temporal-preserving, no GAP)
            │
    Linear Projection → d_model tokens
            │
    BiMamba Block × n_mamba
            │
    Global Average Pooling
            │
    Classifier Head

Comparison with WaveResNetMamba:
    - WaveResNetMamba:  DWT → 2 streams (each 64ch) → concat(128ch) → proj(128→d_model)
    - This model:       Raw → 1 stream (128ch) → proj(128→d_model)
    Same total ResNet capacity, same Mamba config — only difference is wavelet.
"""

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    from .mamba_ref import MambaRef as Mamba
    print("[INFO] mamba_ssm not found — using MambaRef fallback")


# ─── Building blocks (identical to WaveResNetMamba) ───────────────────────────

class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


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


# ─── Single-Stream Temporal-Preserving ResNet ─────────────────────────────────

class TemporalResNet(nn.Module):
    """
    Single ResNet stream on raw PCG that preserves temporal dimension.
    Matches the total capacity of WaveResNetMamba's dual streams:
        Dual: 2 × (32→64) = 128 channels total
        This: 1 × (64→128) = 128 channels

    Output: (B, 128, T') where T' ≈ 1000
    """
    def __init__(self, in_ch=1, base_ch=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base_ch, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.in_ch = base_ch
        self.layer1 = self._make_layer(base_ch,     2, stride=1)  # 64
        self.layer2 = self._make_layer(base_ch * 2, 2, stride=2)  # 128
        self.out_ch = base_ch * 2  # 128

    def _make_layer(self, out_ch, blocks, stride):
        layers = [BasicBlock1D(self.in_ch, out_ch, stride)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x   # (B, 128, T')


# ─── Main Model ──────────────────────────────────────────────────────────────

class ResNetMamba(nn.Module):
    """
    ResNet + BiMamba (No Wavelet) PCG Classifier.

    Parameters
    ----------
    num_classes : int   — 2 for normal/abnormal
    base_ch     : int   — ResNet base channels (64 → out=128, matching dual-stream total)
    d_model     : int   — Mamba model dimension
    n_mamba     : int   — number of stacked BiMamba blocks
    """
    def __init__(self, num_classes=2, base_ch=64, d_model=128, n_mamba=2):
        super().__init__()

        # Single ResNet stream (temporal-preserving)
        self.resnet = TemporalResNet(in_ch=1, base_ch=base_ch)
        resnet_out_ch = self.resnet.out_ch  # 128

        # Project to d_model (if different from resnet output)
        self.proj = nn.Sequential(
            nn.Conv1d(resnet_out_ch, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

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

        # ResNet feature extraction (temporal-preserving)
        features = self.resnet(x)          # (B, 128, T')

        # Project to d_model
        tokens = self.proj(features)       # (B, d_model, T')

        # Transpose to sequence: (B, T', d_model)
        tokens = tokens.transpose(1, 2)

        # Stacked BiMamba
        for block in self.mamba_blocks:
            tokens = block(tokens)

        tokens = self.final_norm(tokens)

        # Global average pooling
        pooled = tokens.mean(dim=1)        # (B, d_model)

        return self.classifier(pooled)


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Initializing ResNetMamba (ResNet+Mamba, No Wavelet)...")
    model = ResNetMamba(num_classes=2, base_ch=64, d_model=128, n_mamba=2).cuda()

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")

    dummy = torch.randn(4, 1, 16000).cuda()
    out   = model(dummy)
    print(f"Output shape: {out.shape}  ✓")
    print("ResNetMamba — READY ✓")
