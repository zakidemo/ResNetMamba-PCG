"""
resnet1d_matched.py — Capacity-Matched ResNet-1D Baseline
==========================================================
ResNet-1D with the SAME architecture as the SubbandResNet streams
used in WaveResNetMamba, but operating on raw PCG (no wavelet).

This ensures a fair ablation: the only difference between this
model and WaveResNetMamba is the wavelet front-end and Mamba —
NOT the ResNet capacity.

Architecture (matches SubbandResNet exactly):
    Raw PCG (B, 1, 16000)
            │
    Stem: Conv1d(1→32, k=7, s=2) → BN → ReLU → MaxPool(3, s=2)
            │
    Layer1: 2× BasicBlock1D (32→32, s=1)
            │
    Layer2: 2× BasicBlock1D (32→64, s=2)
            │
    Global Average Pooling
            │
    Classifier: Linear(64→32) → GELU → Dropout(0.3) → Linear(32→2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock1D(nn.Module):
    """Standard ResNet BasicBlock for 1D signals."""
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


class ResNet1DMatched(nn.Module):
    """
    Capacity-matched ResNet-1D baseline.

    Same depth and channel width as WaveResNetMamba's SubbandResNet:
        Stem → Layer1(32→32) → Layer2(32→64) → GAP → Classifier

    Parameters
    ----------
    n_classes : int — number of output classes (default 2)
    base_ch   : int — base channel width (default 32, matching SubbandResNet)
    """
    def __init__(self, n_classes=2, base_ch=32):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.in_ch = base_ch
        self.layer1 = self._make_layer(base_ch,     2, stride=1)   # 32→32
        self.layer2 = self._make_layer(base_ch * 2, 2, stride=2)   # 32→64

        self.pool = nn.AdaptiveAvgPool1d(1)

        # Same classifier style as WaveResNetMamba
        out_ch = base_ch * 2  # 64
        self.classifier = nn.Sequential(
            nn.Linear(out_ch, out_ch // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(out_ch // 2, n_classes),
        )

    def _make_layer(self, out_ch, blocks, stride):
        layers = [BasicBlock1D(self.in_ch, out_ch, stride)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 1, T)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x).squeeze(-1)    # (B, 64)
        return self.classifier(x)        # (B, n_classes)


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Initializing ResNet1DMatched (capacity-matched baseline)...")
    model = ResNet1DMatched(n_classes=2, base_ch=32).cuda()

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")

    dummy = torch.randn(4, 1, 16000).cuda()
    out   = model(dummy)
    print(f"Output shape: {out.shape}  ✓")
    print("ResNet1DMatched — READY ✓")
