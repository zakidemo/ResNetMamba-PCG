"""
resnet1d.py — ResNet-1D baseline for PCG classification.

Standard 1D residual network operating on raw PCG waveforms.
Used as the baseline in Table 1 (ablation) and Table 2 (SOTA comparison).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock1D(nn.Module):
    """Standard ResNet BasicBlock adapted for 1D signals."""

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet1D(nn.Module):
    """
    ResNet-1D baseline (no wavelet, no Mamba).

    Architecture:
        Stem:    Conv1d(1->64, k=7, s=2) -> BN -> ReLU -> MaxPool
        Layer1:  2x BasicBlock1D (64->64,  s=1)
        Layer2:  2x BasicBlock1D (64->128, s=2)
        Layer3:  2x BasicBlock1D (128->256, s=2)
        Layer4:  2x BasicBlock1D (256->512, s=2)
        GAP -> Linear(512->2)

    Args:
        n_classes: Number of output classes (default: 2)
    """

    def __init__(self, n_classes: int = 2):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(64,  64,  n_blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, n_blocks=2, stride=2)

        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(512, n_classes)

    @staticmethod
    def _make_layer(in_ch, out_ch, n_blocks, stride):
        layers = [BasicBlock1D(in_ch, out_ch, stride=stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock1D(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 1, T)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)   # (B, 512)
        return self.classifier(x)       # (B, n_classes)
