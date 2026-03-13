"""
WaveletResNet1D — Contribution 1
=================================
A dual-stream ResNet-1D that operates directly on wavelet-decomposed
PCG subbands rather than raw waveform.

Architecture overview
─────────────────────
Raw PCG (B, 1, 16000)
        │
   DWT (db4, J=2)          ← 2-level decomposition
   /          \
 Low (S1/S2)  High (murmurs/noise)
   │               │
ResNet Stream   ResNet Stream   ← independent local feature extractors
   │               │
 GAP            GAP
   └──── concat ────┘
          │
   Cross-band Fusion MLP
          │
      Classifier

Why this beats a vanilla ResNet
────────────────────────────────
1. The wavelet front-end gives the network a physics-informed frequency
   prior: S1/S2 structural components live in the low subband, murmurs
   and pathological clicks live in the high subbands.  A vanilla ResNet
   must learn this separation from scratch via many conv layers.

2. Two-level DWT (J=2) produces three streams:
     yl   : ~0–250 Hz  (heart rate, S1/S2 envelope)
     yh[0]: ~250–1000 Hz (murmur body)         ← only used at J=1
     yh[1]: not used here (too noisy at 2 kHz SR)
   We concatenate yl and yh[0] for the dual-stream setup.

3. Each stream's ResNet is shallower than the baseline (3 stages vs 4),
   keeping total parameters comparable to ResNet1D while benefiting from
   the structural prior.

IEEE contribution framing
──────────────────────────
"We propose WaveletResNet-PCG, a dual-stream convolutional architecture
that replaces the single-channel raw-waveform input of ResNet-1D with
frequency-selective wavelet subbands. The low-frequency stream captures
the morphological S1/S2 cycle while the high-frequency stream specialises
in murmur detection. A learned cross-band fusion gate combines evidence
from both streams for the final classification decision."
"""

import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward


# ─── Building blocks ──────────────────────────────────────────────────────────

class BasicBlock1D(nn.Module):
    """Standard residual block for 1-D signals (identical to ResNet baseline)."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
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
        out = self.relu(out + self.shortcut(x))
        return out


class SubbandResNet(nn.Module):
    """
    Lightweight ResNet stream for a single wavelet subband.
    3 stages (vs 4 in the baseline) to keep param count comparable
    when two streams are combined.
    """
    def __init__(self, in_channels=1, base_ch=32):
        super().__init__()
        # Stem: large kernel to capture temporal structure within the subband
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.in_ch = base_ch
        self.layer1 = self._make_layer(base_ch,      2, stride=1)   # 32
        self.layer2 = self._make_layer(base_ch * 2,  2, stride=2)   # 64
        self.layer3 = self._make_layer(base_ch * 4,  2, stride=2)   # 128

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.out_dim = base_ch * 4   # 128

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
        x = self.layer3(x)
        x = self.gap(x)
        return x.squeeze(-1)   # (B, out_dim)


class CrossBandFusion(nn.Module):
    """
    Gated fusion of two subband feature vectors.

    Instead of simple concatenation + MLP, we use a learned gate that
    weights each stream's contribution — this is the 'cross-band interaction'
    cited as novel in the paper.

    gate = sigmoid(W · [f_low; f_high])
    fused = gate * f_low + (1 - gate) * f_high   [element-wise, after projection]
    """
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.proj_low  = nn.Linear(in_dim, out_dim)
        self.proj_high = nn.Linear(in_dim, out_dim)
        self.gate      = nn.Linear(in_dim * 2, out_dim)
        self.norm      = nn.LayerNorm(out_dim)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, f_low, f_high):
        pl = self.proj_low(f_low)
        ph = self.proj_high(f_high)
        g  = torch.sigmoid(self.gate(torch.cat([f_low, f_high], dim=-1)))
        fused = g * pl + (1.0 - g) * ph
        return self.dropout(self.norm(fused))


# ─── Main model ───────────────────────────────────────────────────────────────

class WaveletResNet1D(nn.Module):
    """
    Dual-stream Wavelet ResNet for PCG classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes (2 for normal/abnormal).
    base_ch : int
        Base channel width for each ResNet stream. Default 32 gives
        ~same total params as the ResNet1D baseline.
    wave : str
        Mother wavelet. 'db4' is standard for cardiac sounds.
    J : int
        DWT decomposition levels. J=1 gives one low + one high subband.
    """
    def __init__(self, num_classes=2, base_ch=32, wave='db4', J=1):
        super().__init__()

        # ── Wavelet front-end (non-trainable, physics-informed) ───────────────
        self.dwt = DWT1DForward(J=J, mode='reflect', wave=wave)

        # ── Dual ResNet streams ───────────────────────────────────────────────
        self.stream_low  = SubbandResNet(in_channels=1, base_ch=base_ch)
        self.stream_high = SubbandResNet(in_channels=1, base_ch=base_ch)

        feat_dim = self.stream_low.out_dim   # 128 with default base_ch=32

        # ── Cross-band gated fusion ───────────────────────────────────────────
        self.fusion = CrossBandFusion(in_dim=feat_dim, out_dim=feat_dim, dropout=0.3)

        # ── Classifier head ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim // 2, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, T)

        # Wavelet decomposition
        yl, yh = self.dwt(x)
        # yl  : (B, 1, T//2)   low-frequency approximation
        # yh[0]: (B, 1, T//2)  high-frequency detail coefficients

        # Each stream expects (B, 1, L)
        f_low  = self.stream_low(yl)          # (B, feat_dim)
        f_high = self.stream_high(yh[0])      # (B, feat_dim)

        # Gated cross-band fusion
        fused = self.fusion(f_low, f_high)    # (B, feat_dim)

        # Classification
        logits = self.classifier(fused)       # (B, num_classes)
        return logits


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Initializing WaveletResNet1D (Contribution 1)...")
    model = WaveletResNet1D(num_classes=2, base_ch=32).cuda()

    # Parameter count
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")

    # Forward pass
    dummy = torch.randn(4, 1, 16000).cuda()
    out   = model(dummy)
    print(f"Output shape: {out.shape}  ✓")
    print("WaveletResNet1D — READY TO TRAIN 🚀")
