"""
models/__init__.py — Model registry for all ablation variants.

Ablation Table (capacity-matched):
    resnet1d_matched  : ResNet-1D matched (same capacity as WaveResNetMamba's streams)
    mamba_pcg         : Mamba-only (SSM only, no CNN, no wavelet)
    resnet_mamba      : ResNet + BiMamba (CNN+SSM, no wavelet)
    wavelet_resnet    : WaveletResNet (CNN + wavelet, no Mamba)
    wave_resnet_mamba : WaveResNetMamba (full: wavelet + CNN + SSM)

Also available (original strong baseline, for SOTA table):
    resnet1d          : ResNet-1D full (4 layers, 512 channels)
"""

from .resnet1d          import ResNet1D
from .resnet1d_matched  import ResNet1DMatched
from .wavelet_resnet1d  import WaveletResNet1D
from .wave_resnet_mamba import WaveResNetMamba
from .mamba_pcg         import MambaPCG
from .resnet_mamba      import ResNetMamba


def build_model(name: str, num_classes: int = 2):
    """Factory function — returns an un-initialized-to-device model."""
    registry = {
        "resnet1d":          lambda: ResNet1D(n_classes=num_classes),
        "resnet1d_matched":  lambda: ResNet1DMatched(n_classes=num_classes,
                                                      base_ch=32),
        "mamba_pcg":         lambda: MambaPCG(num_classes=num_classes,
                                              d_model=128, n_mamba=2),
        "resnet_mamba":      lambda: ResNetMamba(num_classes=num_classes,
                                                 base_ch=64, d_model=128,
                                                 n_mamba=2),
        "wavelet_resnet":    lambda: WaveletResNet1D(num_classes=num_classes,
                                                      base_ch=32),
        "wave_resnet_mamba": lambda: WaveResNetMamba(num_classes=num_classes,
                                                      base_ch=32, d_model=128,
                                                      n_mamba=2),
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. "
                         f"Choose from: {list(registry.keys())}")
    return registry[name]()
