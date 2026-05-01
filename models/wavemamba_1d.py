import torch
import torch.nn as nn
from mamba_ssm import Mamba
from pytorch_wavelets import DWT1DForward

class BiMambaBlock(nn.Module):
    """A Bi-Directional Mamba Block that reads the signal forwards and backwards."""
    def __init__(self, d_model):
        super().__init__()
        # Forward Mamba
        self.mamba_fwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        # Backward Mamba
        self.mamba_bwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # 1. Forward pass
        out_fwd = self.mamba_fwd(x)
        
        # 2. Backward pass (flip the sequence along the time dimension)
        x_flipped = torch.flip(x, dims=[1])
        out_bwd_flipped = self.mamba_bwd(x_flipped)
        # Flip it back to match the forward sequence
        out_bwd = torch.flip(out_bwd_flipped, dims=[1])
        
        # 3. Fuse the past and future context
        out = out_fwd + out_bwd
        return self.norm(out) + x  # Residual connection


class WaveMamba1D_Classifier(nn.Module):
    def __init__(self, sequence_length, d_model=64, num_classes=2):
        super().__init__()
        
        # 1. Wavelet Decomposition (Isolate structural thuds from murmurs/noise)
        self.dwt1d = DWT1DForward(J=1, mode='zero', wave='db4')
        
        # 2. Linear Projections
        self.proj_low = nn.Linear(1, d_model)
        self.proj_high = nn.Linear(1, d_model)
        
        # 3. BI-DIRECTIONAL Mamba Blocks (The Novel Contribution!)
        self.bi_mamba_low = BiMambaBlock(d_model)
        self.bi_mamba_high = BiMambaBlock(d_model)
        
        # 4. Feature Fusion
        self.fusion_layer = nn.Linear(d_model * 2, d_model)
        self.act = nn.GELU()
        
        # 5. Classifier head with Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # Wavelets
        yl, yh = self.dwt1d(x)
        low_freq = yl.transpose(1, 2)
        high_freq = yh[0].transpose(1, 2)
        
        # Projections
        low_features = self.proj_low(low_freq)
        high_features = self.proj_high(high_freq)
        
        # Bi-Directional Sequence Modeling
        out_low = self.bi_mamba_low(low_features)
        out_high = self.bi_mamba_high(high_features)
        
        # Global Average Pooling
        pooled_low = out_low.mean(dim=1)
        pooled_high = out_high.mean(dim=1)
        
        # Fusion
        fused = torch.cat([pooled_low, pooled_high], dim=1)
        fused = self.act(self.fusion_layer(fused))
        
        # Classification
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        
        return logits

# --- Sanity Check ---
if __name__ == "__main__":
    print("Initializing Bi-Directional WaveMamba Architecture...")
    model = WaveMamba1D_Classifier(sequence_length=16000, d_model=64, num_classes=2).cuda()
    dummy_pcg = torch.randn(4, 1, 16000).cuda()
    out = model(dummy_pcg)
    print(f"Output Shape: {out.shape} -> BI-MAMBA READY TO TRAIN! 🚀")