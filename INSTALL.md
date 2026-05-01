# Installation

Recommended environment:

- Python 3.10
- CUDA 12.1
- PyTorch 2.5.1 + cu121
- NVIDIA GPU

## Create environment

```bash
conda create -n pcg_mamba_ieee python=3.10 -y
conda activate pcg_mamba_ieee
python -m pip install --upgrade pip setuptools wheel ninja packaging
conda install -c nvidia cuda-toolkit=12.1 -y
```

## Install PyTorch

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

## Install Mamba dependencies

```bash
pip install causal-conv1d==1.5.0.post8 --no-build-isolation
pip install mamba-ssm==2.2.4 --no-build-isolation
pip install transformers==4.36.2 tokenizers==0.15.2
```

## Install remaining packages

```bash
pip install numpy scipy scikit-learn pandas matplotlib tqdm librosa soundfile PyWavelets einops psutil statsmodels "setuptools<81"
pip install pytorch-wavelets
```

## Test installation

```bash
python - << 'PY'
import torch
from mamba_ssm import Mamba
from pytorch_wavelets import DWT1DForward
from models.wave_resnet_mamba import WaveResNetMamba

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("Environment OK")
PY
```
