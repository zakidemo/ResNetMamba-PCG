"""
dataset.py — PhysioNet/CinC Challenge 2016 Dataset Loader.

Dataset: https://physionet.org/content/challenge-2016/1.0.0/
3,240 recordings across 6 sub-databases (training-a through training-f).
Labels: -1 (normal) -> 0, +1 (abnormal) -> 1

Usage:
    train_ds = PhysioNet2016Dataset(root="data/physionet2016",
                                    indices=train_idx, augment=True)
    val_ds   = PhysioNet2016Dataset(root="data/physionet2016",
                                    indices=val_idx,   augment=False)
"""

import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io
import scipy.signal


# Sub-databases in the PhysioNet 2016 challenge
SUBDBS = ["training-a", "training-b", "training-c",
          "training-d", "training-e", "training-f"]

TARGET_SR     = 2000   # Hz — resample everything to this
SIGNAL_LEN    = 16000  # samples = 8 seconds at 2 kHz


class PhysioNet2016Dataset(Dataset):
    """
    Loads PhysioNet/CinC 2016 PCG recordings.

    Args:
        root     : Path to the physionet2016 directory containing
                   training-a/ through training-f/ subdirectories.
        indices  : List of integer indices into self.samples.
                   If None, uses all samples.
        augment  : Apply training augmentation if True.
        transform: Optional callable applied to the waveform tensor.
    """

    def __init__(self, root: str, indices=None, augment: bool = False,
                 transform=None):
        self.root      = root
        self.augment   = augment
        self.transform = transform
        self.samples   = self._load_manifest()

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    # ── Manifest ──────────────────────────────────────────────
    def _load_manifest(self):
        """
        Walks all sub-databases and collects (wav_path, label) pairs.
        Labels are read from the REFERENCE.csv file in each sub-db.
        """
        samples = []
        for subdb in SUBDBS:
            subdb_path = os.path.join(self.root, subdb)
            if not os.path.isdir(subdb_path):
                continue

            ref_path = os.path.join(subdb_path, "REFERENCE.csv")
            if not os.path.exists(ref_path):
                # Some sub-databases use REFERENCE.csv at root level
                ref_path = os.path.join(self.root, f"REFERENCE-{subdb[-1].upper()}.csv")

            label_map = {}
            if os.path.exists(ref_path):
                with open(ref_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split(",")
                        if len(parts) >= 2:
                            fname  = parts[0].strip()
                            raw_lbl = int(parts[1].strip())
                            # Convert: -1 (normal)->0, +1 (abnormal)->1
                            label  = 0 if raw_lbl == -1 else 1
                            label_map[fname] = label

            # Find all .wav files
            wav_files = sorted(glob.glob(os.path.join(subdb_path, "*.wav")))
            for wav_path in wav_files:
                fname = os.path.splitext(os.path.basename(wav_path))[0]
                if fname in label_map:
                    samples.append((wav_path, label_map[fname]))

        if len(samples) == 0:
            raise RuntimeError(
                f"No samples found in {self.root}. "
                "Ensure the directory structure matches: "
                "data/physionet2016/training-a/ ... training-f/"
            )
        return samples

    # ── Item loading ──────────────────────────────────────────
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, label = self.samples[idx]

        # Load and resample
        waveform, sr = self._load_wav(wav_path)
        if sr != TARGET_SR:
            waveform = scipy.signal.resample_poly(
                waveform,
                up=TARGET_SR,
                down=sr
            ).astype(np.float32)

        # Pad / truncate to fixed length
        waveform = self._fix_length(waveform, SIGNAL_LEN)

        # Instance normalisation (zero mean, unit variance)
        waveform = self._normalise(waveform)

        # Augmentation (training only)
        if self.augment:
            waveform = self._augment(waveform)

        # (1, T) tensor
        x = torch.from_numpy(waveform).float().unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    # ── Internal helpers ──────────────────────────────────────
    @staticmethod
    def _load_wav(path: str):
        """Load .wav file, return (numpy_array, sample_rate)."""
        import wave, array as arr
        try:
            with wave.open(path, 'rb') as wf:
                sr     = wf.getframerate()
                nch    = wf.getnchannels()
                sw     = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())
            dtype  = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
            signal = np.frombuffer(frames, dtype=dtype).astype(np.float32)
            if nch > 1:
                signal = signal[::nch]          # take first channel
            signal /= np.iinfo(dtype).max       # normalise to [-1, 1]
            return signal, sr
        except Exception:
            # Fallback: scipy
            sr, signal = scipy.io.wavfile.read(path)
            signal = signal.astype(np.float32)
            if signal.ndim > 1:
                signal = signal[:, 0]
            if np.iinfo(np.int16).min <= signal.min():
                signal /= 32768.0
            return signal, sr

    @staticmethod
    def _fix_length(x: np.ndarray, length: int) -> np.ndarray:
        if len(x) >= length:
            return x[:length]
        pad = length - len(x)
        return np.pad(x, (0, pad), mode="constant")

    @staticmethod
    def _normalise(x: np.ndarray) -> np.ndarray:
        mean = x.mean()
        std  = x.std() + 1e-8
        return (x - mean) / std

    @staticmethod
    def _augment(x: np.ndarray) -> np.ndarray:
        """Apply random time shift, amplitude scaling, Gaussian noise."""
        # Random time shift (circular — wrapped with zero fill)
        if random.random() < 0.5:
            max_shift = int(0.10 * len(x))
            shift     = random.randint(-max_shift, max_shift)
            x         = np.roll(x, shift)
            # Zero out wrapped region (avoids circular boundary artefact)
            if shift > 0:
                x[:shift] = 0.0
            elif shift < 0:
                x[shift:] = 0.0

        # Random amplitude scaling
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            x     = x * scale

        # Additive Gaussian noise (SNR ~ 46 dB)
        if random.random() < 0.5:
            signal_power = np.mean(x ** 2) + 1e-10
            noise_power  = signal_power / (10 ** (46.0 / 10.0))
            noise        = np.random.normal(0, np.sqrt(noise_power), size=x.shape)
            x            = x + noise.astype(np.float32)

        return x


# ── DataLoader factory ────────────────────────────────────────
def get_dataloaders(root: str, train_idx, val_idx,
                    batch_size: int = 16, num_workers: int = 4):
    """
    Build train and validation DataLoaders for one CV fold.
    NOTE: augmentation applied ONLY to training set (separate instance).
    """
    train_ds = PhysioNet2016Dataset(root, indices=train_idx, augment=True)
    val_ds   = PhysioNet2016Dataset(root, indices=val_idx,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
