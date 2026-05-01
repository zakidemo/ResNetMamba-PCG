"""
PhysioNet/CinC Challenge 2016 Dataset Loader.

Compatible call styles:
    PhysioNet2016Dataset(root="data/...", augment=True)
    PhysioNet2016Dataset(base_dir="data/...", folders=[...], augment=True)

Labels: -1 normal -> 0, +1 abnormal -> 1.
"""

import glob
import os
import random
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io.wavfile
import scipy.signal
import torch
from torch.utils.data import DataLoader, Dataset

SUBDBS = ["training-a", "training-b", "training-c", "training-d", "training-e", "training-f"]
TARGET_SR = 2000
SIGNAL_LEN = 16000


class PhysioNet2016Dataset(Dataset):
    def __init__(
        self,
        root: Optional[str] = None,
        indices: Optional[Sequence[int]] = None,
        augment: bool = False,
        transform=None,
        base_dir: Optional[str] = None,
        folders: Optional[Iterable[str]] = None,
    ):
        self.root = base_dir if base_dir is not None else root
        if self.root is None:
            raise ValueError("PhysioNet2016Dataset requires root=... or base_dir=...")
        self.folders = list(folders) if folders is not None else list(SUBDBS)
        self.augment = augment
        self.transform = transform
        self.samples: List[Tuple[str, int]] = self._load_manifest()
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]
        self.file_paths = [p for p, _ in self.samples]
        self.labels = [int(y) for _, y in self.samples]

    def _load_manifest(self) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        for subdb in self.folders:
            subdb_path = os.path.join(self.root, subdb)
            if not os.path.isdir(subdb_path):
                continue
            ref_candidates = [
                os.path.join(subdb_path, "REFERENCE.csv"),
                os.path.join(self.root, f"REFERENCE-{subdb[-1].upper()}.csv"),
            ]
            ref_path = next((p for p in ref_candidates if os.path.exists(p)), None)
            label_map = {}
            if ref_path is not None:
                with open(ref_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split(",")
                        if len(parts) >= 2:
                            fname = parts[0].strip()
                            raw_label = int(parts[1].strip())
                            label_map[fname] = 0 if raw_label == -1 else 1
            for wav_path in sorted(glob.glob(os.path.join(subdb_path, "*.wav"))):
                fname = os.path.splitext(os.path.basename(wav_path))[0]
                if fname in label_map:
                    samples.append((wav_path, label_map[fname]))
        if not samples:
            raise RuntimeError(
                f"No labeled .wav samples found under {self.root}. Expected folders training-a ... training-f with labels."
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, label = self.samples[idx]
        waveform, sr = self._load_wav(wav_path)
        waveform = np.asarray(waveform, dtype=np.float32)
        if sr != TARGET_SR:
            waveform = scipy.signal.resample_poly(waveform, up=TARGET_SR, down=sr).astype(np.float32)
        waveform = self._fix_length(waveform, SIGNAL_LEN)
        waveform = self._normalise(waveform)
        if self.augment:
            waveform = self._augment(waveform)
        x = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0)
        y = torch.tensor(int(label), dtype=torch.long)
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    @staticmethod
    def _load_wav(path: str):
        sr, signal = scipy.io.wavfile.read(path)
        signal = np.asarray(signal)
        if signal.ndim > 1:
            signal = signal[:, 0]
        if np.issubdtype(signal.dtype, np.integer):
            max_abs = max(abs(np.iinfo(signal.dtype).min), np.iinfo(signal.dtype).max)
            signal = signal.astype(np.float32) / float(max_abs)
        else:
            signal = signal.astype(np.float32)
        return signal, sr

    @staticmethod
    def _fix_length(x: np.ndarray, length: int) -> np.ndarray:
        if len(x) >= length:
            return x[:length]
        return np.pad(x, (0, length - len(x)), mode="constant")

    @staticmethod
    def _normalise(x: np.ndarray) -> np.ndarray:
        return (x - float(x.mean())) / (float(x.std()) + 1e-8)

    @staticmethod
    def _augment(x: np.ndarray) -> np.ndarray:
        x = x.copy()
        if random.random() < 0.5:
            max_shift = int(0.10 * len(x))
            shift = random.randint(-max_shift, max_shift)
            x = np.roll(x, shift)
            if shift > 0:
                x[:shift] = 0.0
            elif shift < 0:
                x[shift:] = 0.0
        if random.random() < 0.5:
            x *= random.uniform(0.8, 1.2)
        if random.random() < 0.5:
            signal_power = np.mean(x ** 2) + 1e-10
            noise_power = signal_power / (10 ** (46.0 / 10.0))
            x += np.random.normal(0, np.sqrt(noise_power), size=x.shape).astype(np.float32)
        return x.astype(np.float32)


def get_dataloaders(root: str, train_idx, val_idx, batch_size: int = 16, num_workers: int = 4):
    train_ds = PhysioNet2016Dataset(root=root, indices=train_idx, augment=True)
    val_ds = PhysioNet2016Dataset(root=root, indices=val_idx, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
