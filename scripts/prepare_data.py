"""
scripts/prepare_data.py
========================
Preprocess PhysioNet/CinC 2016 raw .wav files into clean .npy arrays.

Steps:
  1. Read all .wav files from training-a … training-f
  2. Resample to 2 kHz
  3. Truncate or zero-pad to 8 seconds (16,000 samples)
  4. Instance-normalise (zero mean, unit variance)
  5. Parse labels from REFERENCE.csv files
  6. Save signals.npy and labels.npy to output directory

Usage:
    python scripts/prepare_data.py --data_dir data/raw --out_dir data/processed
"""

import os
import csv
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    import torchaudio
    BACKEND = "torchaudio"
except ImportError:
    try:
        import soundfile as sf
        BACKEND = "soundfile"
    except ImportError:
        raise ImportError("Install torchaudio or soundfile: pip install torchaudio")


TARGET_SR = 2000       # Hz
DURATION  = 8          # seconds
N_SAMPLES = TARGET_SR * DURATION  # 16,000


def load_wav(path: str) -> np.ndarray:
    """Load a .wav file and return a mono float32 array at native SR."""
    if BACKEND == "torchaudio":
        import torchaudio
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)   # stereo → mono
        wav = wav.squeeze(0).numpy()
    else:
        import soundfile as sf
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
    return wav, sr


def resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Resample using scipy if needed."""
    if sr == target_sr:
        return wav
    from scipy.signal import resample_poly
    from math import gcd
    g   = gcd(sr, target_sr)
    up  = target_sr // g
    down= sr // g
    return resample_poly(wav, up, down).astype(np.float32)


def pad_or_trim(wav: np.ndarray, n: int) -> np.ndarray:
    """Truncate to n samples or zero-pad on the right."""
    if len(wav) >= n:
        return wav[:n]
    return np.pad(wav, (0, n - len(wav)))


def normalise(wav: np.ndarray) -> np.ndarray:
    """Instance normalisation: zero mean, unit variance."""
    std = wav.std()
    if std < 1e-8:
        return wav - wav.mean()
    return (wav - wav.mean()) / std


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/raw",
                   help="Directory containing training-a … training-f folders")
    p.add_argument("--out_dir",  default="data/processed")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    signals_list, labels_list = [], []
    splits = sorted([d for d in data_dir.iterdir()
                     if d.is_dir() and d.name.startswith("training")])

    print(f"Found {len(splits)} sub-databases: {[d.name for d in splits]}")

    for split_dir in splits:
        ref_file = split_dir / "REFERENCE.csv"
        if not ref_file.exists():
            print(f"  [WARN] No REFERENCE.csv in {split_dir}, skipping.")
            continue

        # Parse label file: filename, label (-1=normal, 1=abnormal)
        rows = {}
        with open(ref_file) as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    fname  = row[0].strip()
                    label  = int(row[1].strip())
                    rows[fname] = 0 if label == -1 else 1

        wav_files = sorted(split_dir.glob("*.wav"))
        print(f"  {split_dir.name}: {len(wav_files)} files, "
              f"{sum(v==1 for v in rows.values())} abnormal")

        for wav_path in tqdm(wav_files, desc=f"  {split_dir.name}", leave=False):
            stem = wav_path.stem
            if stem not in rows:
                continue   # no label → skip

            try:
                wav, sr = load_wav(str(wav_path))
                wav = resample(wav, sr, TARGET_SR)
                wav = pad_or_trim(wav, N_SAMPLES)
                wav = normalise(wav)
                signals_list.append(wav)
                labels_list.append(rows[stem])
            except Exception as e:
                print(f"    [ERROR] {wav_path.name}: {e}")

    signals = np.stack(signals_list, axis=0).astype(np.float32)  # (N, T)
    labels  = np.array(labels_list, dtype=np.int64)               # (N,)

    print(f"\nDataset summary:")
    print(f"  Total recordings : {len(labels)}")
    print(f"  Normal (0)       : {(labels == 0).sum()}")
    print(f"  Abnormal (1)     : {(labels == 1).sum()}")
    print(f"  Signal shape     : {signals.shape}")

    np.save(out_dir / "signals.npy", signals)
    np.save(out_dir / "labels.npy",  labels)
    print(f"\n  Saved → {out_dir}/signals.npy  ({signals.nbytes / 1e6:.1f} MB)")
    print(f"  Saved → {out_dir}/labels.npy")


if __name__ == "__main__":
    main()
