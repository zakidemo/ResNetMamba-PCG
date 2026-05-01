"""
train_wave_resnet_mamba.py
===========================
Training script for WaveResNetMamba — Contribution 2.

Identical experimental conditions to both baselines:
  - Seed 42, 80/20 split
  - Same class weights [1.0, 3.5]
  - Same augmentation strategy (train only)
  - Save on MAcc (PhysioNet metric)

Added vs Contribution 1:
  - Warmup LR schedule (5 epochs) + CosineAnnealing
    Mamba SSMs benefit from a gentle warmup before the main decay.
  - Slightly higher gradient clip (2.0) — deeper model, larger gradients
  - 60 epochs — the Mamba stage needs more steps to converge than pure ResNet
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.metrics import confusion_matrix, classification_report

from models.wave_resnet_mamba import WaveResNetMamba
from utils.dataset import PhysioNet2016Dataset


# ── LR warmup + cosine decay ──────────────────────────────────────────────────

def get_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs          # linear warmup
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    # ── Config ────────────────────────────────────────────────────────────────
    BATCH_SIZE    = 16
    LEARNING_RATE = 1e-4
    EPOCHS        = 60
    WARMUP_EPOCHS = 5
    SEED          = 42
    D_MODEL       = 128
    BASE_CH       = 32
    N_MAMBA       = 2
    NUM_CLASSES   = 2
    SAVE_PATH     = "models/best_wave_resnet_mamba.pth"

    BASE_DIR = (
        "data/classification-of-heart-sound-recordings-the-physionet-"
        "computing-in-cardiology-challenge-2016-1.0.0"
    )
    FOLDERS = [
        "training-a", "training-b", "training-c",
        "training-d", "training-e", "training-f",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training WaveResNetMamba (Contribution 2) on: {device.type.upper()}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\nLoading datasets...")

    base_dataset = PhysioNet2016Dataset(
        base_dir=BASE_DIR, folders=FOLDERS, augment=False
    )
    total      = len(base_dataset)
    train_size = int(0.8 * total)
    val_size   = total - train_size

    generator  = torch.Generator().manual_seed(SEED)
    indices    = list(range(total))
    split      = random_split(indices, [train_size, val_size], generator=generator)
    train_idx, val_idx = split[0], split[1]

    aug_dataset   = PhysioNet2016Dataset(
        base_dir=BASE_DIR, folders=FOLDERS, augment=True
    )
    train_dataset = Subset(aug_dataset,  train_idx)
    val_dataset   = Subset(base_dataset, val_idx)

    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = WaveResNetMamba(
        num_classes=NUM_CLASSES,
        base_ch=BASE_CH,
        d_model=D_MODEL,
        n_mamba=N_MAMBA,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # ── Loss / Optimizer / Scheduler ──────────────────────────────────────────
    class_weights = torch.tensor([1.0, 3.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = get_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_macc = 0.0

    print("\n" + "=" * 60)
    print("  WAVE-RESNET-MAMBA — CONTRIBUTION 2 — TRAINING")
    print("=" * 60)

    for epoch in range(EPOCHS):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0

        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(signals)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # --- Validate ---
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0

        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                logits = model(signals)
                val_loss += criterion(logits, labels).item()
                _, preds = torch.max(logits, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        macc        = (sensitivity + specificity) / 2

        elapsed        = time.time() - t0
        current_lr     = scheduler.get_last_lr()[0]
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg   = val_loss   / len(val_loader)

        print(
            f"Epoch [{epoch+1:02d}/{EPOCHS}] | {elapsed:.1f}s | "
            f"LR: {current_lr:.2e} | "
            f"TLoss: {train_loss_avg:.4f} | VLoss: {val_loss_avg:.4f}"
        )
        print(
            f"   → Sens: {sensitivity*100:.2f}% | "
            f"Spec: {specificity*100:.2f}% | "
            f"MAcc: {macc*100:.2f}%"
        )

        if macc > best_macc:
            best_macc = macc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"   🌟 New Best MAcc: {best_macc*100:.2f}% — Saved!")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION ON BEST CHECKPOINT")
    print("=" * 60)

    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for signals, labels in val_loader:
            signals, labels = signals.to(device), labels.to(device)
            _, preds = torch.max(model(signals), 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    macc        = (sensitivity + specificity) / 2

    print(f"\nConfusion Matrix:\n{cm}\n")
    print(f"Sensitivity (Abnormal) : {sensitivity*100:.2f}%")
    print(f"Specificity (Normal)   : {specificity*100:.2f}%")
    print(f"MAcc                   : {macc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=["Normal (0)", "Abnormal (1)"]
    ))

    print("\n── Full Comparison Table ─────────────────────────────")
    print(f"  ResNet1D baseline       : 88.37%  (Sens: 80.00% | Spec: 89.86%)")
    print(f"  WaveMamba (fixed)       : 84.77%")
    print(f"  WaveletResNet           : 87.71%  (Sens: 91.72% | Spec: 83.70%)")
    print(f"  WaveResNetMamba (ours)  : {best_macc*100:.2f}%  ← Contribution 2")
    print("──────────────────────────────────────────────────────")
    print(f"\n🎉 DONE — Best MAcc: {best_macc*100:.2f}%")


if __name__ == "__main__":
    main()
