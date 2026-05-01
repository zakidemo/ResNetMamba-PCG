import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.metrics import confusion_matrix, classification_report

from models.wavemamba_1d import WaveMamba1D_Classifier
from utils.dataset import PhysioNet2016Dataset


def main():
    # ── Hyperparameters ───────────────────────────────────────────────────────
    BATCH_SIZE      = 16
    LEARNING_RATE   = 1e-4
    EPOCHS          = 50
    SEED            = 42          # Same seed as ResNet for a fair comparison
    D_MODEL         = 64
    NUM_CLASSES     = 2
    SAVE_PATH       = "models/best_wavemamba.pth"

    BASE_DIR = (
        "data/classification-of-heart-sound-recordings-the-physionet-"
        "computing-in-cardiology-challenge-2016-1.0.0"
    )
    FOLDERS = [
        "training-a", "training-b", "training-c",
        "training-d", "training-e", "training-f",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training WaveMamba on: {device.type.upper()}")

    # ── Dataset: two separate instances to prevent augmentation leaking ───────
    # BUG FIX #3: Create separate train/val datasets so val is NEVER augmented.
    # We use index-based splitting to guarantee identical splits across runs.
    print("\nLoading datasets...")

    base_dataset = PhysioNet2016Dataset(
        base_dir=BASE_DIR, folders=FOLDERS, augment=False
    )
    total = len(base_dataset)
    train_size = int(0.8 * total)
    val_size   = total - train_size

    # BUG FIX #1: Fixed seed so split matches the ResNet baseline exactly.
    generator = torch.Generator().manual_seed(SEED)
    train_indices, val_indices = random_split(
        range(total), [train_size, val_size], generator=generator
    )

    # Training subset gets augmentation; validation never does.
    train_dataset_aug = PhysioNet2016Dataset(
        base_dir=BASE_DIR, folders=FOLDERS, augment=True
    )
    train_dataset = Subset(train_dataset_aug, train_indices.indices)
    val_dataset   = Subset(base_dataset,      val_indices.indices)

    print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = WaveMamba1D_Classifier(
        sequence_length=16000, d_model=D_MODEL, num_classes=NUM_CLASSES
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {total_params:,}")

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────────
    # Class weights: abnormal class is underrepresented (~23% of data)
    class_weights = torch.tensor([1.0, 3.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )

    # Cosine annealing: decays LR smoothly, prevents plateau
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_macc = 0.0

    print("\n" + "=" * 55)
    print("  BEGINNING WAVEMAMBA TRAINING (Fixed)")
    print("=" * 55)

    for epoch in range(EPOCHS):
        start_time = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0

        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            # Gradient clipping: Mamba SSMs can have exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                outputs = model(signals)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # BUG FIX #2: Save on MAcc (PhysioNet metric), NOT raw accuracy.
        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        macc        = (sensitivity + specificity) / 2

        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg   = val_loss   / len(val_loader)
        current_lr     = scheduler.get_last_lr()[0]
        elapsed        = time.time() - start_time

        print(
            f"Epoch [{epoch+1:02d}/{EPOCHS}] | {elapsed:.1f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f}"
        )
        print(
            f"   → Sens: {sensitivity*100:.2f}% | "
            f"Spec: {specificity*100:.2f}% | "
            f"MAcc: {macc*100:.2f}%"
        )

        if macc > best_macc:
            best_macc = macc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"   🌟 New Best MAcc: {best_macc*100:.2f}% — Model Saved!")

    # ── Final Evaluation ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  LOADING BEST MODEL FOR FINAL EVALUATION")
    print("=" * 55)

    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for signals, labels in val_loader:
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
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
    print(f"\n🎉 TRAINING COMPLETE — Best MAcc: {best_macc*100:.2f}%")


if __name__ == "__main__":
    main()