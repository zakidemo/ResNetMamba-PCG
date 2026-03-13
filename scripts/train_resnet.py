"""
train_resnet.py — Single-run training for ResNet-1D baseline.

For full 5-fold CV (used in the paper), use cross_validate.py instead.
"""

import os, sys, argparse, torch, numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataset import PhysioNet2016Dataset, get_dataloaders
from utils.metrics import compute_metrics, print_metrics
from models import build_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="data/physionet2016")
    parser.add_argument("--epochs",  type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr",      type=float, default=1e-4)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--output",  default="results/resnet_run/")
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_ds = PhysioNet2016Dataset(args.data, augment=False)
    n = len(full_ds)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split].tolist(), idx[split:].tolist()

    train_loader, val_loader = get_dataloaders(
        args.data, train_idx, val_idx, args.batch_size)

    model     = build_model("resnet1d").to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 3.5]).to(device), label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - 5, eta_min=1e-6)

    best_macc, best_state = 0.0, None
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        if epoch >= 5: scheduler.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                preds.extend(model(x.to(device)).argmax(1).cpu().tolist())
                labels.extend(y.tolist())
        m = compute_metrics(np.array(labels), np.array(preds))
        if m["macc"] > best_macc:
            best_macc  = m["macc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: MAcc={m['macc']:.2f}% Se={m['sensitivity']:.2f}% Sp={m['specificity']:.2f}%")

    model.load_state_dict(best_state)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            preds.extend(model(x.to(device)).argmax(1).cpu().tolist())
            labels.extend(y.tolist())
    print_metrics(compute_metrics(np.array(labels), np.array(preds)), "Final ResNet-1D")
    os.makedirs(args.output, exist_ok=True)
    torch.save(best_state, os.path.join(args.output, "best_resnet.pth"))

if __name__ == "__main__":
    main()
