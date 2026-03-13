"""
cross_validate.py — 5-fold stratified cross-validation.

Reproduces Tables 1 and 2 from the paper.
Saves per-fold results and per-sample predictions to JSON.

Usage:
    # Train and evaluate one model:
    python scripts/cross_validate.py \
        --model wave_resnet_mamba \
        --data  data/physionet2016 \
        --output results/cv_wavemamba.json

    # McNemar's test between two saved result files:
    python scripts/cross_validate.py \
        --mcnemar \
        --baseline results/cv_resnet.json \
        --proposed results/cv_wavemamba.json
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset  import PhysioNet2016Dataset, get_dataloaders
from utils.metrics  import compute_metrics, mcnemar_test, print_metrics
from models         import build_model


# ── Config defaults (override via CLI) ────────────────────────
DEFAULT = dict(
    epochs       = 60,
    batch_size   = 16,
    lr           = 1e-4,
    weight_decay = 1e-4,
    grad_clip    = 2.0,
    warmup_epochs= 5,
    n_folds      = 5,
    seed         = 42,
    num_workers  = 4,
    class_weights= [1.0, 3.5],
    label_smooth = 0.1,
)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_one_epoch(model, loader, criterion, optimizer,
                    scheduler, grad_clip, device, warmup_steps, step):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # Warmup LR
        if step < warmup_steps:
            lr_scale = (step + 1) / warmup_steps
            for pg in optimizer.param_groups:
                pg["lr"] = DEFAULT["lr"] * lr_scale
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        step += 1
    return total_loss / len(loader), step


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.numpy().tolist())
    return np.array(all_labels), np.array(all_preds)


def run_cv(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Model  : {args.model}")
    print(f"  Device : {device}")
    print(f"  Folds  : {args.n_folds}")
    print(f"{'='*55}\n")

    # ── Load full dataset (no augmentation) for fold splitting
    full_ds = PhysioNet2016Dataset(args.data, indices=None, augment=False)
    all_labels = np.array([full_ds.samples[i][1] for i in range(len(full_ds))])
    all_indices = np.arange(len(full_ds))

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                          random_state=args.seed)

    fold_results = []
    all_true, all_pred = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_indices, all_labels)):
        print(f"\n── Fold {fold+1}/{args.n_folds} ──────────────────────────")

        train_loader, val_loader = get_dataloaders(
            args.data, train_idx.tolist(), val_idx.tolist(),
            batch_size=args.batch_size, num_workers=args.num_workers
        )

        # ── Build model ───────────────────────────────────────
        model = build_model(args.model).to(device)

        # ── Loss (weighted cross-entropy + label smoothing) ───
        cw        = torch.tensor(args.class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw,
                                        label_smoothing=args.label_smooth)

        # ── Optimiser & Scheduler ─────────────────────────────
        optimizer = AdamW(model.parameters(),
                          lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=args.epochs - args.warmup_epochs,
                                      eta_min=1e-6)
        warmup_steps = args.warmup_epochs * len(train_loader)

        best_macc, best_state, step = 0.0, None, 0

        for epoch in range(args.epochs):
            loss, step = train_one_epoch(
                model, train_loader, criterion, optimizer,
                scheduler, args.grad_clip, device, warmup_steps, step
            )
            if epoch >= args.warmup_epochs:
                scheduler.step()

            y_true, y_pred = evaluate(model, val_loader, device)
            metrics = compute_metrics(y_true, y_pred)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d} | Loss {loss:.4f} | "
                      f"MAcc {metrics['macc']:.2f}% | "
                      f"Se {metrics['sensitivity']:.2f}% | "
                      f"Sp {metrics['specificity']:.2f}%")

            if metrics["macc"] > best_macc:
                best_macc  = metrics["macc"]
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}

        # ── Final evaluation on best checkpoint ───────────────
        model.load_state_dict(best_state)
        y_true, y_pred = evaluate(model, val_loader, device)
        metrics = compute_metrics(y_true, y_pred)
        print_metrics(metrics, prefix=f"Fold {fold+1} best checkpoint")

        fold_results.append(metrics)
        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())

        # Save fold checkpoint
        os.makedirs("results", exist_ok=True)
        torch.save(best_state, f"results/{args.model}_fold{fold+1}.pth")

    # ── Summary across folds ──────────────────────────────────
    maccs = [r["macc"]        for r in fold_results]
    ses   = [r["sensitivity"] for r in fold_results]
    sps   = [r["specificity"] for r in fold_results]

    print(f"\n{'='*55}")
    print(f"  5-FOLD CV SUMMARY  —  {args.model}")
    print(f"{'='*55}")
    print(f"  MAcc : {np.mean(maccs):.2f}% ± {np.std(maccs):.2f}%")
    print(f"  Se   : {np.mean(ses):.2f}% ± {np.std(ses):.2f}%")
    print(f"  Sp   : {np.mean(sps):.2f}% ± {np.std(sps):.2f}%")
    print(f"{'='*55}\n")

    # ── Save results to JSON ──────────────────────────────────
    output = {
        "model":        args.model,
        "n_folds":      args.n_folds,
        "seed":         args.seed,
        "fold_results": fold_results,
        "summary": {
            "macc_mean": float(np.mean(maccs)),
            "macc_std":  float(np.std(maccs)),
            "se_mean":   float(np.mean(ses)),
            "se_std":    float(np.std(ses)),
            "sp_mean":   float(np.mean(sps)),
            "sp_std":    float(np.std(sps)),
        },
        # Per-sample predictions for McNemar's test
        "all_true": all_true,
        "all_pred": all_pred,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to: {args.output}")


def run_mcnemar(args):
    with open(args.baseline) as f:
        baseline = json.load(f)
    with open(args.proposed) as f:
        proposed = json.load(f)

    y_true_a = np.array(baseline["all_true"])
    y_pred_a = np.array(baseline["all_pred"])
    y_pred_b = np.array(proposed["all_pred"])

    errors_a = y_true_a != y_pred_a
    errors_b = y_true_a != y_pred_b

    result = mcnemar_test(errors_a, errors_b)

    print(f"\n{'='*55}")
    print(f"  McNemar's Test")
    print(f"  Baseline : {baseline['model']}")
    print(f"  Proposed : {proposed['model']}")
    print(f"{'='*55}")
    print(f"  Discordant b (A wrong, B correct) : {result['b']}")
    print(f"  Discordant c (A correct, B wrong) : {result['c']}")
    print(f"  Chi-squared statistic : {result['chi2_stat']:.4f}")
    print(f"  p-value               : {result['p_value']:.6f}")
    print(f"  Significant (p<0.05)  : {result['significant_005']}")
    print(f"  Significant (p<0.001) : {result['significant_001']}")
    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser(
        description="5-Fold CV for WaveResNetMamba paper"
    )
    # Mode
    parser.add_argument("--mcnemar", action="store_true",
                        help="Run McNemar's test instead of CV training")
    parser.add_argument("--baseline", type=str,
                        help="Path to baseline CV results JSON")
    parser.add_argument("--proposed", type=str,
                        help="Path to proposed model CV results JSON")

    # Training args
    parser.add_argument("--model",   type=str, default="wave_resnet_mamba",
                        choices=["resnet1d", "wavelet_resnet", "wave_resnet_mamba"])
    parser.add_argument("--data",    type=str, default="data/physionet2016")
    parser.add_argument("--output",  type=str, default="results/cv_results.json")
    parser.add_argument("--epochs",  type=int, default=DEFAULT["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT["batch_size"])
    parser.add_argument("--lr",      type=float, default=DEFAULT["lr"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULT["weight_decay"])
    parser.add_argument("--grad_clip", type=float, default=DEFAULT["grad_clip"])
    parser.add_argument("--warmup_epochs", type=int, default=DEFAULT["warmup_epochs"])
    parser.add_argument("--n_folds", type=int, default=DEFAULT["n_folds"])
    parser.add_argument("--seed",    type=int, default=DEFAULT["seed"])
    parser.add_argument("--num_workers", type=int, default=DEFAULT["num_workers"])
    parser.add_argument("--class_weights", type=float, nargs=2,
                        default=DEFAULT["class_weights"])
    parser.add_argument("--label_smooth", type=float, default=DEFAULT["label_smooth"])

    args = parser.parse_args()

    if args.mcnemar:
        run_mcnemar(args)
    else:
        run_cv(args)


if __name__ == "__main__":
    main()
