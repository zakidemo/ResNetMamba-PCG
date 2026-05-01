"""
cross_validate_ieee.py
======================
IEEE-quality 5-fold stratified cross-validation for ResNetMamba-PCG.

What this script adds compared with the original cross_validate.py:
  - trains the proposed WaveResNetMamba and real ResNet1D baseline on the SAME folds
  - saves matched out-of-fold predictions for valid McNemar testing
  - reports per-class precision/recall/F1 and macro-F1
  - saves confusion matrices per fold and overall OOF results
  - saves parameter counts and GPU inference-time measurements
  - writes reviewer-ready CSV/JSON files under results/ieee_cv/

Usage:
  python cross_validate_ieee.py

Optional fast smoke test:
  python cross_validate_ieee.py --epochs 1 --n-folds 2 --tag smoke

Important:
  Use this only in the clean CUDA/Mamba environment, with official mamba_ssm installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.contingency_tables import mcnemar
from torch.utils.data import DataLoader, Subset

from models.resnet1d import ResNet1D
from models.wave_resnet_mamba import WaveResNetMamba
from utils.dataset import PhysioNet2016Dataset


@dataclass
class Config:
    n_folds: int = 5
    epochs: int = 60
    warmup_epochs: int = 5
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_classes: int = 2
    seed: int = 42
    num_workers: int = 4
    class_weight_normal: float = 1.0
    class_weight_abnormal: float = 3.5
    label_smoothing: float = 0.1
    base_ch: int = 32
    d_model: int = 128
    n_mamba: int = 2
    max_grad_norm: float = 2.0
    base_dir: str = (
        "data/classification-of-heart-sound-recordings-the-physionet-"
        "computing-in-cardiology-challenge-2016-1.0.0"
    )
    folders: Tuple[str, ...] = (
        "training-a",
        "training-b",
        "training-c",
        "training-d",
        "training-e",
        "training-f",
    )
    tag: str = "final"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_scheduler(optimizer: optim.Optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = float(epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_model(model_name: str, cfg: Config) -> nn.Module:
    if model_name == "proposed":
        return WaveResNetMamba(
            num_classes=cfg.num_classes,
            base_ch=cfg.base_ch,
            d_model=cfg.d_model,
            n_mamba=cfg.n_mamba,
        )
    if model_name == "resnet1d":
        return ResNet1D(n_classes=cfg.num_classes)
    raise ValueError(f"Unknown model_name: {model_name}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


def measure_inference_time(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 16,
    signal_length: int = 16000,
    warmup: int = 25,
    repeats: int = 100,
) -> Dict[str, float]:
    model.eval()
    x = torch.randn(batch_size, 1, signal_length, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    per_batch_ms = 1000.0 * elapsed / repeats
    per_sample_ms = per_batch_ms / batch_size
    throughput = 1000.0 / per_sample_ms if per_sample_ms > 0 else 0.0
    return {
        "batch_size": batch_size,
        "repeats": repeats,
        "per_batch_ms": per_batch_ms,
        "per_sample_ms": per_sample_ms,
        "samples_per_second": throughput,
    }


def compute_binary_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float | List[List[int]]]:
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0  # abnormal recall
    specificity = tn / (tn + fp) if (tn + fp) else 0.0  # normal recall
    macc = 0.5 * (sensitivity + specificity)

    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], zero_division=0
    )

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": cm.astype(int).tolist(),
        "accuracy": float(accuracy_score(labels, preds)),
        "sensitivity_abnormal_recall": float(sensitivity),
        "specificity_normal_recall": float(specificity),
        "macc": float(macc),
        "precision_normal": float(precision_per_class[0]),
        "recall_normal": float(recall_per_class[0]),
        "f1_normal": float(f1_per_class[0]),
        "support_normal": int(support[0]),
        "precision_abnormal": float(precision_per_class[1]),
        "recall_abnormal": float(recall_per_class[1]),
        "f1_abnormal": float(f1_per_class[1]),
        "support_abnormal": int(support[1]),
        "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


def train_one_model_one_fold(
    model_name: str,
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    aug_dataset: PhysioNet2016Dataset,
    base_dataset: PhysioNet2016Dataset,
    device: torch.device,
    cfg: Config,
    out_dir: Path,
) -> Dict:
    print(f"\n{'=' * 72}")
    print(f"MODEL: {model_name} | FOLD {fold + 1}/{cfg.n_folds}")
    print(f"{'=' * 72}")
    print(f"Train: {len(train_idx):,} | Val: {len(val_idx):,}")

    train_loader = DataLoader(
        Subset(aug_dataset, train_idx.tolist()),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        Subset(base_dataset, val_idx.tolist()),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = make_model(model_name, cfg).to(device)
    class_weights = torch.tensor(
        [cfg.class_weight_normal, cfg.class_weight_abnormal], dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_scheduler(optimizer, cfg.warmup_epochs, cfg.epochs)

    best_macc = -1.0
    best_epoch = -1
    best_preds: List[int] = []
    best_probs: List[float] = []
    best_labels: List[int] = []
    checkpoint_path = out_dir / f"{model_name}_fold_{fold + 1}.pth"

    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0

        for signals, labels in train_loader:
            signals = signals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(signals)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
            optimizer.step()
            running_loss += float(loss.item())

        scheduler.step()

        model.eval()
        val_labels: List[int] = []
        val_preds: List[int] = []
        val_probs: List[float] = []
        val_loss = 0.0

        with torch.no_grad():
            for signals, labels in val_loader:
                signals = signals.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(signals)
                loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)
                val_loss += float(loss.item())
                val_labels.extend(labels.cpu().numpy().astype(int).tolist())
                val_preds.extend(preds.cpu().numpy().astype(int).tolist())
                val_probs.extend(probs.cpu().numpy().astype(float).tolist())

        metrics = compute_binary_metrics(np.array(val_labels), np.array(val_preds))
        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch + 1:02d}/{cfg.epochs}] | {elapsed:6.1f}s | "
            f"loss {running_loss / max(1, len(train_loader)):.4f} | "
            f"Sens {metrics['sensitivity_abnormal_recall'] * 100:6.2f}% | "
            f"Spec {metrics['specificity_normal_recall'] * 100:6.2f}% | "
            f"MAcc {metrics['macc'] * 100:6.2f}% | "
            f"F1_abn {metrics['f1_abnormal'] * 100:6.2f}%"
        )

        if metrics["macc"] > best_macc:
            best_macc = float(metrics["macc"])
            best_epoch = epoch + 1
            best_preds = val_preds.copy()
            best_probs = val_probs.copy()
            best_labels = val_labels.copy()
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  New best {model_name} fold {fold + 1}: MAcc={best_macc * 100:.2f}%")

    best_metrics = compute_binary_metrics(np.array(best_labels), np.array(best_preds))
    best_metrics["fold"] = fold + 1
    best_metrics["model"] = model_name
    best_metrics["best_epoch"] = best_epoch
    best_metrics["checkpoint"] = str(checkpoint_path)

    prediction_rows = []
    for local_i, dataset_index in enumerate(val_idx.tolist()):
        file_path = getattr(base_dataset, "file_paths", [None] * len(base_dataset))[dataset_index]
        prediction_rows.append(
            {
                "model": model_name,
                "fold": fold + 1,
                "dataset_index": int(dataset_index),
                "file_path": file_path,
                "label": int(best_labels[local_i]),
                "prediction": int(best_preds[local_i]),
                "prob_abnormal": float(best_probs[local_i]),
                "correct": int(best_labels[local_i] == best_preds[local_i]),
            }
        )

    print(f"\nFinal fold {fold + 1} ({model_name})")
    print(json.dumps({k: v for k, v in best_metrics.items() if k not in {'confusion_matrix', 'checkpoint'}}, indent=2))
    print("Confusion matrix [[TN, FP], [FN, TP]]:", best_metrics["confusion_matrix"])

    return {"metrics": best_metrics, "predictions": prediction_rows}


def aggregate_fold_metrics(fold_metrics: List[Dict]) -> Dict[str, Dict]:
    keys = [
        "accuracy",
        "sensitivity_abnormal_recall",
        "specificity_normal_recall",
        "macc",
        "precision_abnormal",
        "recall_abnormal",
        "f1_abnormal",
        "macro_f1",
        "weighted_f1",
    ]
    summary = {}
    for key in keys:
        values = np.array([float(m[key]) for m in fold_metrics], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "std_population": float(values.std(ddof=0)),
            "std_sample": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "values": values.tolist(),
        }
    return summary


def run_mcnemar_from_prediction_rows(pred_rows: List[Dict], model_a: str, model_b: str) -> Dict:
    by_model: Dict[str, Dict[int, Dict]] = {model_a: {}, model_b: {}}
    for row in pred_rows:
        if row["model"] in by_model:
            by_model[row["model"]][int(row["dataset_index"])] = row

    common_indices = sorted(set(by_model[model_a]).intersection(by_model[model_b]))
    if not common_indices:
        return {"error": "No matched prediction indices found."}

    a_correct = np.array([by_model[model_a][i]["correct"] for i in common_indices], dtype=bool)
    b_correct = np.array([by_model[model_b][i]["correct"] for i in common_indices], dtype=bool)

    both_correct = int(np.sum(a_correct & b_correct))
    a_wrong_b_correct = int(np.sum(~a_correct & b_correct))
    a_correct_b_wrong = int(np.sum(a_correct & ~b_correct))
    both_wrong = int(np.sum(~a_correct & ~b_correct))
    table = [[both_correct, a_wrong_b_correct], [a_correct_b_wrong, both_wrong]]

    result = mcnemar(table, exact=False, correction=True)
    return {
        "model_a": model_a,
        "model_b": model_b,
        "n_matched": int(len(common_indices)),
        "contingency_table": table,
        "table_interpretation": "[[both correct, model_a wrong/model_b correct], [model_a correct/model_b wrong, both wrong]]",
        "p_value_chi2_corrected": float(result.pvalue),
        "statistic": float(result.statistic),
    }


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tag", type=str, default="final")
    parser.add_argument(
        "--models",
        type=str,
        default="proposed,resnet1d",
        help="Comma-separated list: proposed,resnet1d",
    )
    parser.add_argument("--base-dir", type=str, default=Config.base_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        tag=args.tag,
        base_dir=args.base_dir,
    )
    set_seed(cfg.seed)

    models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]
    out_dir = Path("results") / "ieee_cv" / cfg.tag
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"IEEE CV run tag: {cfg.tag}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Torch CUDA: {torch.version.cuda}")

    print("\nLoading datasets...")
    base_dataset = PhysioNet2016Dataset(base_dir=cfg.base_dir, folders=list(cfg.folders), augment=False)
    aug_dataset = PhysioNet2016Dataset(base_dir=cfg.base_dir, folders=list(cfg.folders), augment=True)
    labels = np.array(base_dataset.labels, dtype=int)
    indices = np.arange(len(base_dataset))

    print(f"Total samples: {len(base_dataset):,}")
    print(f"Normal: {(labels == 0).sum():,}")
    print(f"Abnormal: {(labels == 1).sum():,}")
    print(f"Class weights used: normal={cfg.class_weight_normal}, abnormal={cfg.class_weight_abnormal}")

    # Complexity/inference timing before training.
    complexity = {}
    for model_name in models_to_run:
        model = make_model(model_name, cfg).to(device)
        complexity[model_name] = count_parameters(model)
        complexity[model_name]["inference_time"] = measure_inference_time(
            model, device=device, batch_size=cfg.batch_size
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    fold_splits = list(skf.split(indices, labels))

    all_prediction_rows: List[Dict] = []
    per_model_fold_metrics: Dict[str, List[Dict]] = {m: [] for m in models_to_run}

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        for model_name in models_to_run:
            set_seed(cfg.seed + fold)  # same fold seed for each model
            result = train_one_model_one_fold(
                model_name=model_name,
                fold=fold,
                train_idx=train_idx,
                val_idx=val_idx,
                aug_dataset=aug_dataset,
                base_dataset=base_dataset,
                device=device,
                cfg=cfg,
                out_dir=ckpt_dir,
            )
            per_model_fold_metrics[model_name].append(result["metrics"])
            all_prediction_rows.extend(result["predictions"])
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Save predictions.
    write_csv(out_dir / "fold_predictions.csv", all_prediction_rows)

    # Per-class metrics CSV.
    metric_rows = []
    for model_name, rows in per_model_fold_metrics.items():
        for m in rows:
            metric_rows.append(m)
    write_csv(out_dir / "per_class_metrics.csv", metric_rows)

    # Aggregate summaries.
    summary = {
        model_name: aggregate_fold_metrics(rows)
        for model_name, rows in per_model_fold_metrics.items()
    }

    # OOF metrics by model.
    oof_metrics = {}
    for model_name in models_to_run:
        rows = [r for r in all_prediction_rows if r["model"] == model_name]
        y_true = np.array([r["label"] for r in rows], dtype=int)
        y_pred = np.array([r["prediction"] for r in rows], dtype=int)
        oof_metrics[model_name] = compute_binary_metrics(y_true, y_pred)

    # Statistical tests.
    statistical_tests = {}
    if "proposed" in models_to_run and "resnet1d" in models_to_run:
        statistical_tests["mcnemar_proposed_vs_resnet1d"] = run_mcnemar_from_prediction_rows(
            all_prediction_rows, "proposed", "resnet1d"
        )

    # Save JSON reports.
    cv_results = {
        "config": asdict(cfg),
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "dataset": {
            "total": int(len(base_dataset)),
            "normal": int((labels == 0).sum()),
            "abnormal": int((labels == 1).sum()),
        },
        "summary": summary,
        "oof_metrics": oof_metrics,
        "fold_metrics": per_model_fold_metrics,
    }
    with (out_dir / "cv_results.json").open("w") as f:
        json.dump(cv_results, f, indent=2)
    with (out_dir / "complexity_report.json").open("w") as f:
        json.dump(complexity, f, indent=2)
    with (out_dir / "inference_time_report.json").open("w") as f:
        json.dump({k: v["inference_time"] for k, v in complexity.items()}, f, indent=2)
    with (out_dir / "statistical_tests.json").open("w") as f:
        json.dump(statistical_tests, f, indent=2)

    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    for model_name in models_to_run:
        s = summary[model_name]
        print(f"\n{model_name}")
        for key in ["macc", "sensitivity_abnormal_recall", "specificity_normal_recall", "f1_abnormal", "macro_f1", "accuracy"]:
            print(
                f"  {key:<30}: "
                f"{s[key]['mean'] * 100:.2f}% ± {s[key]['std_sample'] * 100:.2f}%"
            )
        print(f"  OOF CM [[TN, FP], [FN, TP]]: {oof_metrics[model_name]['confusion_matrix']}")

    if statistical_tests:
        print("\nStatistical tests")
        print(json.dumps(statistical_tests, indent=2))

    print(f"\nSaved all outputs to: {out_dir}")
    print("Files:")
    for name in [
        "cv_results.json",
        "fold_predictions.csv",
        "per_class_metrics.csv",
        "complexity_report.json",
        "inference_time_report.json",
        "statistical_tests.json",
    ]:
        print(f"  - {out_dir / name}")


if __name__ == "__main__":
    main()
