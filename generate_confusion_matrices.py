"""
generate_confusion_matrices.py
===============================
Generates a single figure with 5 confusion matrices (one per ablation model).
Run from your project root where results/ folder contains the JSON files.

Usage:
    python generate_confusion_matrices.py

Output:
    confusion_matrices.pdf  → upload to Overleaf
    confusion_matrices.png  → preview
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# ══════════════════════════════════════════════════════════════
# CONFIG — adjust paths if needed
# ══════════════════════════════════════════════════════════════
RESULTS = {
    "ResNet-1D\n(Matched)": "results/cv_resnet_matched.json",
    "BiMamba\n-only":       "results/cv_mamba_pcg.json",
    "ResNetMamba\n(Ours)":  "results/cv_resnet_mamba.json",
    "Wavelet+\nResNet":     "results/cv_wavelet_resnet.json",
    "WaveResNet\nMamba":    "results/cv_wavemamba.json",
}

OUTPUT_PDF = "confusion_matrices.pdf"
OUTPUT_PNG = "confusion_matrices.png"
CLASS_NAMES = ["Normal", "Abnormal"]


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def load_cm(json_path):
    """Load JSON and compute confusion matrix from all_true/all_pred."""
    with open(json_path) as f:
        data = json.load(f)
    y_true = np.array(data["all_true"])
    y_pred = np.array(data["all_pred"])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return cm, data.get("summary", {})


def main():
    # IEEE formatting
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'text.usetex': False,
    })

    n_models = len(RESULTS)
    fig, axes = plt.subplots(1, n_models, figsize=(7.16, 1.8),
                             gridspec_kw={'wspace': 0.4})

    # Color maps — highlight the proposed model
    cmap_default = plt.cm.Blues
    cmap_ours = plt.cm.Greens

    for idx, (name, path) in enumerate(RESULTS.items()):
        ax = axes[idx]
        try:
            cm, summary = load_cm(path)
        except FileNotFoundError:
            ax.text(0.5, 0.5, f"File not found:\n{path}",
                    ha='center', va='center', fontsize=7,
                    transform=ax.transAxes)
            ax.set_title(name, fontsize=8, fontweight='bold')
            continue

        # Choose colormap
        is_ours = "Ours" in name
        cmap = cmap_ours if is_ours else cmap_default

        # Normalize for color intensity
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        # Plot
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap,
                       vmin=0, vmax=1, aspect='equal')

        # Add text annotations
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = cm_norm[i, j] * 100
                color = "white" if cm_norm[i, j] > 0.6 else "black"
                ax.text(j, i, f"{val}\n({pct:.1f}%)",
                        ha='center', va='center', fontsize=7,
                        fontweight='bold' if i == j else 'normal',
                        color=color)

        # Labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(CLASS_NAMES, fontsize=7)
        ax.set_yticklabels(CLASS_NAMES, fontsize=7, rotation=90,
                           va='center')

        if idx == 0:
            ax.set_ylabel('True Label', fontsize= 8)
        ax.set_xlabel('Predicted', fontsize=7)

        # Title with MAcc
        macc_mean = summary.get("macc_mean", 0)
        macc_std = summary.get("macc_std", 0)
        title_str = f"{name}\nMAcc: {macc_mean:.2f}±{macc_std:.2f}%"
        ax.set_title(title_str, fontsize=7.5,
                     fontweight='bold' if is_ours else 'normal',
                     color='#1B5E20' if is_ours else 'black')

        # Border for proposed model
        if is_ours:
            for spine in ax.spines.values():
                spine.set_edgecolor('#2E7D32')
                spine.set_linewidth(2)

    # Save
    fig.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight',
                pad_inches=0.03)
    fig.savefig(OUTPUT_PNG, format='png', bbox_inches='tight',
                pad_inches=0.03, dpi=300)
    plt.close()

    print(f"Saved: {OUTPUT_PDF}")
    print(f"Saved: {OUTPUT_PNG}")
    print("Done!")


if __name__ == "__main__":
    main()
