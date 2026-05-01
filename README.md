# ResNetMamba-PCG

Official code for the IEEE Signal Processing Letters submission:

**ResNetMamba-PCG: Bidirectional State Space Modeling for Phonocardiogram Classification**

This repository contains PyTorch code for binary phonocardiogram (PCG) classification using 1D residual convolutional front-ends and bidirectional Mamba state-space modeling.

## What is included

- ResNet-1D baseline
- ResNetMamba / WaveResNetMamba model variants
- 5-fold stratified cross-validation
- Per-class precision, recall, and F1-score
- Macro-F1 and balanced accuracy
- Parameter count and inference-time reporting
- Matched statistical comparison against a ResNet-1D baseline
- Final validation outputs under `results/ieee_cv/final/`

## Dataset

The dataset is **not included** in this repository.

Download the PhysioNet/CinC 2016 heart sound dataset from PhysioNet and place it under:

```text
data/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0/
```

Expected subfolders:

```text
training-a/
training-b/
training-c/
training-d/
training-e/
training-f/
```

## Installation

See [INSTALL.md](INSTALL.md).

## Smoke test

Use this only to verify that the pipeline works:

```bash
python cross_validate_ieee.py --epochs 1 --n-folds 2 --tag smoke
```

The smoke test is **not** paper evidence.

## Final 5-fold experiment

To reproduce the final paper-validation run, use:

```bash
python cross_validate_ieee.py --epochs 60 --n-folds 5 --tag final
```

This may take several hours on a GPU. Precomputed final result files are provided under:

```text
results/ieee_cv/final/
```

## Final output files

The final experiment produces:

```text
results/ieee_cv/final/cv_results.json
results/ieee_cv/final/fold_predictions.csv
results/ieee_cv/final/per_class_metrics.csv
results/ieee_cv/final/complexity_report.json
results/ieee_cv/final/inference_time_report.json
results/ieee_cv/final/statistical_tests.json
```

## Important notes

- The dataset is not redistributed in this repository.
- Trained checkpoints are not included.
- Large files such as `.wav`, `.pth`, `.pt`, `.zip`, and dataset folders are ignored.
- The smoke test is only for checking the pipeline and should not be used as paper evidence.
- Avoid reporting results produced with any CPU fallback implementation as official paper results.

## Citation

If this work is useful, please cite the corresponding IEEE Signal Processing Letters manuscript after publication.
