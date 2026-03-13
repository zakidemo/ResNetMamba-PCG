"""
utils/trainer.py — Training utilities.

Helper functions for the training loop.
Main training logic is in scripts/cross_validate.py.
"""

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
