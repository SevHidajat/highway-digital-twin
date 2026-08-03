"""
Basic image-fidelity metric shared across reconstruction methods (NeRF
and, later, 3DGS). SSIM/LPIPS will be added in src/metrics/run_benchmark.py
once both reconstruction pipelines exist and can be compared side by side --
PSNR alone is enough to sanity-check a single method's training progress.
"""

from __future__ import annotations

import torch


def compute_psnr(pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> float:
    """Peak signal-to-noise ratio between predicted and target images/pixels.

    Args:
        pred_rgb, target_rgb: tensors of the same shape, values in [0, 1].
    Returns:
        PSNR in decibels (higher is better; >30 dB is generally considered
        good reconstruction quality for natural images).
    """
    if pred_rgb.shape != target_rgb.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_rgb.shape} vs target {target_rgb.shape}"
        )
    mse = torch.mean((pred_rgb - target_rgb) ** 2).item()
    if mse <= 0:
        return float("inf")
    return -10.0 * torch.log10(torch.tensor(mse)).item()
