"""
Volumetric rendering: turning per-sample (sigma, rgb) predictions along a
ray into a single composited pixel color. This is the classical NeRF
volume rendering equation, discretized -- the physics is "light passing
through a medium of varying density and color gets absorbed and emits
color proportionally to local density."
"""

from __future__ import annotations

import torch


def volume_render(
    sigma: torch.Tensor, rgb: torch.Tensor, z_vals: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Composite per-sample density/color into final pixel colors.

    Args:
        sigma: (N, n_samples) volume density at each sample.
        rgb: (N, n_samples, 3) color at each sample.
        z_vals: (N, n_samples) depth of each sample (used to compute the
            distance between consecutive samples).

    Returns:
        pixel_rgb: (N, 3) final composited color per ray.
        pixel_depth: (N,) expected depth per ray (weighted average of
            z_vals) -- useful for depth visualization/debugging, and later
            for LiDAR-style range comparisons.
        weights: (N, n_samples) the per-sample contribution weights,
            returned for diagnostic/visualization purposes (e.g. plotting
            where along the ray the model believes the surface is).
    """
    if sigma.shape != z_vals.shape:
        raise ValueError(f"sigma {sigma.shape} and z_vals {z_vals.shape} must match")
    if rgb.shape[:2] != sigma.shape:
        raise ValueError(f"rgb {rgb.shape} must match sigma {sigma.shape} on first 2 dims")

    # Distance between consecutive samples; the last sample gets a large
    # placeholder distance (effectively "to infinity") which is standard
    # NeRF practice so the final sample can fully terminate the ray.
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat([dists, torch.full_like(dists[:, :1], 1e10)], dim=-1)

    # Alpha = probability of the ray terminating exactly at this sample.
    alpha = 1.0 - torch.exp(-sigma * dists)

    # Transmittance = probability the ray survives (doesn't terminate)
    # through all PRIOR samples -- computed via cumulative product of
    # (1 - alpha), shifted by one so sample i only accounts for samples
    # before it, not itself.
    ones = torch.ones_like(alpha[:, :1])
    transmittance = torch.cumprod(
        torch.cat([ones, 1.0 - alpha + 1e-10], dim=-1), dim=-1
    )[:, :-1]

    weights = alpha * transmittance  # (N, n_samples)

    pixel_rgb = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)
    pixel_depth = torch.sum(weights * z_vals, dim=1)

    return pixel_rgb, pixel_depth, weights
