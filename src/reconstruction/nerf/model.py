"""
The NeRF network itself: a small MLP that consumes hash-encoded position
features (+ a lightweight view-direction encoding) and predicts volume
density (sigma) and view-dependent color (RGB).

Kept deliberately small (a few dozen thousand parameters) -- with a good
positional encoding, NeRF's representational power comes mostly from the
encoding, not a deep network. This is the opposite trade-off from the
original NeRF paper's 8-layer, 256-wide MLP over raw coordinates.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.reconstruction.nerf.hash_encoding import MultiresHashEncoding


def sinusoidal_encoding(x: torch.Tensor, n_freqs: int) -> torch.Tensor:
    """Simple Fourier feature encoding for view directions (not hashed --
    direction only has 2 effective degrees of freedom on the unit sphere,
    so a hash grid would be overkill; a handful of sin/cos bands suffices).

    Args:
        x: (N, 3) tensor, expected roughly in [-1, 1] (unit direction vectors).
        n_freqs: number of frequency bands.
    Returns:
        (N, 3 * 2 * n_freqs) encoded tensor.
    """
    freq_bands = 2.0 ** torch.arange(n_freqs, device=x.device, dtype=x.dtype)
    x_scaled = x.unsqueeze(-1) * freq_bands  # (N, 3, n_freqs)
    return torch.cat([torch.sin(x_scaled), torch.cos(x_scaled)], dim=-1).flatten(1)


class HashGridNeRF(nn.Module):
    """Hash-grid encoded NeRF for a single reconstructed scene.

    Forward pass takes sample points along rays plus the ray's view
    direction, and returns (sigma, rgb) for each point -- consumed by the
    volumetric renderer in renderer.py to composite a final pixel color.
    """

    def __init__(
        self,
        hash_encoding: MultiresHashEncoding | None = None,
        density_hidden_dim: int = 64,
        color_hidden_dim: int = 64,
        view_dir_freqs: int = 4,
    ):
        super().__init__()
        self.hash_encoding = hash_encoding or MultiresHashEncoding()
        self.view_dir_freqs = view_dir_freqs
        view_dir_dim = 3 * 2 * view_dir_freqs

        # Density head: hash features -> sigma + a feature vector passed to
        # the color head (this feature vector is what lets color depend on
        # both position AND view direction, matching NeRF's core design:
        # density should NOT depend on view direction -- a physical surface
        # has one true occupancy regardless of where you look from -- but
        # color legitimately can, e.g. specular highlights).
        self.density_net = nn.Sequential(
            nn.Linear(self.hash_encoding.output_dim, density_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(density_hidden_dim, density_hidden_dim + 1),
        )

        self.color_net = nn.Sequential(
            nn.Linear(density_hidden_dim + view_dir_dim, color_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(color_hidden_dim, 3),
        )

    def forward(
        self, points: torch.Tensor, view_dirs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points: (N, 3) points in normalized [0, 1]^3 scene coordinates.
            view_dirs: (N, 3) unit view direction for each point (same
                direction repeated for every sample along a given ray).
        Returns:
            sigma: (N,) non-negative volume density.
            rgb: (N, 3) color in [0, 1].
        """
        encoded_pos = self.hash_encoding(points)
        density_out = self.density_net(encoded_pos)
        sigma_raw, density_features = density_out[:, 0], density_out[:, 1:]
        # Softplus keeps density non-negative and smooth near zero (avoids
        # the dead-ReLU problem you'd get from a hard relu on raw sigma).
        sigma = torch.nn.functional.softplus(sigma_raw)

        encoded_dir = sinusoidal_encoding(view_dirs, self.view_dir_freqs)
        color_input = torch.cat([density_features, encoded_dir], dim=-1)
        rgb = torch.sigmoid(self.color_net(color_input))

        return sigma, rgb
