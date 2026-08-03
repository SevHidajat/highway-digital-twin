"""
Multiresolution hash encoding, as introduced in Instant-NGP (Muller et al.,
2022). This is what makes training tractable in minutes instead of hours,
compared to the original NeRF's single large MLP over raw (x,y,z).

Core idea: instead of feeding raw 3D coordinates into an MLP (which forces
the network to learn high-frequency detail from scratch, slowly), we look
up learned feature vectors from a set of multi-resolution grids -- coarse
grids for large-scale structure, fine grids for detail -- and let a tiny
MLP interpret those features. Fine grids would need an enormous number of
vertices to cover a whole scene, so above a resolution threshold we hash
grid-cell coordinates into a fixed-size table instead of indexing directly
(hash collisions are tolerated -- gradient descent learns to resolve them
because colliding cells are rarely both important in the same way).
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Large primes used for spatial hashing, per the Instant-NGP paper.
_HASH_PRIMES = torch.tensor([1, 2654435761, 805459861], dtype=torch.int64)


class MultiresHashEncoding(nn.Module):
    """Maps 3D points in [0, 1]^3 to a concatenated multi-level feature vector.

    Args:
        n_levels: number of resolution levels (coarse to fine).
        n_features_per_level: feature dimensionality stored per grid vertex.
        log2_hashmap_size: hash table size per level = 2^log2_hashmap_size.
        base_resolution: grid resolution at the coarsest level.
        finest_resolution: grid resolution at the finest level. The growth
            factor between levels is derived automatically from this.
    """

    def __init__(
        self,
        n_levels: int = 8,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 15,
        base_resolution: int = 16,
        finest_resolution: int = 512,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.hashmap_size = 2 ** log2_hashmap_size
        self.base_resolution = base_resolution

        # Geometric growth factor between levels so that resolution goes
        # from base_resolution (level 0) to finest_resolution (last level).
        if n_levels > 1:
            self.growth_factor = (finest_resolution / base_resolution) ** (
                1.0 / (n_levels - 1)
            )
        else:
            self.growth_factor = 1.0

        # One embedding table per level. Using nn.ParameterList (not a single
        # tensor) keeps per-level tables independently sized in principle,
        # though here all levels share hashmap_size for simplicity.
        self.embeddings = nn.ParameterList(
            [
                nn.Parameter(
                    (torch.rand(self.hashmap_size, n_features_per_level) * 2 - 1) * 1e-4
                )
                for _ in range(n_levels)
            ]
        )

        self.register_buffer("hash_primes", _HASH_PRIMES)

    @property
    def output_dim(self) -> int:
        return self.n_levels * self.n_features_per_level

    def _resolution_at_level(self, level: int) -> int:
        return int(self.base_resolution * (self.growth_factor**level))

    def _hash(self, grid_coords: torch.Tensor) -> torch.Tensor:
        """Spatial hash of integer grid coordinates -> table index.

        Args:
            grid_coords: (..., 3) integer tensor.
        Returns:
            (...,) tensor of indices into [0, hashmap_size).
        """
        primes = self.hash_primes.to(grid_coords.device)
        # XOR-based spatial hash from the Instant-NGP paper.
        hashed = grid_coords[..., 0] * primes[0]
        hashed = hashed ^ (grid_coords[..., 1] * primes[1])
        hashed = hashed ^ (grid_coords[..., 2] * primes[2])
        return (hashed % self.hashmap_size).long()

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (N, 3) tensor of coordinates in [0, 1]^3.
        Returns:
            (N, n_levels * n_features_per_level) encoded features.
        """
        if points.min() < -1e-4 or points.max() > 1 + 1e-4:
            raise ValueError(
                "MultiresHashEncoding expects points normalized to [0, 1]^3; "
                f"got range [{points.min().item():.4f}, {points.max().item():.4f}]. "
                "Normalize scene coordinates before encoding."
            )

        outputs = []
        for level in range(self.n_levels):
            resolution = self._resolution_at_level(level)
            scaled = points * resolution  # (N, 3), continuous grid coords

            floor_coords = torch.floor(scaled).long()
            frac = scaled - floor_coords.float()  # (N, 3) interpolation weights

            # Gather the 8 corners of the enclosing voxel and trilinearly
            # interpolate their embeddings.
            level_table = self.embeddings[level]
            interpolated = torch.zeros(
                points.shape[0], self.n_features_per_level, device=points.device
            )
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        corner = floor_coords + torch.tensor(
                            [dx, dy, dz], device=points.device
                        )
                        weight = (
                            (frac[:, 0] if dx else (1 - frac[:, 0]))
                            * (frac[:, 1] if dy else (1 - frac[:, 1]))
                            * (frac[:, 2] if dz else (1 - frac[:, 2]))
                        )
                        # Direct indexing is only valid if the grid is small
                        # enough to fit without collision; hash otherwise.
                        n_vertices_if_dense = (resolution + 1) ** 3
                        if n_vertices_if_dense <= self.hashmap_size:
                            idx = (
                                corner[:, 0] * (resolution + 1) ** 2
                                + corner[:, 1] * (resolution + 1)
                                + corner[:, 2]
                            ) % self.hashmap_size
                        else:
                            idx = self._hash(corner)
                        interpolated += weight.unsqueeze(-1) * level_table[idx]

            outputs.append(interpolated)

        return torch.cat(outputs, dim=-1)
