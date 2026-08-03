"""
Ray generation: converting (camera pose, intrinsics, pixel coordinates)
into 3D rays in world space -- the geometric bridge between "which pixel
are we rendering" and "what points in 3D does the model need to evaluate."
"""

from __future__ import annotations

import torch


def get_rays_for_pixels(
    pixel_x: torch.Tensor,
    pixel_y: torch.Tensor,
    pose: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ray origin and direction for a batch of pixel coordinates.

    Args:
        pixel_x, pixel_y: (N,) pixel coordinates (can be fractional, for
            sub-pixel sampling during training).
        pose: (4, 4) camera-to-world transform, OpenGL/COLMAP convention
            (x-right, y-up, z-backward -- i.e. the camera looks down -z).
            If your poses came from kitti_loader.load_sequence, run them
            through convert_pose_opencv_to_opengl first.
        fx, fy, cx, cy: pinhole intrinsics.

    Returns:
        ray_origins: (N, 3) world-space ray origins (all equal to the
            camera center, but returned per-pixel for a uniform batched API).
        ray_dirs: (N, 3) normalized world-space ray directions.
    """
    if pose.shape != (4, 4):
        raise ValueError(f"pose must be 4x4, got {pose.shape}")

    # Directions in camera space (OpenGL convention: camera looks down -z).
    dirs_camera = torch.stack(
        [
            (pixel_x - cx) / fx,
            -(pixel_y - cy) / fy,
            -torch.ones_like(pixel_x),
        ],
        dim=-1,
    )  # (N, 3)

    rotation = pose[:3, :3]
    translation = pose[:3, 3]

    # Rotate camera-space directions into world space; no translation applied
    # to directions (they're vectors, not points).
    dirs_world = dirs_camera @ rotation.T
    dirs_world = dirs_world / torch.norm(dirs_world, dim=-1, keepdim=True)

    origins_world = translation.unsqueeze(0).expand(dirs_world.shape[0], -1)

    return origins_world, dirs_world


def sample_points_along_rays(
    ray_origins: torch.Tensor,
    ray_dirs: torch.Tensor,
    near: float,
    far: float,
    n_samples: int,
    stratified: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample 3D points along each ray between near and far bounds.

    Args:
        ray_origins, ray_dirs: (N, 3) each.
        near, far: scalar depth bounds (in the same units as the scene's
            normalized coordinate system -- see note in dataset.py about
            scene normalization).
        n_samples: number of samples per ray.
        stratified: if True, jitter sample positions randomly within each
            depth bin (standard NeRF trick -- prevents the model from
            overfitting to a fixed set of depths and improves fine detail
            when combined with hierarchical sampling; here we keep it
            simple with stratified sampling only, no importance-sampled
            second pass, as a deliberate v1 scope cut).

    Returns:
        points: (N, n_samples, 3) world-space sample points.
        z_vals: (N, n_samples) depth of each sample along its ray.
    """
    device = ray_origins.device
    t_vals = torch.linspace(0.0, 1.0, n_samples, device=device)
    z_vals = near * (1 - t_vals) + far * t_vals  # (n_samples,)
    z_vals = z_vals.unsqueeze(0).expand(ray_origins.shape[0], -1).clone()  # (N, n_samples)

    if stratified:
        mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, :1], mids], dim=-1)
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand

    points = (
        ray_origins.unsqueeze(1) + ray_dirs.unsqueeze(1) * z_vals.unsqueeze(-1)
    )  # (N, n_samples, 3)

    return points, z_vals
