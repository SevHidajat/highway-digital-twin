"""
Dataset wrapper for NeRF training: loads the .npz cache produced by
src/data_ingestion/prepare_dataset.py, converts poses into the convention
this NeRF implementation expects, normalizes the scene into a unit cube
(hash grids are defined over [0,1]^3, so raw KITTI world coordinates --
which can span hundreds of meters along a highway -- must be rescaled),
and exposes both random-ray sampling (for training) and full-image ray
generation (for validation/rendering).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data_ingestion.kitti_loader import convert_pose_opencv_to_opengl
from src.reconstruction.nerf.rays import get_rays_for_pixels


class NeRFRayDataset:
    """Loads one train or val split (.npz) and serves rays for training.

    Scene normalization: camera positions are recentered on their centroid
    and rescaled so all camera positions fit within roughly [-1, 1]. This
    is necessary because:
        1. The hash grid encoding expects inputs in [0,1]^3.
        2. A shared normalization must be computed from TRAIN data only and
           reused for val -- otherwise train/val scenes would be normalized
           inconsistently and poses would not be comparable.
    """

    def __init__(
        self,
        npz_path: Path,
        near: float = 0.1,
        far: float = 2.0,
        scene_center: np.ndarray | None = None,
        scene_scale: float | None = None,
    ):
        data = np.load(npz_path)
        self.frame_ids = data["frame_ids"]
        self.image_paths = data["image_paths"]
        raw_poses = data["poses"]  # (N, 4, 4), OpenCV convention from kitti_loader
        self.intrinsics = data["intrinsics"]  # (N, 4): fx, fy, cx, cy
        self.image_width = int(data["image_width"])
        self.image_height = int(data["image_height"])

        opengl_poses = np.stack([convert_pose_opencv_to_opengl(p) for p in raw_poses])

        camera_positions = opengl_poses[:, :3, 3]
        if scene_center is None:
            scene_center = camera_positions.mean(axis=0)
        if scene_scale is None:
            # Scale so the furthest camera from center lands near radius 1.
            max_dist = np.linalg.norm(camera_positions - scene_center, axis=1).max()
            scene_scale = 1.0 / max(max_dist, 1e-6)

        self.scene_center = scene_center
        self.scene_scale = scene_scale

        normalized_poses = opengl_poses.copy()
        normalized_poses[:, :3, 3] = (camera_positions - scene_center) * scene_scale
        self.poses = torch.from_numpy(normalized_poses).float()

        self.near = near
        self.far = far
        self._image_cache: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.frame_ids)

    def _load_image(self, idx: int) -> torch.Tensor:
        """Load an image, caching it in memory after the first read.

        Without this cache, sample_random_rays would re-read and re-decode
        a .png file from disk every time that image is touched -- and since
        a random batch of rays scatters across a large fraction of all
        training images, that meant thousands of disk reads PER TRAINING
        STEP on a real dataset (this is what caused training to appear
        "stuck" on real KITTI data despite working fine on the small
        synthetic smoke-test scene, where there were only ~10 images total).
        """
        if idx not in self._image_cache:
            img = Image.open(self.image_paths[idx]).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            self._image_cache[idx] = torch.from_numpy(arr)
        return self._image_cache[idx]

    def get_image_rays(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return every ray for a full image (for validation/rendering).

        Returns:
            ray_origins: (H*W, 3)
            ray_dirs: (H*W, 3)
            target_rgb: (H*W, 3) ground-truth pixel colors.
        """
        fx, fy, cx, cy = self.intrinsics[idx]
        ys, xs = torch.meshgrid(
            torch.arange(self.image_height, dtype=torch.float32),
            torch.arange(self.image_width, dtype=torch.float32),
            indexing="ij",
        )
        pixel_x, pixel_y = xs.flatten(), ys.flatten()

        ray_origins, ray_dirs = get_rays_for_pixels(
            pixel_x, pixel_y, self.poses[idx], fx, fy, cx, cy
        )
        target_rgb = self._load_image(idx).reshape(-1, 3)

        return ray_origins, ray_dirs, target_rgb

    def sample_random_rays(
        self, batch_size: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random batch of rays, potentially from different images.

        This is the standard NeRF training pattern: rather than training on
        one full image at a time, sample a random scatter of pixels across
        (possibly) many images each step -- this gives more diverse gradient
        signal per step and avoids overfitting to one viewpoint's noise.
        """
        n_images = len(self)
        image_indices = torch.randint(0, n_images, (batch_size,), generator=generator)

        ray_origins = torch.empty(batch_size, 3)
        ray_dirs = torch.empty(batch_size, 3)
        target_rgb = torch.empty(batch_size, 3)

        # Group by image to avoid reloading the same image file repeatedly.
        for img_idx in image_indices.unique():
            mask = image_indices == img_idx
            n_in_image = int(mask.sum())

            pixel_x = torch.randint(
                0, self.image_width, (n_in_image,), generator=generator
            ).float()
            pixel_y = torch.randint(
                0, self.image_height, (n_in_image,), generator=generator
            ).float()

            fx, fy, cx, cy = self.intrinsics[int(img_idx)]
            o, d = get_rays_for_pixels(
                pixel_x, pixel_y, self.poses[int(img_idx)], fx, fy, cx, cy
            )

            image = self._load_image(int(img_idx))
            colors = image[pixel_y.long(), pixel_x.long()]

            ray_origins[mask] = o
            ray_dirs[mask] = d
            target_rgb[mask] = colors

        return ray_origins, ray_dirs, target_rgb
