"""
NeRF-family reconstruction training loop for a highway corridor segment.

Trains a HashGridNeRF (see model.py) to reconstruct a scene from the
train split, periodically rendering held-out val images to track PSNR --
this is the "ray-tracing" side of the eventual comparison against the
Gaussian Splatting path (see docs/PROJECT_SPEC.md Section 3).

Usage:
    python -m src.reconstruction.nerf.train --config configs/kitti_baseline.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
import yaml

from src.metrics.psnr import compute_psnr
from src.reconstruction.nerf.dataset import NeRFRayDataset
from src.reconstruction.nerf.hash_encoding import MultiresHashEncoding
from src.reconstruction.nerf.model import HashGridNeRF
from src.reconstruction.nerf.rays import sample_points_along_rays
from src.reconstruction.nerf.renderer import volume_render

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def render_rays(
    model: HashGridNeRF,
    ray_origins: torch.Tensor,
    ray_dirs: torch.Tensor,
    near: float,
    far: float,
    n_samples: int,
) -> torch.Tensor:
    """Full forward pass: sample points along rays, query the model, and
    volume-render to final pixel colors. Shared by both the training step
    and validation rendering, so there is exactly one code path for
    "how do we turn rays into pixels" -- avoiding subtle train/val mismatch
    bugs.
    """
    points, z_vals = sample_points_along_rays(ray_origins, ray_dirs, near, far, n_samples)
    n_rays, n_pts = points.shape[0], points.shape[1]

    # Points must be in [0, 1]^3 for the hash grid; our dataset normalizes
    # camera positions to roughly [-1, 1], so remap here to [0, 1].
    points_01 = (points + 1.0) * 0.5
    points_01 = points_01.clamp(0.0, 1.0)

    flat_points = points_01.reshape(-1, 3)
    flat_dirs = ray_dirs.unsqueeze(1).expand(-1, n_pts, -1).reshape(-1, 3)

    sigma, rgb = model(flat_points, flat_dirs)
    sigma = sigma.reshape(n_rays, n_pts)
    rgb = rgb.reshape(n_rays, n_pts, 3)

    pixel_rgb, _, _ = volume_render(sigma, rgb, z_vals)
    return pixel_rgb


def render_image_chunked(
    model: HashGridNeRF,
    ray_origins: torch.Tensor,
    ray_dirs: torch.Tensor,
    near: float,
    far: float,
    n_samples: int,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Render a full image's rays in chunks to bound memory use -- an image
    has H*W rays, which won't fit through the model in one batch once you
    multiply by n_samples points per ray.
    """
    outputs = []
    with torch.no_grad():
        for start in range(0, ray_origins.shape[0], chunk_size):
            end = start + chunk_size
            chunk_rgb = render_rays(
                model, ray_origins[start:end], ray_dirs[start:end], near, far, n_samples
            )
            outputs.append(chunk_rgb)
    return torch.cat(outputs, dim=0)


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(config: dict) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    dataset_cfg = config["dataset"]
    nerf_cfg = config["reconstruction"]["nerf"]
    sequence_id = dataset_cfg["sequence"]
    processed_path = Path(dataset_cfg["processed_path"])

    train_npz = processed_path / f"{sequence_id}_train.npz"
    val_npz = processed_path / f"{sequence_id}_val.npz"
    if not train_npz.exists():
        raise FileNotFoundError(
            f"{train_npz} not found -- run "
            f"'python -m src.data_ingestion.prepare_dataset --config <config>' first"
        )

    train_dataset = NeRFRayDataset(train_npz)
    val_dataset = NeRFRayDataset(
        val_npz, scene_center=train_dataset.scene_center, scene_scale=train_dataset.scene_scale
    )
    logger.info(f"Loaded {len(train_dataset)} train / {len(val_dataset)} val frames")

    hash_encoding = MultiresHashEncoding(
        n_levels=nerf_cfg.get("n_levels", 8),
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        finest_resolution=512,
    )
    model = HashGridNeRF(
        hash_encoding=hash_encoding, density_hidden_dim=nerf_cfg["hidden_dim"], color_hidden_dim=nerf_cfg["hidden_dim"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=nerf_cfg["max_steps"]
    )

    batch_size = 4096
    n_samples_per_ray = 64
    near, far = train_dataset.near, train_dataset.far
    log_every = 100
    val_every = 1000
    checkpoint_dir = Path("outputs/checkpoints/nerf")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    for step in range(nerf_cfg["max_steps"]):
        model.train()
        ray_origins, ray_dirs, target_rgb = train_dataset.sample_random_rays(batch_size)
        ray_origins, ray_dirs, target_rgb = (
            ray_origins.to(device),
            ray_dirs.to(device),
            target_rgb.to(device),
        )

        pred_rgb = render_rays(model, ray_origins, ray_dirs, near, far, n_samples_per_ray)
        loss = torch.nn.functional.mse_loss(pred_rgb, target_rgb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % log_every == 0:
            psnr = compute_psnr(pred_rgb.detach(), target_rgb)
            elapsed = time.time() - start_time
            logger.info(
                f"step {step:6d} | loss {loss.item():.5f} | train PSNR {psnr:5.2f} dB | "
                f"{elapsed:.1f}s elapsed"
            )

        if step % val_every == 0 and step > 0:
            model.eval()
            val_idx = 0  # spot-check the first val image each time for a consistent trend line
            ray_o, ray_d, target = val_dataset.get_image_rays(val_idx)
            ray_o, ray_d, target = ray_o.to(device), ray_d.to(device), target.to(device)
            pred = render_image_chunked(model, ray_o, ray_d, near, far, n_samples_per_ray)
            val_psnr = compute_psnr(pred, target)
            logger.info(f"  -> val PSNR (frame {val_idx}): {val_psnr:.2f} dB")

            torch.save(
                {"step": step, "model_state_dict": model.state_dict(), "val_psnr": val_psnr},
                checkpoint_dir / f"step_{step:06d}.pt",
            )

    logger.info("Training complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(Path(args.config))
    train(config)


if __name__ == "__main__":
    main()
