"""
Smoke test for the NeRF reconstruction pipeline: builds a tiny synthetic
mock KITTI sequence, runs it through prepare_dataset -> NeRF training for
a handful of steps, and checks that (a) nothing crashes and (b) training
loss decreases -- i.e. the model is actually learning to overfit this tiny
scene, which is the right first sanity check before trusting the pipeline
on a real dataset.

This does NOT require CUDA -- it runs on CPU, just slowly. Run it once
after setting up your environment (see README) to confirm everything is
wired correctly before starting a real training run.

Usage:
    python -m scripts.smoke_test_nerf
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data_ingestion.prepare_dataset import prepare_kitti
from src.reconstruction.nerf.dataset import NeRFRayDataset
from src.reconstruction.nerf.hash_encoding import MultiresHashEncoding
from src.reconstruction.nerf.model import HashGridNeRF
from src.reconstruction.nerf.train import render_rays
from src.metrics.psnr import compute_psnr


def build_mock_kitti(raw_path: Path, sequence_id: str = "00", n_frames: int = 12) -> None:
    sequence_dir = raw_path / "sequences" / sequence_id
    sequence_dir.mkdir(parents=True)

    p2 = "50.0 0.0 32.0 0.0 0.0 50.0 24.0 0.0 0.0 0.0 1.0 0.0"
    (sequence_dir / "calib.txt").write_text(
        f"P0: {p2}\nP1: {p2}\nP2: {p2}\nP3: {p2}\nTr: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    )
    (sequence_dir / "times.txt").write_text(
        "\n".join(f"{i * 0.1:.6f}" for i in range(n_frames)) + "\n"
    )

    image_dir = sequence_dir / "image_2"
    image_dir.mkdir()
    for i in range(n_frames):
        # A simple gradient pattern (not random noise) so there's actual
        # structure for the model to learn -- pure noise would make the
        # "loss decreases" check meaningless.
        arr = np.zeros((48, 64, 3), dtype=np.uint8)
        arr[:, :, 0] = np.linspace(0, 255, 64, dtype=np.uint8)[None, :]
        arr[:, :, 1] = (i * 20) % 255
        Image.fromarray(arr).save(image_dir / f"{i:06d}.png")

    poses_dir = raw_path / "poses"
    poses_dir.mkdir()
    lines = []
    for i in range(n_frames):
        # Cameras arranged on a small arc, all looking toward the origin-ish
        # region, so rays actually converge on the same scene content.
        angle = (i / n_frames) * 0.6 - 0.3
        x = 5.0 * np.sin(angle)
        z = 5.0 * np.cos(angle)
        row = [1, 0, 0, x, 0, 1, 0, 0.0, 0, 0, 1, z]
        lines.append(" ".join(str(v) for v in row))
    (poses_dir / f"{sequence_id}.txt").write_text("\n".join(lines) + "\n")


def main():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        raw_path = tmp_dir / "mock_kitti"
        processed_path = tmp_dir / "processed"
        build_mock_kitti(raw_path)

        config = {
            "dataset": {
                "name": "kitti",
                "sequence": "00",
                "raw_path": str(raw_path),
                "processed_path": str(processed_path),
                "val_every_n_frames": 4,
            }
        }
        prepare_kitti(config)

        train_dataset = NeRFRayDataset(processed_path / "00_train.npz")

        # Deliberately tiny model/grid for a fast CPU smoke test -- not
        # representative of real training quality, only of correctness.
        hash_encoding = MultiresHashEncoding(
            n_levels=4, n_features_per_level=2, log2_hashmap_size=12,
            base_resolution=8, finest_resolution=64,
        )
        model = HashGridNeRF(hash_encoding=hash_encoding, density_hidden_dim=32, color_hidden_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        losses = []
        n_steps = 30
        for step in range(n_steps):
            ray_o, ray_d, target = train_dataset.sample_random_rays(256)
            pred = render_rays(model, ray_o, ray_d, train_dataset.near, train_dataset.far, n_samples=32)
            loss = torch.nn.functional.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if step % 5 == 0:
                psnr = compute_psnr(pred.detach(), target)
                print(f"step {step:3d} | loss {loss.item():.5f} | PSNR {psnr:.2f} dB")

        avg_first_5 = sum(losses[:5]) / 5
        avg_last_5 = sum(losses[-5:]) / 5
        print(f"\nAvg loss, first 5 steps: {avg_first_5:.5f}")
        print(f"Avg loss, last 5 steps:  {avg_last_5:.5f}")

        assert avg_last_5 < avg_first_5, (
            "Loss did not decrease -- something in the pipeline (hash encoding, "
            "rays, or volume rendering) likely has a bug. Do not proceed to a "
            "real training run until this passes."
        )
        print("\nSMOKE TEST PASSED: loss decreased, pipeline is wired correctly.")

    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
