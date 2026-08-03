"""
Shared pytest fixtures.

We don't have the real KITTI dataset available in every environment (it's
several GB and requires a manual license-accepted download), so tests
build a small synthetic mock sequence on disk with the same directory
layout and file formats as real KITTI odometry data. This validates the
*parsing logic* thoroughly without depending on external data -- the same
approach you'd want in a production repo where CI can't download
multi-GB datasets on every run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _write_calib(calib_path: Path) -> None:
    # Simple, valid-looking rectified projection matrices. Values are
    # plausible KITTI-scale focal lengths/principal points, not real
    # calibration -- fine for testing parsing logic.
    p2 = "718.856 0.0 607.19 0.0 0.0 718.856 185.22 0.0 0.0 0.0 1.0 0.0"
    p0 = p1 = p3 = p2
    tr = "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0"
    calib_path.write_text(
        f"P0: {p0}\nP1: {p1}\nP2: {p2}\nP3: {p3}\nTr: {tr}\n"
    )


def _write_poses(poses_path: Path, n_frames: int) -> None:
    lines = []
    for i in range(n_frames):
        # Simple straight-line trajectory along +z (forward), 1 unit apart --
        # enough to exercise real parsing/reshaping without needing a
        # realistic trajectory.
        pose_row = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, float(i)]
        lines.append(" ".join(str(v) for v in pose_row))
    poses_path.write_text("\n".join(lines) + "\n")


def _write_times(times_path: Path, n_frames: int) -> None:
    # 10 Hz capture, matches typical KITTI timing order of magnitude.
    times_path.write_text("\n".join(f"{i * 0.1:.6f}" for i in range(n_frames)) + "\n")


def _write_images(image_dir: Path, n_frames: int, width: int = 64, height: int = 48) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        img = Image.new("RGB", (width, height), color=(i % 255, 0, 0))
        img.save(image_dir / f"{i:06d}.png")


@pytest.fixture
def mock_kitti_root(tmp_path: Path) -> tuple[Path, str, int]:
    """Build a minimal valid mock KITTI odometry dataset under tmp_path.

    Returns (raw_path, sequence_id, n_frames).
    """
    sequence_id = "00"
    n_frames = 20

    raw_path = tmp_path / "kitti_mock"
    sequence_dir = raw_path / "sequences" / sequence_id
    sequence_dir.mkdir(parents=True)

    _write_calib(sequence_dir / "calib.txt")
    _write_times(sequence_dir / "times.txt", n_frames)
    _write_images(sequence_dir / "image_2", n_frames)

    poses_dir = raw_path / "poses"
    poses_dir.mkdir(parents=True)
    _write_poses(poses_dir / f"{sequence_id}.txt", n_frames)

    return raw_path, sequence_id, n_frames
