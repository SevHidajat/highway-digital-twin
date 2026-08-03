"""
End-to-end integration test for the full ingestion pipeline: parse a mock
KITTI sequence, split it, and cache to .npz -- exercising prepare_kitti()
exactly as the CLI entry point would.
"""

from __future__ import annotations

import numpy as np

from src.data_ingestion.prepare_dataset import prepare_kitti, sequence_to_npz
from src.data_ingestion.kitti_loader import load_sequence
from src.data_ingestion.splits import train_val_split


def test_prepare_kitti_end_to_end(mock_kitti_root, tmp_path):
    raw_path, sequence_id, n_frames = mock_kitti_root
    processed_path = tmp_path / "processed"

    config = {
        "dataset": {
            "name": "kitti",
            "sequence": sequence_id,
            "raw_path": str(raw_path),
            "processed_path": str(processed_path),
            "val_every_n_frames": 5,
        }
    }

    prepare_kitti(config)

    train_npz = processed_path / f"{sequence_id}_train.npz"
    val_npz = processed_path / f"{sequence_id}_val.npz"
    assert train_npz.exists()
    assert val_npz.exists()

    train_data = np.load(train_npz)
    val_data = np.load(val_npz)

    # val_every_n_frames=5 over 20 frames -> indices 0,5,10,15 held out (4 frames)
    assert len(val_data["frame_ids"]) == 4
    assert len(train_data["frame_ids"]) == n_frames - 4

    assert train_data["poses"].shape == (len(train_data["frame_ids"]), 4, 4)
    assert int(train_data["image_width"]) == 64
    assert int(train_data["image_height"]) == 48


def test_prepare_kitti_rejects_placeholder_sequence(mock_kitti_root, tmp_path):
    raw_path, _, _ = mock_kitti_root
    config = {
        "dataset": {
            "name": "kitti",
            "sequence": "TBD",
            "raw_path": str(raw_path),
            "processed_path": str(tmp_path / "processed"),
            "val_every_n_frames": 8,
        }
    }
    import pytest

    with pytest.raises(ValueError, match="placeholder 'TBD'"):
        prepare_kitti(config)


def test_sequence_to_npz_roundtrip_preserves_pose_values(mock_kitti_root):
    raw_path, sequence_id, n_frames = mock_kitti_root
    sequence = load_sequence(raw_path=raw_path, sequence_id=sequence_id)
    train_seq, _ = train_val_split(sequence, val_every_n_frames=5)

    arrays = sequence_to_npz(train_seq)
    expected_poses = np.stack([f.pose for f in train_seq.frames])
    assert np.allclose(arrays["poses"], expected_poses)
