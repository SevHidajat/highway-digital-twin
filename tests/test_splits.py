from __future__ import annotations

import numpy as np
import pytest

from src.data_ingestion.schema import CameraIntrinsics, Frame, Sequence
from src.data_ingestion.splits import train_val_split


def _make_dummy_sequence(n_frames: int) -> Sequence:
    intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240, width=640, height=480)
    frames = [
        Frame(
            frame_id=f"00_{i:06d}",
            sequence_id="00",
            image_path=f"/fake/{i:06d}.png",
            pose=np.eye(4),
            intrinsics=intrinsics,
            timestamp=i * 0.1,
        )
        for i in range(n_frames)
    ]
    return Sequence(sequence_id="00", frames=frames)


def test_split_holds_out_every_nth_frame():
    sequence = _make_dummy_sequence(20)
    train, val = train_val_split(sequence, val_every_n_frames=8)

    val_indices = [int(f.frame_id.split("_")[1]) for f in val.frames]
    assert val_indices == [0, 8, 16]

    train_indices = [int(f.frame_id.split("_")[1]) for f in train.frames]
    assert set(train_indices).isdisjoint(set(val_indices))
    assert len(train_indices) + len(val_indices) == 20


def test_split_rejects_n_less_than_2():
    sequence = _make_dummy_sequence(10)
    with pytest.raises(ValueError, match="val_every_n_frames must be >= 2"):
        train_val_split(sequence, val_every_n_frames=1)


def test_split_naming_reflects_source_sequence():
    sequence = _make_dummy_sequence(10)
    train, val = train_val_split(sequence, val_every_n_frames=4)
    assert train.sequence_id == "00_train"
    assert val.sequence_id == "00_val"
