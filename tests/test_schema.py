from __future__ import annotations

import numpy as np
import pytest

from src.data_ingestion.schema import CameraIntrinsics, Frame, Sequence


def _valid_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(fx=500, fy=500, cx=320, cy=240, width=640, height=480)


def test_camera_intrinsics_as_matrix():
    intrinsics = _valid_intrinsics()
    K = intrinsics.as_matrix()
    assert K.shape == (3, 3)
    assert K[0, 0] == 500
    assert K[2, 2] == 1.0


def test_camera_intrinsics_rejects_nonpositive_focal_length():
    with pytest.raises(ValueError, match="Focal lengths must be positive"):
        CameraIntrinsics(fx=0, fy=500, cx=320, cy=240, width=640, height=480)


def test_camera_intrinsics_rejects_nonpositive_dimensions():
    with pytest.raises(ValueError, match="Image dimensions must be positive"):
        CameraIntrinsics(fx=500, fy=500, cx=320, cy=240, width=-1, height=480)


def test_frame_rejects_non_4x4_pose():
    with pytest.raises(ValueError, match="pose must be a 4x4 matrix"):
        Frame(
            frame_id="00_000000",
            sequence_id="00",
            image_path="/fake/000000.png",
            pose=np.eye(3),
            intrinsics=_valid_intrinsics(),
            timestamp=0.0,
        )


def test_frame_rejects_invalid_homogeneous_bottom_row():
    bad_pose = np.eye(4)
    bad_pose[3] = [1, 2, 3, 4]  # not [0,0,0,1]
    with pytest.raises(ValueError, match=r"bottom row must be \[0,0,0,1\]"):
        Frame(
            frame_id="00_000000",
            sequence_id="00",
            image_path="/fake/000000.png",
            pose=bad_pose,
            intrinsics=_valid_intrinsics(),
            timestamp=0.0,
        )


def test_sequence_rejects_empty_frame_list():
    with pytest.raises(ValueError, match="has no frames"):
        Sequence(sequence_id="00", frames=[])


def test_sequence_rejects_out_of_order_timestamps():
    intrinsics = _valid_intrinsics()
    frames = [
        Frame(
            frame_id=f"00_{i}",
            sequence_id="00",
            image_path=f"/fake/{i}.png",
            pose=np.eye(4),
            intrinsics=intrinsics,
            timestamp=t,
        )
        for i, t in enumerate([0.0, 0.2, 0.1])  # out of order
    ]
    with pytest.raises(ValueError, match="not in chronological order"):
        Sequence(sequence_id="00", frames=frames)
