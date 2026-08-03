from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data_ingestion.kitti_loader import (
    convert_pose_opencv_to_opengl,
    intrinsics_from_projection_matrix,
    list_images,
    load_sequence,
    parse_calib,
    parse_poses,
    parse_times,
)


def test_parse_calib_reads_all_projection_matrices(mock_kitti_root):
    raw_path, sequence_id, _ = mock_kitti_root
    calib = parse_calib(raw_path / "sequences" / sequence_id / "calib.txt")

    for key in ["P0", "P1", "P2", "P3"]:
        assert key in calib
        assert calib[key].shape == (3, 4)
    assert "Tr" in calib
    assert calib["Tr"].shape == (4, 4)


def test_parse_calib_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_calib(tmp_path / "does_not_exist.txt")


def test_parse_calib_missing_required_key_raises(tmp_path):
    bad_calib = tmp_path / "calib.txt"
    bad_calib.write_text("P0: 1 0 0 0 0 1 0 0 0 0 1 0\n")  # missing P1, P2, P3
    with pytest.raises(ValueError, match="missing required keys"):
        parse_calib(bad_calib)


def test_intrinsics_from_projection_matrix():
    P = np.array(
        [
            [718.856, 0.0, 607.19, 0.0],
            [0.0, 718.856, 185.22, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    intrinsics = intrinsics_from_projection_matrix(P, width=1241, height=376)
    assert intrinsics.fx == pytest.approx(718.856)
    assert intrinsics.fy == pytest.approx(718.856)
    assert intrinsics.cx == pytest.approx(607.19)
    assert intrinsics.cy == pytest.approx(185.22)


def test_parse_poses_returns_correct_count_and_shape(mock_kitti_root):
    raw_path, sequence_id, n_frames = mock_kitti_root
    poses = parse_poses(raw_path / "poses" / f"{sequence_id}.txt")
    assert len(poses) == n_frames
    for pose in poses:
        assert pose.shape == (4, 4)
        assert np.allclose(pose[3], [0, 0, 0, 1])


def test_parse_poses_missing_file_raises_with_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="only released for sequences 00-10"):
        parse_poses(tmp_path / "11.txt")


def test_parse_times(mock_kitti_root):
    raw_path, sequence_id, n_frames = mock_kitti_root
    times = parse_times(raw_path / "sequences" / sequence_id / "times.txt")
    assert len(times) == n_frames
    assert times == sorted(times)  # must be chronological


def test_list_images_sorted_numerically(tmp_path):
    image_dir = tmp_path / "image_2"
    image_dir.mkdir()
    # Deliberately create out of lexicographic order to catch a naive sort bug
    # (e.g. "10.png" < "2.png" lexicographically but not numerically).
    for i in [0, 2, 10, 1]:
        (image_dir / f"{i:06d}.png").touch()

    images = list_images(tmp_path, camera="image_2")
    stems = [int(p.stem) for p in images]
    assert stems == [0, 1, 2, 10]


def test_list_images_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_images(tmp_path, camera="image_2")


def test_load_sequence_end_to_end(mock_kitti_root):
    raw_path, sequence_id, n_frames = mock_kitti_root
    sequence = load_sequence(raw_path=raw_path, sequence_id=sequence_id)

    assert len(sequence) == n_frames
    assert sequence.sequence_id == sequence_id

    # Frames must be chronologically ordered (enforced by Sequence.__post_init__,
    # but assert explicitly here too since it's the key correctness property
    # a downstream reconstruction pipeline depends on).
    timestamps = [f.timestamp for f in sequence.frames]
    assert timestamps == sorted(timestamps)

    first_frame = sequence.frames[0]
    assert first_frame.intrinsics.width == 64
    assert first_frame.intrinsics.height == 48
    assert Path(first_frame.image_path).exists()


def test_load_sequence_frame_count_mismatch_raises(mock_kitti_root):
    raw_path, sequence_id, n_frames = mock_kitti_root
    # Corrupt the dataset: truncate times.txt so it no longer matches
    # the number of images -- this should be caught, not silently ignored.
    times_path = raw_path / "sequences" / sequence_id / "times.txt"
    lines = times_path.read_text().splitlines()
    times_path.write_text("\n".join(lines[:-1]) + "\n")

    with pytest.raises(ValueError, match="Frame count mismatch"):
        load_sequence(raw_path=raw_path, sequence_id=sequence_id)


def test_convert_pose_opencv_to_opengl_flips_y_and_z():
    identity = np.eye(4)
    converted = convert_pose_opencv_to_opengl(identity)
    expected = np.diag([1.0, -1.0, -1.0, 1.0])
    assert np.allclose(converted, expected)


def test_convert_pose_rejects_wrong_shape():
    with pytest.raises(ValueError, match="must be 4x4"):
        convert_pose_opencv_to_opengl(np.eye(3))
