"""
Parser for the KITTI Odometry benchmark format.

Expected directory layout (standard KITTI odometry download):

    <raw_path>/
        sequences/
            00/
                calib.txt
                times.txt
                image_2/   000000.png  000001.png  ...
                image_3/   000000.png  000001.png  ...
        poses/
            00.txt   # ground-truth poses, only available for sequences 00-10

References for format details:
    - calib.txt: P0-P3 (3x4 projection matrices, rectified), Tr (velodyne->cam0)
    - poses/XX.txt: one line per frame, 12 numbers = row-major 3x4 [R|t],
      giving T_world_cam0 where world origin = cam0 pose at frame 0.

IMPORTANT gotcha (do not skip): KITTI's camera convention is x-right,
y-down, z-forward (standard computer-vision/OpenCV convention). Most
NeRF/3DGS reference codebases (built on top of COLMAP conventions)
expect x-right, y-up, z-backward (OpenGL convention). Poses returned by
this loader are in the *original KITTI/OpenCV convention* -- use
`convert_pose_opencv_to_opengl` before feeding into a NeRF/3DGS training
pipeline built against COLMAP-convention code, and document which
convention every downstream module expects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data_ingestion.schema import CameraIntrinsics, Frame, Sequence


def parse_calib(calib_path: Path) -> dict[str, np.ndarray]:
    """Parse a KITTI calib.txt file into a dict of {name: matrix}.

    P0-P3 are returned as 3x4 arrays. Tr (velodyne-to-cam0) is returned
    as a 4x4 homogeneous matrix if present.
    """
    if not calib_path.exists():
        raise FileNotFoundError(f"calib.txt not found at {calib_path}")

    calib: dict[str, np.ndarray] = {}
    with open(calib_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, _, values = line.partition(":")
            key = key.strip()
            nums = np.array([float(x) for x in values.split()], dtype=np.float64)
            if key.startswith("P"):
                if nums.size != 12:
                    raise ValueError(f"Expected 12 values for {key}, got {nums.size}")
                calib[key] = nums.reshape(3, 4)
            elif key == "Tr":
                if nums.size != 12:
                    raise ValueError(f"Expected 12 values for Tr, got {nums.size}")
                tr = np.eye(4, dtype=np.float64)
                tr[:3, :4] = nums.reshape(3, 4)
                calib[key] = tr

    required = ["P0", "P1", "P2", "P3"]
    missing = [k for k in required if k not in calib]
    if missing:
        raise ValueError(f"calib.txt at {calib_path} missing required keys: {missing}")

    return calib


def intrinsics_from_projection_matrix(
    P: np.ndarray, width: int, height: int
) -> CameraIntrinsics:
    """Extract pinhole intrinsics from a KITTI rectified projection matrix.

    For rectified KITTI cameras, P = [K | K @ t], so fx, fy, cx, cy can be
    read directly off the top-left 3x3 block.
    """
    if P.shape != (3, 4):
        raise ValueError(f"Expected a 3x4 projection matrix, got shape {P.shape}")
    fx, fy = P[0, 0], P[1, 1]
    cx, cy = P[0, 2], P[1, 2]
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


def parse_poses(poses_path: Path) -> list[np.ndarray]:
    """Parse a KITTI poses/XX.txt file into a list of 4x4 T_world_cam0
    matrices, one per frame, in chronological order.
    """
    if not poses_path.exists():
        raise FileNotFoundError(
            f"poses file not found at {poses_path} -- note ground-truth poses "
            f"are only released for sequences 00-10"
        )

    poses = []
    with open(poses_path, "r") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            nums = np.array([float(x) for x in line.split()], dtype=np.float64)
            if nums.size != 12:
                raise ValueError(
                    f"{poses_path}:{line_num}: expected 12 values, got {nums.size}"
                )
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :4] = nums.reshape(3, 4)
            poses.append(pose)

    return poses


def parse_times(times_path: Path) -> list[float]:
    """Parse a KITTI times.txt file (one timestamp in seconds per line)."""
    if not times_path.exists():
        raise FileNotFoundError(f"times.txt not found at {times_path}")
    with open(times_path, "r") as f:
        return [float(line.strip()) for line in f if line.strip()]


def list_images(sequence_dir: Path, camera: str = "image_2") -> list[Path]:
    """Return the sorted list of image paths for a given camera folder.

    Sorting is done numerically on the filename stem (not lexicographically)
    to guard against ordering bugs if filenames are ever not zero-padded
    consistently.
    """
    image_dir = sequence_dir / camera
    if not image_dir.exists():
        raise FileNotFoundError(f"Camera image directory not found: {image_dir}")

    images = sorted(image_dir.glob("*.png"), key=lambda p: int(p.stem))
    if not images:
        raise ValueError(f"No .png images found in {image_dir}")
    return images


def convert_pose_opencv_to_opengl(pose: np.ndarray) -> np.ndarray:
    """Convert a camera-to-world pose from OpenCV convention (x-right,
    y-down, z-forward) to OpenGL/COLMAP convention (x-right, y-up,
    z-backward), by flipping the y and z axes of the camera's local frame.

    Apply this when feeding KITTI poses into NeRF/3DGS codebases that
    assume COLMAP-style camera conventions.
    """
    if pose.shape != (4, 4):
        raise ValueError(f"pose must be 4x4, got {pose.shape}")
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return pose @ flip


def load_sequence(
    raw_path: Path,
    sequence_id: str,
    camera: str = "image_2",
    image_width: int | None = None,
    image_height: int | None = None,
    frame_range: tuple[int, int] | None = None,
) -> Sequence:
    """Load one KITTI odometry sequence into the unified `Sequence` schema.

    Args:
        raw_path: root of the KITTI odometry download (contains
            'sequences/' and 'poses/' subdirectories).
        sequence_id: e.g. "00".
        camera: "image_2" (left, color) or "image_3" (right, color).
        image_width / image_height: KITTI image resolution varies slightly
            per sequence (typically ~1241x376) -- pass explicitly if known,
            otherwise this function will read the first image to infer it
            (requires Pillow or OpenCV to be installed).
        frame_range: optional (start, end) frame indices (end-exclusive) to
            slice out of the full sequence. IMPORTANT: a full KITTI odometry
            sequence can be thousands of frames covering several kilometers
            of driving -- far more scene content than a single NeRF/3DGS
            reconstruction can represent well. Almost all driving-scene
            neural rendering papers reconstruct short segments (tens to a
            couple hundred frames, ~100-300m of road), not an entire
            sequence. Leave this unset only if you deliberately want the
            full sequence (e.g. for a multi-segment/large-scale experiment
            you've scoped for separately).

    Returns:
        A Sequence with one Frame per image, poses in original KITTI/OpenCV
        convention (see module docstring re: convention conversion).
    """
    raw_path = Path(raw_path)
    sequence_dir = raw_path / "sequences" / sequence_id
    poses_path = raw_path / "poses" / f"{sequence_id}.txt"

    calib = parse_calib(sequence_dir / "calib.txt")
    times = parse_times(sequence_dir / "times.txt")
    image_paths = list_images(sequence_dir, camera=camera)

    if len(image_paths) != len(times):
        raise ValueError(
            f"Frame count mismatch in sequence {sequence_id}: "
            f"{len(image_paths)} images vs. {len(times)} timestamps"
        )

    # Ground-truth poses are only available for sequences 00-10.
    try:
        poses = parse_poses(poses_path)
        if len(poses) != len(image_paths):
            raise ValueError(
                f"Frame count mismatch in sequence {sequence_id}: "
                f"{len(image_paths)} images vs. {len(poses)} poses"
            )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No ground-truth poses for sequence {sequence_id}. "
            f"Sequences 11+ require running your own SfM/odometry to get poses "
            f"before they can be used for reconstruction -- see docs/PROJECT_SPEC.md."
        )

    if frame_range is not None:
        start, end = frame_range
        if not (0 <= start < end <= len(image_paths)):
            raise ValueError(
                f"frame_range {frame_range} is invalid for a sequence with "
                f"{len(image_paths)} frames -- must satisfy 0 <= start < end <= n_frames"
            )
        image_paths = image_paths[start:end]
        poses = poses[start:end]
        times = times[start:end]

    if image_width is None or image_height is None:
        image_width, image_height = _infer_image_size(image_paths[0])

    P_key = "P2" if camera == "image_2" else "P3"
    intrinsics = intrinsics_from_projection_matrix(
        calib[P_key], width=image_width, height=image_height
    )

    frames = [
        Frame(
            frame_id=f"{sequence_id}_{img_path.stem}",
            sequence_id=sequence_id,
            image_path=str(img_path),
            pose=pose,
            intrinsics=intrinsics,
            timestamp=timestamp,
        )
        for img_path, pose, timestamp in zip(image_paths, poses, times)
    ]

    return Sequence(sequence_id=sequence_id, frames=frames)


def _infer_image_size(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            return img.width, img.height
    except ImportError:
        raise ImportError(
            "Pillow is required to auto-infer image size (or pass "
            "image_width/image_height explicitly). Install with: pip install Pillow"
        )
