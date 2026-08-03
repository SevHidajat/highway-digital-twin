"""
Unified internal schema for driving-log data, independent of source dataset
format (KITTI, KITTI-360, CADC, ...). Every dataset-specific loader in this
package must produce a list of `Frame` objects conforming to this schema, so
everything downstream (reconstruction, rendering, metrics) is dataset-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics. Extend with distortion params if a
    dataset's camera model requires it (e.g. fisheye for CADC)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def as_matrix(self) -> np.ndarray:
        """Return the 3x3 intrinsic matrix K."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def __post_init__(self):
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(f"Focal lengths must be positive, got fx={self.fx}, fy={self.fy}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Image dimensions must be positive, got width={self.width}, height={self.height}"
            )


@dataclass(frozen=True)
class Frame:
    """A single synchronized observation at one point along the route.

    `pose` is the camera-to-world transform (4x4, homogeneous), in a
    right-handed world frame consistent across all frames in a sequence
    -- this consistency is what a NeRF/3DGS reconstruction depends on,
    so loaders must be careful to apply any dataset-specific axis
    conventions (KITTI, for instance, does NOT use the same convention
    as COLMAP/NeRF codebases) before returning Frame objects.
    """

    frame_id: str
    sequence_id: str
    image_path: str
    pose: np.ndarray  # 4x4 camera-to-world
    intrinsics: CameraIntrinsics
    timestamp: float
    lidar_path: str | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.pose.shape != (4, 4):
            raise ValueError(f"pose must be a 4x4 matrix, got shape {self.pose.shape}")
        # Sanity-check the bottom row of a valid homogeneous transform.
        if not np.allclose(self.pose[3], [0, 0, 0, 1], atol=1e-5):
            raise ValueError(
                f"pose bottom row must be [0,0,0,1], got {self.pose[3]} "
                f"(frame_id={self.frame_id}) -- check axis convention conversion"
            )


@dataclass
class Sequence:
    """A parsed, ordered set of frames representing one continuous drive
    (e.g. one KITTI odometry sequence, or one CADC run)."""

    sequence_id: str
    frames: list[Frame]

    def __len__(self) -> int:
        return len(self.frames)

    def __post_init__(self):
        if len(self.frames) == 0:
            raise ValueError(f"Sequence {self.sequence_id} has no frames")
        # Timestamps should be monotonically increasing -- catches loader bugs
        # (e.g. reading files in filesystem order instead of numeric order).
        timestamps = [f.timestamp for f in self.frames]
        if timestamps != sorted(timestamps):
            raise ValueError(
                f"Sequence {self.sequence_id} frames are not in chronological "
                f"order -- check that the loader sorts by numeric frame index."
            )
