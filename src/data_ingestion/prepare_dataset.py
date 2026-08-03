"""
Dataset ingestion and preprocessing for the Highway Digital Twin project.

Responsibilities:
    - Parse raw driving-log data (currently: KITTI odometry; CADC support
      to follow the same pattern in a future cadc_loader.py) into the
      unified internal schema defined in schema.py.
    - Compute a train/val split along the route, preserving spatial
      continuity (see splits.py).
    - Cache processed data to data/processed/ as a single .npz per split,
      so training doesn't need to re-parse calib/pose text files or
      recompute intrinsics on every run.

Usage:
    python -m src.data_ingestion.prepare_dataset --config configs/kitti_baseline.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from src.data_ingestion.kitti_loader import load_sequence
from src.data_ingestion.schema import Sequence
from src.data_ingestion.splits import train_val_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def sequence_to_npz(sequence: Sequence) -> dict[str, np.ndarray]:
    """Flatten a Sequence into arrays suitable for np.savez.

    Image paths are kept as strings (loaded lazily at training time,
    not decoded here) to keep this cache small and fast to write/read.
    """
    return {
        "frame_ids": np.array([f.frame_id for f in sequence.frames]),
        "image_paths": np.array([f.image_path for f in sequence.frames]),
        "poses": np.stack([f.pose for f in sequence.frames]),  # (N, 4, 4)
        "timestamps": np.array([f.timestamp for f in sequence.frames]),
        "intrinsics": np.array(
            [
                [f.intrinsics.fx, f.intrinsics.fy, f.intrinsics.cx, f.intrinsics.cy]
                for f in sequence.frames
            ]
        ),
        "image_width": np.array(sequence.frames[0].intrinsics.width),
        "image_height": np.array(sequence.frames[0].intrinsics.height),
    }


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_kitti(config: dict) -> None:
    dataset_cfg = config["dataset"]
    raw_path = Path(dataset_cfg["raw_path"])
    processed_path = Path(dataset_cfg["processed_path"])
    sequence_id = dataset_cfg["sequence"]
    val_every_n_frames = dataset_cfg["val_every_n_frames"]

    if sequence_id == "TBD":
        raise ValueError(
            "configs/*.yaml has dataset.sequence set to the placeholder 'TBD' -- "
            "set it to an actual KITTI odometry sequence id (e.g. '00') before running."
        )

    frame_range = dataset_cfg.get("frame_range")
    frame_range = tuple(frame_range) if frame_range is not None else None

    logger.info(f"Loading KITTI sequence {sequence_id} from {raw_path}")
    if frame_range:
        logger.info(f"Restricting to frame_range={frame_range} (a manageable corridor segment)")
    sequence = load_sequence(raw_path=raw_path, sequence_id=sequence_id, frame_range=frame_range)
    logger.info(f"Loaded {len(sequence)} frames")

    train_seq, val_seq = train_val_split(sequence, val_every_n_frames=val_every_n_frames)
    logger.info(f"Split into {len(train_seq)} train / {len(val_seq)} val frames")

    processed_path.mkdir(parents=True, exist_ok=True)
    train_out = processed_path / f"{sequence_id}_train.npz"
    val_out = processed_path / f"{sequence_id}_val.npz"

    np.savez(train_out, **sequence_to_npz(train_seq))
    np.savez(val_out, **sequence_to_npz(val_seq))
    logger.info(f"Wrote {train_out}")
    logger.info(f"Wrote {val_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    dataset_name = config["dataset"]["name"]

    if dataset_name == "kitti":
        prepare_kitti(config)
    else:
        raise NotImplementedError(
            f"Dataset '{dataset_name}' not yet supported -- only 'kitti' is implemented so far. "
            f"Add a '{dataset_name}_loader.py' following the pattern in kitti_loader.py."
        )


if __name__ == "__main__":
    main()
