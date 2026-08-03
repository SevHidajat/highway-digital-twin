"""
Train/val splitting for a Sequence.

The split must preserve spatial continuity along the route: holding out
every Nth frame gives genuine novel-view-synthesis targets (the model
never trained on that exact pose), as opposed to a random shuffle split,
which could leave near-duplicate neighboring frames on both sides and
make the evaluation artificially easy.
"""

from __future__ import annotations

from src.data_ingestion.schema import Sequence


def train_val_split(sequence: Sequence, val_every_n_frames: int) -> tuple[Sequence, Sequence]:
    """Split a Sequence into train/val by holding out every Nth frame.

    Args:
        sequence: the full parsed Sequence.
        val_every_n_frames: e.g. 8 means frames at index 0, 8, 16, ... go
            to validation; the rest go to train. Must be >= 2 so at least
            some frames remain for training.

    Returns:
        (train_sequence, val_sequence)
    """
    if val_every_n_frames < 2:
        raise ValueError(
            f"val_every_n_frames must be >= 2 (else no training data remains), "
            f"got {val_every_n_frames}"
        )

    train_frames = []
    val_frames = []
    for i, frame in enumerate(sequence.frames):
        if i % val_every_n_frames == 0:
            val_frames.append(frame)
        else:
            train_frames.append(frame)

    if not train_frames:
        raise ValueError(
            f"Split produced zero training frames for sequence "
            f"{sequence.sequence_id} -- check val_every_n_frames "
            f"({val_every_n_frames}) against sequence length ({len(sequence)})"
        )
    if not val_frames:
        raise ValueError(
            f"Split produced zero validation frames for sequence "
            f"{sequence.sequence_id} -- this shouldn't happen since index 0 "
            f"always goes to val; check for an empty input sequence"
        )

    return (
        Sequence(sequence_id=f"{sequence.sequence_id}_train", frames=train_frames),
        Sequence(sequence_id=f"{sequence.sequence_id}_val", frames=val_frames),
    )
