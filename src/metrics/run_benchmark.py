"""
Benchmark harness: computes and reports all comparison metrics between
the NeRF and 3DGS reconstructions of the same corridor segment.

Metrics computed:
    - Fidelity: PSNR, SSIM, LPIPS on held-out real frames.
    - Speed: wall-clock training time, inference FPS (camera[, LiDAR]).
    - Novel-view degradation: fidelity metrics as a function of lateral
      camera offset from the real recorded trajectory.
    - Downstream detector consistency: run a fixed pretrained object
      detector (e.g., YOLO or similar) on paired real/synthetic frames
      at the same held-out pose; report detection agreement (IoU-matched
      precision/recall) as a proxy for "would this digital twin fool or
      mislead a perception stack."

Outputs a results table (and optionally plots) written to
docs/RESULTS.md and/or a results/ directory for the paper writeup.

Usage:
    python -m src.metrics.run_benchmark --config configs/kitti_baseline.yaml
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    raise NotImplementedError("Implement metric computation per docs/PROJECT_SPEC.md Section 2")


if __name__ == "__main__":
    main()
