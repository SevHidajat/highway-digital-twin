"""
3D Gaussian Splatting reconstruction pipeline for a highway corridor segment.

Intended design (see docs/PROJECT_SPEC.md Section 3):
    - Initialize Gaussians from a sparse point cloud (SfM or LiDAR-derived).
    - Optimize position, scale, rotation, opacity, and spherical-harmonic
      color coefficients per Gaussian against held-in frames.
    - The rasterization forward/backward pass is implemented in
      cpp/rasterizer/ (CUDA) and bound into this module via a PyTorch
      custom autograd Function -- this is the "native systems language"
      component of the project and should be understood and modified
      (not just imported as a black box).
    - Adaptive densification/pruning of Gaussians during training to
      handle the scale of a highway segment (long, thin, mostly
      background) efficiently.

Usage:
    python -m src.reconstruction.gaussian_splat.train --config configs/kitti_baseline.yaml
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    raise NotImplementedError("Implement 3DGS training loop, calling into cpp/rasterizer bindings")


if __name__ == "__main__":
    main()
