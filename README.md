# Highway Digital Twin Benchmark

Neural-rendered digital twin of a long-haul highway corridor, benchmarking NeRF-family vs. 3D Gaussian Splatting reconstruction for multi-sensor (camera[, LiDAR]) simulation — with a focus on adverse-weather robustness and downstream perception impact.

> Inspired by and benchmarked in the spirit of Waabi's [SaLF](https://waabi.ai/salf/) (ICRA 2026), which unifies NeRF and 3DGS into one representation supporting both real-time rasterization and physically accurate ray-tracing for multi-sensor self-driving simulation. This project is an independent, smaller-scale exploration of the same problem space — see [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md) for full design rationale.

## Status

🚧 Early development — see [Milestones](docs/PROJECT_SPEC.md#5-implementation-plan-milestones)

## Why this project

Autonomous trucking validation increasingly relies on simulated sensor data reconstructed from real driving logs ("digital twins"), rather than exclusively on real-world miles. This project builds and benchmarks that pipeline end-to-end: reconstruct a real highway segment with two competing neural rendering paradigms, render novel viewpoints and conditions, and measure both rendering fidelity and downstream perception impact.

## Quick Start

```bash
git clone <this-repo>
cd highway-digital-twin
pip install -r requirements.txt
python -m src.data_ingestion.prepare_dataset --config configs/kitti_baseline.yaml
python -m src.reconstruction.nerf.train --config configs/kitti_baseline.yaml
python -m src.reconstruction.gaussian_splat.train --config configs/kitti_baseline.yaml
python -m src.metrics.run_benchmark --config configs/kitti_baseline.yaml
```

## Results

See [docs/RESULTS.md](docs/RESULTS.md) (populated as experiments complete).

## Repo Structure

See the full breakdown in [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md#6-github-repo-structure).

## License

MIT — see [LICENSE](LICENSE)
