# Highway Digital Twin Benchmark

**A neural-rendered digital twin of a long-haul highway corridor, with multi-sensor simulation and downstream metric benchmarking.**

Reference point: Waabi's SaLF (Sparse Local Fields, ICRA 2026) — a unified representation supporting both rasterization (3DGS-style speed) and ray-tracing (NeRF-style physical accuracy) for multi-sensor (camera + LiDAR) simulation. This project builds a smaller-scale, personally implemented analog: reconstruct a highway corridor as a digital twin, render synthetic camera/LiDAR data from novel viewpoints/conditions, and benchmark the fidelity + downstream impact against real data.

---

## 1. Problem Statement

Validating autonomous trucking systems purely on real-world miles is slow and expensive, especially for rare/adverse-weather scenarios. Neural rendering enables "digital twins" of real routes — reconstructed once from real sensor data, then replayable under novel conditions (weather, time of day, actor placement) for scalable testing.

**Research question this project answers:**
> For a given highway corridor, how do NeRF-family and 3DGS-family digital twin reconstructions compare in (a) sensor-level rendering fidelity, (b) rendering speed/scalability, and (c) their effect on downstream perception/planning metrics (e.g., detection accuracy, estimated delivery-relevant metrics) when used as a substitute for real sensor logs?

---

## 2. Scope (v1 — buildable solo in ~3-4 months)

- **Route:** one publicly available driving dataset segment representing a highway corridor (see Data section) — treat it as a stand-in for a long-haul route.
- **Reconstruction:** implement both a NeRF-family and a 3DGS-family reconstruction pipeline for the same corridor segment, so they are directly comparable.
- **Sensors:** camera (primary) + LiDAR (stretch goal, depending on dataset availability) — matches your original sensor fusion research background.
- **Metrics tracked:**
  - **Rendering fidelity:** PSNR, SSIM, LPIPS vs. held-out real frames.
  - **Speed:** training time, frames-per-second at inference.
  - **Downstream task impact:** run an off-the-shelf object detector on real vs. synthetic frames of the same viewpoint; compare detection consistency (a common self-driving sim validation technique — this is your "safety-relevant" metric).
  - **Novel-view extrapolation quality:** render from a laterally shifted virtual camera position (simulating a different lane) — a known failure mode for both NeRF and 3DGS, valuable to characterize.
- **"Delivery time / safety" framing:** since you don't have real logistics data, simulate this analytically — e.g., use rendering realism degradation at novel viewpoints as a proxy for "how far can we trust this digital twin for planning," and speed metrics as a proxy for "how fast can we validate a new route." Be explicit in your writeup that this is a proxy, not real fleet data — reviewers and interviewers will respect the honesty more than an overclaim.

---

## 3. System Architecture

```
Raw sensor logs (camera[, LiDAR])
        │
        ▼
[data_ingestion]  — parse dataset, extract poses/calibration, build train/val split
        │
        ▼
[reconstruction]  — two parallel pipelines:
        ├── NeRF-family model (PyTorch)
        └── 3DGS-family model (PyTorch + custom CUDA/C++ rasterizer)
        │
        ▼
[rendering]  — novel-view synthesis: held-out poses, shifted-lane poses, relit/weather-augmented poses (stretch)
        │
        ▼
[metrics]  — fidelity (PSNR/SSIM/LPIPS), speed benchmarks, downstream detector consistency
        │
        ▼
[sim_agents] (stretch) — simple traffic-agent placement to test "digital twin as sim environment" angle
```

**Language split (intentional, matches the job qualifications):**
- Python/PyTorch: data pipeline, model training, evaluation harness, plotting.
- C++ (with CUDA): the core rasterization kernel for the 3DGS path — this is the single highest-leverage piece of "production systems language" evidence you can show, since it's exactly what shipping a real-time renderer requires.

---

## 4. Suggested Datasets (public, no data collection needed)

- **CADC (Canadian Adverse Driving Conditions dataset)** — Waterloo-built, snow-focused. Strong "I understand Canadian winter driving" signal, and thematically ties back to your original adverse-weather research.
- **KITTI / KITTI-360** — standard baseline, well-supported by existing NeRF/3DGS driving-scene codebases, good for validating your pipeline works before trying CADC's harder conditions.
- **nuScenes** — good multi-sensor (camera+LiDAR+radar) coverage if you want closer parity to the job posting's "camera, LiDAR, and RADAR" language.

**Recommended path:** start with KITTI (easiest, most tooling support) to get the pipeline working end-to-end, then port to a CADC segment as your "hero result" showcasing adverse-weather digital twins specifically.

---

## 5. Implementation Plan (Milestones)

| Milestone | Target | Output |
|---|---|---|
| M1 | Week 2 | Data pipeline: load KITTI segment, extract poses/calibration, train/val split script |
| M2 | Week 6 | Baseline NeRF reconstruction working, fidelity metrics computed |
| M3 | Week 10 | 3DGS reconstruction working (can use a from-scratch simplified rasterizer or heavily adapted open-source core, but you must understand and modify the CUDA kernel yourself — don't just call a black-box library) |
| M4 | Week 12 | Side-by-side comparison report: fidelity, speed, novel-view degradation |
| M5 | Week 14 | Downstream detector-consistency experiment |
| M6 | Week 16 | Port best pipeline to a CADC (adverse-weather) segment; final report + repo polish |
| M7 | Week 18-20 | Paper writeup for MDPI/IEEE submission (see Section 7) |

---

## 6. GitHub Repo Structure

```
highway-digital-twin/
├── README.md                  # project overview, results summary, GIFs/images of renders
├── LICENSE
├── requirements.txt
├── configs/                   # YAML configs per experiment (dataset, model, hyperparams)
├── src/
│   ├── data_ingestion/        # dataset parsers, calibration handling, pose extraction
│   ├── reconstruction/
│   │   ├── nerf/               # NeRF-family model implementation
│   │   └── gaussian_splat/     # 3DGS-family model implementation (Python side)
│   ├── rendering/              # novel-view synthesis, camera path generation
│   ├── metrics/                # PSNR/SSIM/LPIPS, speed benchmarking, detector-consistency eval
│   └── sim_agents/             # (stretch) simple actor placement for sim scenarios
├── cpp/
│   └── rasterizer/             # custom CUDA/C++ rasterization kernel + PyTorch bindings
├── notebooks/                  # exploratory analysis, result visualization (kept OUT of main pipeline)
├── tests/                      # unit tests — this matters a lot for the "production quality" signal
├── docs/
│   ├── PROJECT_SPEC.md         # this document
│   ├── ARCHITECTURE.md         # deeper technical design notes
│   └── RESULTS.md              # tables/plots of final benchmark results
└── paper/                      # LaTeX source for the MDPI/IEEE submission
```

**Production-quality signals to bake in from day one (this is what Waabi's job posting explicitly screens for):**
- Type hints throughout Python code, docstrings, `pytest` unit tests for data pipeline and metrics.
- CI via GitHub Actions (lint + test on push) — trivial to set up, disproportionately impressive.
- Config-driven experiments (no hardcoded paths/hyperparams in scripts).
- A `RESULTS.md` with actual numbers/tables, not just qualitative claims — reviewers and interviewers trust measured tradeoffs over prose.

---

## 7. Paper Angle (MDPI / IEEE)

Realistic target framing given solo, ~4-month scope: an **applied benchmark/comparison paper**, not a novel-methods paper. This is a legitimate and common paper type, and better fits your timeline than trying to out-innovate SaLF itself.

**Candidate venues:**
- *MDPI Sensors* or *MDPI Remote Sensing* — both publish applied sensor-simulation/benchmark papers, faster review cycles, good fit for a comparative empirical study.
- *IEEE Transactions on Intelligent Transportation Systems (T-ITS)* — more prestigious, longer review, strong fit topically (digital twins + highway driving).
- *IEEE Access* — good fallback for faster publication with solid but not groundbreaking novelty.
- *IEEE IV Symposium* or *ITSC* conference papers — worth checking submission deadlines, since conference papers are faster to land than journal papers and still carry real signal for job applications.

**Suggested paper framing:** *"Benchmarking Neural Rendering Approaches for Highway Digital Twins: Fidelity, Speed, and Downstream Perception Impact under Adverse Weather."* This framing does three things at once: ties directly to your job target's language, is honest about scope (a benchmark, not a new method), and gives you a legitimate empirical contribution (the CADC adverse-weather angle is genuinely under-explored in NeRF/3DGS literature).

---

## 8. What NOT to Overbuild

Given solo + 4-month scope, explicitly descope:
- Full RADAR simulation (mention as future work only).
- Real logistics/delivery-time data (use the proxy framing described in Section 2).
- A from-scratch CUDA rasterizer with zero reference — heavily study and adapt from an open, permissively-licensed reference implementation, but modify it meaningfully (e.g., add a feature, change memory layout, add a new sensor model) so you can honestly say you understand and extended it, not just ran it.

---

## 9. Phase 2 (after NeRF vs. 3DGS comparison is complete): Multi-Sensor + GPS Integration

This phase directly extends the sensor fusion methodology from Severin's M.A.Sc. thesis
("Sensor Fusion of LiDAR and Camera for Lane Detection with Instance Segmentation and
GPS Integration using Kalman Filtering for Target Tracking on Snowy Roads," McMaster
University, 2024) into the neural-rendering digital twin built in Phases 1. The thesis
used CADC (camera + LiDAR + GPS, snowy Waterloo roads) with a low-level fusion
(LiDAR-to-camera point projection) feeding a YOLOv8 instance segmentation model, a
mid-level fusion step (segmented road boundaries + GPS/OpenStreetMap lane geometry),
and Kalman filtering to track the ego-vehicle's position ("red dot") smoothly across
frames. Phase 2 re-implements the *spirit* of that pipeline against synthetic sensor
output from the digital twin, rather than live sensor logs.

**Motivation / narrative:** Phase 1 answers "how faithfully can we regenerate real
camera views." Phase 2 answers "can this regenerated environment support the same
kind of downstream perception + tracking pipeline a real self-driving stack would
run" -- the actual point of building a digital twin at all, and a direct thread back
to Severin's original adverse-weather sensor fusion research.

### 9.1 LiDAR rendering from the same digital twin

- Extend the trained NeRF/3DGS reconstruction (Phase 1) to render synthetic LiDAR
  point clouds, not just camera images, from arbitrary viewpoints -- mirroring SaLF's
  core claim of one representation supporting both sensor modalities.
- For 3DGS: LiDAR rendering means ray-casting against the Gaussians (depth queries)
  rather than rasterizing -- a genuinely different code path from the fast camera
  rasterizer, worth explicitly comparing (speed, point cloud density/noise) against
  the NeRF path's ray marching, which naturally supports arbitrary ray queries.
- Evaluation: compare synthetic LiDAR point clouds against real LiDAR frames from the
  same held-out pose (point-to-point distance, or a simpler density/coverage check).

### 9.2 Re-running the thesis's fusion + segmentation pipeline on synthetic sensor output

- Take the (now available) synthetic camera + LiDAR pair from the digital twin,
  project LiDAR points onto the camera image exactly as in thesis Section 4.3/6.2
  (low-level fusion), and run an off-the-shelf pretrained segmentation model (YOLOv8
  segmentation, or a simpler road/lane segmentation model) on both real and synthetic
  fused images at matched poses.
- Compare segmentation output on real vs. synthetic fused images -- this directly
  extends the "downstream detector consistency" metric from Section 2, but now with
  the added fusion step your thesis introduced, rather than camera-only detection.

### 9.3 GPS-referenced agent tracking within the digital twin ("red dot" replay)

- Reuse the Kalman filter tracking approach from thesis Section 6.1(4): track a
  simulated ego-vehicle (or another agent) as a "red dot" moving through the
  reconstructed digital twin, using the digital twin's known camera trajectory as
  the ground-truth GPS-equivalent track.
- This produces a rendered visualization directly comparable to Waabi's public
  simulation demos (a highlighted tracked vehicle over a reconstructed road scene
  with surrounding camera views) -- valuable both as a portfolio visual and as a
  genuine technical checkpoint tying tracking + rendering together.
- Populates the `src/sim_agents/` module (currently a stub -- see its README).

### 9.4 Suggested milestones

| Milestone | Target (after Phase 1 M6) | Output |
|---|---|---|
| P2-M1 | +3 weeks | LiDAR rendering working for at least one method (NeRF or 3DGS) |
| P2-M2 | +5 weeks | Real vs. synthetic LiDAR fidelity comparison |
| P2-M3 | +7 weeks | Fusion + segmentation pipeline re-run on synthetic sensor pairs, compared to real |
| P2-M4 | +9 weeks | Kalman-filtered agent tracking rendered over the digital twin ("red dot" replay), portfolio-ready visualization |

### 9.5 Paper framing update

With Phase 2 included, the paper (Section 7) can honestly claim a broader empirical
contribution: not just rendering fidelity, but *fidelity of a full sensor-fusion +
tracking pipeline* run on synthetic vs. real sensor data -- a more novel and thesis-
connected angle than Phase 1 alone. Consider retitling to something like: "From
Real to Synthetic: Benchmarking Neural Digital Twins for Multi-Sensor Perception and
Tracking on Adverse-Weather Highway Corridors."

