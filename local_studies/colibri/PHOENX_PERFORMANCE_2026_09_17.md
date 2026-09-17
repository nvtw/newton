# Colibri temporal contact performance

## Workload and method

RTX PRO 6000 Blackwell, Warp 1.17.0, Nsight Systems 2026.2.1.
Use the unchanged Colibri defaults: 120 Hz collision detection, 24 temporal
substeps per refresh, one biased iteration and final-substep relaxation,
latest normal contact matching, density scale 1.0, and a passive frame axle.
Measurements replay the captured physics graph through `ViewerNull`; rendering,
initial compilation, graph capture, and quality checks are outside step timing.

The initial Nsight trace identified the temporal dynamic-contact sweep as
60.8% of GPU kernel time. At frame 61, 20 of 40 contact groups (944 of 2,516
normal rows) had both friction coefficients zero.

## Retained optimization

Do not partition, correlate anchors, or solve friction for groups whose static
and dynamic coefficients are both zero. Keep every normal row, its ordering,
effective mass, common impulse point, and paired impulse application intact.
Represent such groups with an empty friction-patch list. Still write their
material key and generation markers, so changing the material back to frictional
rebuilds anchors instead of reusing stale history. Empty-patch solves return
before copying normal loads or constructing poses.

## Short measurements

Same authored initial state, 60 warmup frames, then 180 measured frames:

| Case | Mean physics time [ms/frame] |
| --- | ---: |
| Baseline, unprofiled | 22.623 |
| Scalar normal-row loading (rejected) | 24.002 |
| Uniform subgroup arithmetic (not retained) | 22.375 |
| Uniform arithmetic with unrolling (not retained) | 22.404 |
| Skip only zero-friction solving (intermediate) | 21.469 |
| Omit zero-friction preparation and solving | 19.037 |

The retained version reduces this short-run frame time by 15.9% (18.8% higher
throughput). Its body poses and velocities at frame 240 match the baseline
bit for bit. Uniform subgroup changes were not retained because their small
apparent improvement did not justify extra code.

Nsight GPU kernel sums: 21.713 -> 18.611 ms/frame. The dominant biased dynamic
sweep averages 274.94 -> 219.27 microseconds/call. There are still exactly 8,640
calls across 180 frames (48 per frame); normal-row and collision work counts
are not reduced. Profiled wall times are 22.311 -> 19.267 ms/frame; use the
unprofiled measurements for speed claims.

## Full-minute verification

Both runs advanced 3,600 frames from the same initial scene. Timing excludes
60 warmup frames; validation includes all 3,600 frames.

| Metric | Baseline | Optimized |
| --- | ---: | ---: |
| Mean physics [ms/frame] | 22.581 | 18.846 |
| Median physics [ms/frame] | 22.565 | 18.771 |
| p95 physics [ms/frame] | 24.277 | 20.469 |
| Peak fresh penetration [mm] | 0.470375 | 0.470375 |

This is a 16.54% time reduction and 19.82% throughput increase. Every saved
body pose and velocity is **bitwise identical**: 3,600 x 36 x 7 pose values
and 3,600 x 36 x 6 velocity values. Assembly checks and crank tracking passed
at every frame. The pre-existing support-creep failure is also identical:
0.170 mm at 6.75 seconds (frame index 404). This optimization does not fix
that remaining support-stationarity issue or relax its acceptance threshold.

The new CPU/CUDA regression fails on the parent version because frictionless
groups still create patches. It checks frictional -> frictionless -> frictional
transitions, stale-history rejection, and both combinations with only one
nonzero friction coefficient. Existing split-body and contact tests verify
paired linear/angular momentum and reported impulse wrenches. The focused
temporal-contact/scene suite passed 28 tests, and the existing one-second
120 Hz gear-contact regression also passed. Changed-file hooks passed; the
required repository-wide hook run found pre-existing lint/format issues.

Full-minute artifacts: `/tmp/colibri_perf_{baseline,optimized}_minute.{json,npz}`.

## Reproduction

```bash
uv run -m local_studies.colibri.profile_temporal_performance --output /tmp/colibri-perf
uv run -m local_studies.colibri.profile_temporal_performance --frames 3540 --warmup 60 --validate --output /tmp/colibri-minute
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop -o /tmp/colibri-nsys uv run -m local_studies.colibri.profile_temporal_performance --profile --output /tmp/colibri-profile
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum /tmp/colibri-nsys.nsys-rep
```

Run the same harness on the parent revision for the baseline. JSON records
timing and quality metrics; NPZ records timings and, with `--validate`, every
body pose and velocity. Validation checks assembly bounds, crank tracking,
and fresh contact penetration every frame. It records the existing support
stationarity failure separately without changing that acceptance threshold.

Profiler artifacts: `/tmp/colibri_baseline_nsys.nsys-rep` and
`/tmp/colibri_optimized_nsys.nsys-rep`, with corresponding stats CSV files.
