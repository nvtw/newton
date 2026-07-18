# Full PhoenX Throughput Experiments

This is an append-only ledger for changes to the full PhoenX solver. Results
must use the same Newton collision pipeline, scene, contact policy, substeps,
and iteration count as the comparison run. Kernel-only improvements and
end-to-end improvements are reported separately.

## Hardware and rooflines

Hardware: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB). Software:
Warp 1.15.0.dev20260626, CUDA toolkit 12.9, driver 13.1.

The accepted above-cache calibration from the mini experiments is:

- Sequential float bandwidth: **1,489.14 GB/s**.
- Random scalar gather: **609.60 GB/s**.
- Random vec4 gather: **1,036.82 GB/s**.
- Independent FP32 FMA: **87.810 TFLOP/s**.
- vec3 dot/cross: **82.496 TFLOP/s**.

Hardware performance counters remain unavailable because the driver reports
`ERR_NVGPUCTRPERM`. Roofline percentages are therefore algorithmic lower-bound
models unless explicitly identified as profiler counters.

## F0 — full-solver baseline

Workload: 32,768 worlds, eight free boxes per world on shared ground, one
substep, four PGS iterations, point friction, sticky contact matching, and 200
captured-graph replays after 30 settle and 20 warm-up steps.

- Frame time: **4.025 ms**.
- World-steps/s: **8.140 M**.
- Body-steps/s: **65.122 M**.
- Final contacts: 1,030,013.
- Comparable mini C2 sticky time: **2.866 ms**.
- Full PhoenX is **1.40x slower** than mini C2 sticky end to end.

The full solver does not yet have an accepted logical-byte model. Reporting a
memory-throughput percentage using the mini byte count would be misleading
because the full path performs additional warm-start, ingest, and scheduling
traffic.

## F1 — consume prepared contact lever arms

The multi-world iterate recomputed two quaternion rotations and loaded body
poses, local anchors, and margins during every contact sweep even though rigid
contact preparation already stored the two world-space lever arms in the
derived contact stream. F1 consumes those prepared values. It introduces no
new allocation, solver path, or feature gate.

A controlled 30-replay Nsight Systems A/B used the F0 workload. Final contact
counts were 1,176,460 before and 1,176,795 after, a 0.03% difference.

| Variant | Fused prepare/solve median | Change |
| :--- | ---: | ---: |
| Recompute lever arms per sweep | 669.58 us | baseline |
| Consume prepared lever arms | **573.37 us** | **-14.4%** |

The profiler-run end-to-end time remained about 3.880 ms and was dominated by
other kernels and variable collision/sort work. A separate 200-replay run was
3.858 ms, but it had 4.3% fewer final contacts than F0 and is rejected as a
clean end-to-end speedup claim. F1 is accepted as a hot-kernel improvement;
its end-to-end gain is not yet statistically resolved.

The next target is common contact scheduling and state maintenance. In the F1
trace, median times included 528.4 us for per-world coloring, 267.6 us for
contact matching, 208.6 us for warm-start gather, 140.0 us for current-to-prev
state copy, and 130.1 us for sorted-state save. These costs explain most of the
remaining gap to mini PhoenX.
