# Current public Colibri profile

> Stationary-sculpture requirement: the recorded 300-second run is NOT an
> acceptable Colibri result. FrameGround drifts 437 mm and rotates 0.859 rad.
> It passes only the earlier scoped stability checks. See BASE_STATIONARITY.md.

Configuration: public example, groups of four colors, 64 sweep blocks,
30 substeps per 120 Hz collision update, one biased iteration, SOR 1,
no global damping. Two warmed frames (31 and 32) were captured with
Nsight Systems CUDA graph node tracing.

| Work | GPU ms per displayed frame |
|---|---:|
| Main frequent group sweep | 5.235 |
| Other frequent group sweep | 1.639 |
| Bilateral joint block preparation | 1.602 |
| Deterministic color builder | 0.727 |
| Rigid-to-copy broadcasts | 0.684 |
| Parallel contact preparation | 0.211 |
| Mesh SDF collision/reduction kernel | 0.147 |

Total summed kernel time is 13.649 ms/frame. These are instrumented
attribution numbers, not FPS: the sweep has outlier launches, and collision
also includes candidate work, sorting, and commit kernels outside the single
SDF kernel listed above. The main ordered solve remains the leading cost.

The source/launch path was not changed by the profiling harness.
Artifacts:

- `/tmp/colibri_public_groups4_grid64_profile.nsys-rep`
- `/tmp/colibri_public_groups4_grid64_profile.sqlite`
- `/tmp/colibri_public_groups4_grid64_kernels.csv`
- `/tmp/colibri_public_groups4_grid64_profile_summary.json`

Reproduce from the Phoenx worktree:

```sh
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true -o /tmp/colibri_public_groups4_grid64_profile uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -u -m local_studies.colibri.profile_public_colibri
```

## Direct public-example timing after exact coloring optimization

A separate isolated run of the public example passed all checks and measured
13.77873 ms/frame mean, 14.66780 ms/frame p95, or72.5756 FPS (30 warmup,
300 measured frames). Final q/qd are byte-identical to the previous linear
color-builder run. Timing excludes rendering and read-only physical audits.
The GPU process list was empty before launch; peers held GPU work.
Artifact: `/tmp/colibri_public_group4_binary_isolated330.json` and `.npz`.

This direct command uses the example's built-in exact offsets and needs no
separate temporary map:

```sh
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.check_public_analytic_gradient --frames 330 --substeps 30 --warmup 30 --output /tmp/colibri_public_group4_binary_isolated330.json
```

## Validated geometric default after inactive-joint preparation

Fresh two-frame Nsight capture uses the unmodified public default,30 substeps
per120Hz contact update. Summed GPU kernel time is12.894531ms/frame. This is
instrumented attribution, not a new isolated FPS measurement (the separate
isolated default remains77.3636FPS).

| Work | ms/frame |
|---|---:|
| Two main ordered sweep kernels |6.062966|
| Bilateral joint block preparation |1.603714|
| Serial graph coloring |0.872968|
| Main radix-sort kernel |0.450955|
| Body-to-copy broadcast |0.376555|
| Copy averaging and broadcast |0.372155|
| Speculative candidate picking |0.350702|
| Direct joint row preparation |0.308413|
| Speculative candidate validation/commit |0.282893|
| Mesh/SDF collision kernel |0.2612|
| Dynamic joint velocity snapshot |0.2423|
| Parallel contact preparation |0.2105|

The mesh/SDF row is one kernel, not the complete collision pipeline. Two
captured frames include a large sweep outlier (861microseconds), so this
profile is useful for priorities rather than precise speedup predictions.
The main measured costs remain constraint sweeps and joint preparation.

Artifacts: `/tmp/colibri_geometric_default_profile.nsys-rep`, corresponding
`.sqlite`, `/tmp/colibri_geometric_default_kernels.csv`, and
`/tmp/colibri_geometric_default_profile_summary.json`.

## Latest isolated integrated-default measurement

After rigid-row coloring integration, the public default measured12.411642ms
per frame (80.5695FPS), p9513.477665ms,30 warmup/300 measured frames.
All ten saved arrays match the fused-preparation and launch-size experiments
exactly; neither experiment improved timing and neither was promoted.
Artifact: `/tmp/colibri_fused_begin_baseline_ab330.json`.


## Overlap-priority reducer, current cooperative solver

A30-frame capture after30warmups uses the corrected public baseline with
30substeps per120Hz update, one final relaxation sweep, SOR1 and no damping.
Sources/assets match the300s validation reference.

| Work | GPU ms/frame | Calls/frame |
| --- | ---: | ---: |
| Main ordered sweep |3.281204|60|
| Other frequent sweep |1.180624|60|
| Bilateral joint block preparation |1.678999|62|
| Color-row builder |0.544138|2|
| Main radix-sort kernel |0.448359|—|
| Copy averaging/broadcast |0.378431|—|
| Speculative candidate picking |0.376215|—|
| Direct joint rows |0.305631|—|
| Speculative candidate validation |0.294149|—|

Summed and union kernel durations both equal10.952999ms/frame; first-to-last
kernel span is11.621256ms/frame. These are instrumented attribution numbers,
not FPS. Outliers remain (maximum main sweep873us versus55us p95); no paired
speedup is inferred from the older two-frame capture.

Bilateral preparation is now a substantial target: each launch handles about
38 joints with one thread per joint; each thread builds/factors up to6rows.
Measured mean27.08us, p9525.92us,104registers/thread and zero local memory.
A local cooperative row-wise prototype is being prepared to retain exact
FP64 accumulation/subtraction order and FP32 responses. It is not integrated.

The independent300s baseline validation measured89.4906FPS/11.174356ms,
with the limitations documented in SPATIAL_PRIORITY_VALIDATION.md.

Artifacts: /tmp/colibri_spatial_priority_profile.nsys-rep, corresponding.sqlite,
_profile_summary.json and _profile_provenance.json. Reproduce capture with
`profile_public_colibri --capture-frames 30`, then
`summarize_kernel_profile DATABASE --frames 30 --output REPORT`.


## Integrated cooperative bilateral preparation

The CUDA preparation now uses eight lanes per joint, with a scalar CPU
fallback. Ordered FP64 factorization and FP32 body responses match the
reference exactly; no solver options or iteration counts changed.

An isolated paired run of 330 frames (30 warmup) measured:

| Preparation | Mean ms/frame | FPS |
| --- | ---: | ---: |
| Scalar control | 11.112763 | 89.9866 |
| Integrated cooperative | 10.131364 | 98.7034 |

This reduces frame time by 8.83%. All ten saved arrays, including every
Colibri pose/velocity sample, are byte-identical to the current reference.
Sources and assets remained unchanged during both measurements. The frozen
38-joint kernel benchmark measured 25.7535 us versus 11.0835 us per call,
using twelve forward/reverse interleaved samples of 128-call graphs.

Kapla's six saved arrays (180 frames) and G1's two state histories (20 learned
controller steps) match their existing references byte for byte. These checks
preserve those references; they do not establish arbitrary-scene determinism
or erase previously documented performance differences versus older versions.
The integrated 300-second Colibri stability/full-trajectory gate passed:
98.6414 FPS (10.137728 ms mean, 11.220050 ms p95), with all ten arrays
byte-identical to the pre-optimization 18,000-frame reference. Source/assets
and the reference remained unchanged. Peak penetration was 0.762952 mm,
anchor error 0.794343 mm, and axis error 0.0289467 rad. The axis limit is
0.03 rad, so the remaining margin is small. Free-assembly drift and difficult
mass-ratio convergence remain separate open issues.

Artifacts:

- `/tmp/colibri_prepare_integrated_scalar330.{json,npz,source.json,log}`
- `/tmp/colibri_prepare_integrated_cooperative330.{json,npz,source.json,log}`
- `/tmp/cooperative_prepare_frozen_timing.json`
- `/tmp/colibri_cooperative_prepare_kapla.{json,npz}`
- `/tmp/colibri_cooperative_prepare_g1.{json,npz}`
- `/tmp/colibri_cooperative_prepare_velocity1_long18000.{json,npz,source.json,log}`

See `COOPERATIVE_BILATERAL_PREPARE.md` for the exactness failures, rounding
fix, integration tests, and reproduction commands.
