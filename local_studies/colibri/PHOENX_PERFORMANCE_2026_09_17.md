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

## Follow-up: reduce preparation and memory traffic

The next candidates retain the same arithmetic and simulation settings:

- Correlate history through a compact list of live group indices, choosing the
  lowest matching original index to preserve dense-scan ordering.
- Parallelize first-fit normal comparisons while retaining the earliest seed;
  build point membership in shared memory and export it coalescently.
- Build a solve-only list of patches with anchors. Preserve the complete
  partition and history, including empty patches.
- Read the normal-impulse row directly instead of serially copying it before
  every friction solve.

The combined full-minute candidate measured 15.933 ms/frame (median 15.887,
p95 17.579), compared with the preceding retained baseline of 18.846 ms/frame.
Every pose and velocity across all 3,600 frames is bitwise identical to that
baseline. Peak penetration remains 0.470375 mm and the same support-creep
failure first appears at frame 404. A fresh paired full-minute baseline/candidate run measured 19.441/16.176
ms per frame: 16.80% less time, or 20.19% more throughput. Candidate final
poses and velocities also match the paired baseline exactly. Artifacts are
`/tmp/colibri_preparation_paired_{baseline,candidate}.{json,npz}`.

Nsight supports the intended reduction in work: average partition kernel time
falls from 28.58 to 4.95 microseconds, history preparation from 23.60 to 15.42,
and the dominant biased sweep from 219.27 to 195.29. Each still runs 8,640 times
in the 180-frame trace. The combined trace is
`/tmp/colibri_combined_preparation_nsys.nsys-rep`; full-minute states and metrics
are `/tmp/colibri_combined_preparation_minute.{npz,json}`.

Rejected experiments include splitting contact and joint sweeps, staging
normal rows in shared memory, skipping zero normal-velocity updates, and
deferring normal-impulse stores: none earned a reproducible gain sufficient
to retain. Parallel angular-response preparation also changed trajectories
and was discarded.

Kernel work, data movement, and layout remain the preferred optimization
route. Solver improvements that support fewer substeps may count only with
independent stability and physical-accuracy evidence; reducing the setting
alone does not qualify. No tolerance relaxation, damping, pinning, or contact
pruning is part of this work.

### Joint accuracy audit

The inherited assembly test checks joint-anchor separation (<2 mm) and
revolute-axis chord error (<0.03) on every validated frame. The harness now
also reports actual source-joint anchor and angular peaks, their joint names,
and frame indices; it stores body labels with the trajectory. These are
geometric measurements outside physics timing, not solver-internal residuals.

An audit of all 3,600 saved candidate frames found:

| Metric | Peak | Joint | Frame index |
| --- | ---: | --- | ---: |
| Anchor separation | 0.630504 mm | Gear_Large__3x_02/Hypocycloid_Gear__3x_02 | 3265 |
| Axis misalignment | 0.0108056 rad (0.6191 degrees) | GearedSpinner/WingLinkArcLeft | 1121 |

These errors are identical to the baseline because every body pose is
bitwise identical. Existing bounds passing does not imply negligible error;
these measured peaks, contact penetration, and drive/support behavior must
all be considered when evaluating any future arithmetic or substep change.
Joint audit: `/tmp/colibri_combined_joint_peaks.json`. The updated reporting
passed a one-frame validation smoke run. The combined candidate passed all
29 focused tests (one module was rerun after correcting a command-line typo).

All changed-file hooks passed. The required repository-wide hook run found
pre-existing lint/format issues; unrelated auto-format edits were restored.

A follow-up experiment accumulated each patch's normal load during the
cooperative normal solve, in the original point order. The final short-run
poses and velocities remained bitwise equal, but time increased to 19.820
ms/frame versus the preceding roughly 16.4 ms short candidate run. The extra
per-row patch loads/stores did not pay for removing the friction-list walk;
the experiment was reverted. Artifact: `/tmp/colibri_fused_normal_load.json`.

A structure-of-arrays normal cache (11 contiguous scalar rows) measured
16.419 ms/frame with exact final pose/velocity parity, essentially unchanged
from the 16.419 ms short run with the original layout. It was reverted rather
than introducing extra load/store helpers for no demonstrated gain. Artifact:
`/tmp/colibri_soa_normal_rows.json`.

Temporary clock64 instrumentation after 60 warmup frames attributed 45.7% of
summed subgroup cycles to normal solving and 54.3% to friction. These include
SIMT waiting between eight-lane groups and are attribution evidence, not an
additive frame-time breakdown. Counters were removed after collecting
`/tmp/colibri_contact_cycle_profile.npy` and its log.

Skipping paired velocity updates for exactly zero friction impulse changes
also preserved exact final states. Short runs measured 16.213 ms initially,
16.326 ms for the matched original implementation, and 16.299 ms for the
closing candidate repeat; candidate median/p95 were not better. This is not
a convincing reproducible gain, so the extra branch was reverted. Artifacts:
`/tmp/colibri_zero_friction_update.json`,
`/tmp/colibri_zero_update_{control,repeat}.json`.

Friction-row prefetching preserved exact final states but measured 16.909
ms/frame and was reverted. Moving reciprocal friction responses into parallel
preparation measured 16.179 ms/frame, too close to the recent 16.326 ms control
to justify changed rounding and further numerical complexity. That experiment
was also reverted without claiming full-minute quality validation. PhysX's
`source/gpusolver/src/CUDA/solverBlockTGS.cuh` motivated the prefetch/response
comparison; its impulse arithmetic was not copied into PhoenX. Artifacts:
`/tmp/colibri_friction_{prefetch,reciprocal}.{json,npz}`.

An explicitly bounded two-anchor cached-friction loop was also rejected:
19.286 ms/frame with exact final states. This exposed the existing maximum
without omitting anchors, but increased time substantially. The original
loop remains in production. Artifact: `/tmp/colibri_friction_bounded_loop.json`.

At frame 60, a read-only census found 231 active friction patches, 341 anchors,
and 211 patches with both zero normal load and zero stored friction impulse
(`/tmp/colibri_patch_load_census.json`). An experimental zero-load fast path
still computed the trial and preserved broken-anchor decisions before skipping
zero clamp/update work. Final body bit patterns matched, but times were
19.090 ms initially (noisy) and 17.648 ms on repeat, so it was reverted.
Artifacts: `/tmp/colibri_unloaded_friction{,_repeat}.json`.

The retained full-minute candidate/baseline comparison was also rechecked as
uint32 bit patterns rather than numeric array equality: every pose and
velocity bit, including zero signs, matches.

A contiguous active-patch index list measured 16.174 ms in a short run versus
16.571 ms for its following control, but its full-minute result was 16.261
ms/frame, within the existing baseline variation. Every full-minute body bit
and all penetration/joint metrics matched. Frozen sweep medians were 165.888
versus 167.936 microseconds, too small to establish a useful frame-time gain.
The representation change and its provisional tests were not retained.
Artifacts: `/tmp/colibri_flat_list_minute.{json,npz}` and
`/tmp/colibri_{flat,linked}_sweep.json`.

### Revised creep interpretation

Powered support displacement is not by itself evidence of creep: shaking and
reaction to crank torque can move a free assembly. Following the user's
clarification, stationarity is evaluated only in motor-off diagnostics;
powered motion still undergoes penetration, joint, and drive tracking checks.
Historical support-failure entries above describe powered displacement under
the old criterion and must not be read as measured unpowered creep. Numerical
parity still proves these optimizations did not alter that powered trajectory.


### Unpowered validation and asynchronous rendering

The new `--motor-off` option removes crank drive stiffness and damping. A
60-second run passed all checks: maximum penetration 0.240596 mm, joint anchor
separation 0.267415 mm, and joint axis error 0.00430454 rad. After the initial
two seconds, maximum horizontal base displacement was 1.31 micrometers and
final displacement was 0.16 micrometers. Artifacts:
`/tmp/colibri_motor_off_minute.{json,npz}` and
`/tmp/colibri_motor_off_drift.json`. Its 15.068 ms/frame is a different workload,
not a solver speedup relative to the powered benchmark.

Colibri defaults to double-buffered render snapshots when supported. Per-buffer
CUDA events protect reuse; rendering waits for the snapshot copy while physics
continues on a nonblocking stream. Rigid snapshots are released after
`log_state`; shared ported examples retain particle snapshots through
`end_frame`. OptiX owns previous instance transforms for motion vectors.
Physics uses the highest available priority (-5 on this GPU), rendering 0.
Priority is a scheduling hint, not preemption.

`profile_optix_overlap.py` measures completed headless 1920x1080 frames with
DLSS performance mode, 30 warmup frames and 120 measured frames:

| Configuration | Mean frame time | FPS |
| --- | ---: | ---: |
| Serialized, two runs | 19.737–20.137 ms | 49.66–50.67 |
| Overlap, equal priorities | 19.184 ms | 52.13 |
| Overlap, priority physics, per-buffer events | 18.811 ms | 53.16 |

All modes render the same pre-step snapshot. Every measured body pose/velocity
bit and the final rendered image match exactly. These rates exclude window
presentation and do not replace the physics-only optimization target.
Artifacts: `/tmp/colibri_optix_serial_repeat.{json,npz}` and
`/tmp/colibri_optix_minimal_priority.{json,npz}`.

Nsight confirms overlapping GPU execution with stream-scoped waits and no
context-wide synchronization in the measured frame loop. The trace used the
earlier conservative buffer wait; subsequent per-buffer events retained image
and physics parity. Trace: `/tmp/colibri_optix_priority_nsys.nsys-rep`.
The installed renderer is the editable `otk-pyoptix` checkout at `49e540d`.

Run `uv run -m newton.examples phoenx_colibri --viewer optix`; append
`--no-render-overlap` for serialized execution or `--motor-off` for unpowered
diagnostics. Five new regression tests fail against the previous implementation;
the focused Colibri/viewer suite passes 34 tests with one skipped.


### Uniform subgroup arithmetic candidate

The cooperative normal solve originally repeatedly entered lane zero for each
row's velocity arithmetic. Running the identical ordered arithmetic in all
lanes while retaining lane-zero impulse/wrench/body writes reduces divergence.
No contact rows, arithmetic expressions, settings, or impulse order change.

The frozen sweep improved from 167.936 to 159.744 microseconds median, with
identical velocity and impulse bits. Short full-frame candidate/control means
were 16.337/16.699 ms. Full-minute candidate/control/candidate means were
16.046/16.416/16.103 ms (about 2.1% throughput improvement for the candidate
mean). Every pose and velocity bit over all 3,600 frames matches both the
control and the previously retained trajectory. Peak penetration remains
0.470375 mm, anchor error 0.630504 mm, and axis error 0.0108056 rad. These are
powered runs, so support stationarity is intentionally not labeled creep.

Matched Nsight graph-node traces attribute the change to the biased sweep:
median 184.430 -> 177.487 microseconds, mean 198.011 -> 189.494 microseconds;
5,760 calls in each trace. Static sweep median remains 16.544 microseconds.
The initial trace omitted graph nodes and had no kernel attribution; only
`*_nodes.nsys-rep` supports these kernel measurements.

All three existing dynamic contact tests pass, covering split/unsplit physical
momentum, full/partial subgroups, and recorded impulse wrenches. No new
behavioral regression is claimed: the intended change is performance only.
The motor-off full minute also matches every baseline pose/velocity bit and
passes unchanged creep, penetration, and joint checks. The reusable saved-run
performance gate requires at least 1% improvement plus exact trajectory and
quality parity. Both candidate minutes pass. A fresh old-code repeat measures
16.473 ms/frame and fails that gate; this is a performance regression
check on measured runs, not a new behavioral difference. Required global hooks
encounter existing unrelated issues; their unrelated auto-edits were restored.
All changed-file hooks pass.
Artifacts: `/tmp/colibri_uniform_lane_{control,candidate}.{json,npz}`,
`/tmp/colibri_uniform_lane_minute{,_control,_repeat}.{json,npz}`,
`/tmp/colibri_uniform_lane_{candidate,control}_nodes.nsys-rep`, and
`/tmp/colibri_uniform_lane_momentum_tests.log`.


Reproduce the performance gate after collecting the above runs:

```bash
uv run --no-project .venv/bin/python -m local_studies.colibri.check_temporal_performance \
  --baseline /tmp/colibri_uniform_lane_minute_control \
  --candidate /tmp/colibri_uniform_lane_minute /tmp/colibri_uniform_lane_minute_repeat \
  --minimum-speedup 1.01
```

Replacing the candidate with `/tmp/colibri_uniform_lane_control_repeat` fails
at the speedup assertion while retaining exact trajectory/quality parity.
This gate complements the repeated measurements and Nsight attribution; it is
not a universal timing threshold or a replacement for controlled hardware.


Skipping empty joint/contact color phases was rejected. A frame-60 census
found 19 colors, with ten of their 38 phases empty; IDs were sorted within
each color. A block-uniform early skip based on the first/last ID preserved
frozen velocity/impulse bits, but median sweep time increased from 159.744 to
161.792 microseconds. The candidate was removed. Artifacts:
`/tmp/colibri_color_phase_census.json` and
`/tmp/colibri_empty_phase_{candidate,control}.{json,npz}`.


### Full-warp contact loading candidate and cross-scene checks

Contact subgroup widths 4/8/16/32 measured frozen sweep medians of
169.984/159.744/155.648/139.264 microseconds. Joint subgroups remain eight
lanes. The full-warp contact variant removes inter-contact warp divergence
while retaining ordered row arithmetic and single-writer impulse/body updates.
Its first finalized full-minute Colibri run measured 15.0246 ms/frame and
matched every retained pose/velocity bit, including unchanged penetration and
joint peaks. The fresh control measured 16.0880 ms/frame; a closing candidate repeat
measured 15.2141 ms/frame. Both candidate minutes pass the saved-run
gate requiring 3% improvement and exact trajectory/quality parity. No Colibri-name check or scene-specific physical setting was
introduced.

Following the user's cross-scene requirement, the candidate was checked
against the prior implementation on these additional workloads:

| Workload | Candidate/control time | Saved-state comparison |
| --- | --- | --- |
| G1, 1,024 worlds, production reduced recipe | 1.9527/1.9501 ms per policy step | All 60 saved joint-position/velocity frames exact |
| DR Legs hold, 64 worlds, 5 substeps and 8 iterations | 11.5283/11.7488 ms per policy step | All 120 saved body-state samples exact over 240 measured steps |
| Kapla, 11,341 bodies, 6 substeps and 10 iterations | 28.0930/28.1547 ms per frame | Baseline is not bitwise repeatable; see below |

These runs use 60 warmup frames; robot resets are disabled. G1's physical
smoke window has no terminations or nonfinite states. DR Legs remains finite,
with minimum pelvis height 0.253724 m and minimum upright cosine 0.999522.
The robot timings include environment observation/reward work. They are not
trained-policy balance acceptance, and small timing differences are not
claimed as gains. Kapla and DR Legs runtime probes confirm both temporal
contact state and color-group topology are absent; G1 uses reduced articulation
kernels, not the changed maximal temporal dispatch.

Kapla candidate/control are initially bitwise equal and first differ at
measured frame 32. A second unchanged-code control also first differs at frame
32, with the same maximum pose-component difference (0.0114795) and velocity-
component difference (1.53025). Candidate/control logs have identical sets of
loaded kernel hashes. The repeated control takes 28.0041 ms/frame. Final tower
heights are 2.792855/2.792775/2.792824 m for candidate/control/repeated control.
This establishes pre-existing nondeterminism in this check; it does not prove
universal bitwise Kapla determinism or long-duration stability.

The older animated DR Legs benchmark fails before simulation because it
expects `/DR_Legs/RigidBodies/pelvis`, while the current cached asset contains
`/dr_legs/pelvis`. The current RL loader supports that asset and was used for
the comparison above. No asset or robot parameters were rewritten to make the
old loader pass.

Artifacts: `/tmp/colibri_contact32_{g1,dr,kapla}_{candidate,control}.{json,npz}`,
`/tmp/colibri_contact32_kapla_control_repeat.{json,npz}`,
`/tmp/colibri_contact32_kapla_variation.json`,
`/tmp/colibri_cross_scene_check.py`, and
`/tmp/colibri_contact32_minute.{json,npz}`. G1 uses the existing
`local_studies.colibri.check_g1_regressions` command with `--world-count 1024
--frames 60 --benchmark-frames 120`. Four focused contact tests pass; the new
full-warp tail-row regression fails against the prior implementation.


The full-warp candidate is retained after the closing checks. All 3,600
motor-off pose/velocity frames also match the retained solver exactly, with
unchanged 0.240596 mm peak penetration and support stationarity passing.
Artifact: `/tmp/colibri_contact32_motor_off.{json,npz}`. Its unpowered timing
is a different workload and is not compared against the powered goal baseline.

Matched Nsight graph-node captures show 5,760 biased sweeps in each run:
median 178.286 -> 156.254 microseconds, mean 189.596 -> 168.642 microseconds.
Artifacts: `/tmp/colibri_contact32_{candidate,control}_nodes.nsys-rep` and
`/tmp/colibri_contact32_{candidate,control}_stats.csv`. All loaded kernel hashes
match across the G1, DR Legs, and Kapla candidate/control pairs; see
`/tmp/colibri_contact32_cross_scene_modules.json`.

The expanded suite passes 11 tests, including ragged temporal contact groups
with multiple batches per color, inter-color ordering, full/partial warp tail
rows, momentum, joint springs, and force reporting. The new tail-row test was
verified to fail against the previous implementation. Required repository-wide
hooks report existing unrelated issues; unrelated auto-edits were restored,
and all changed-file hooks pass.

The Colibri runs here cover 3,600 total frames: 60 warmup and 3,540 measured.
The final optimization-goal gate must measure 3,600 frames after warmup, not
count warmup toward that window. The 9.4 ms target remains unachieved.


Coalescing normal-impulse stores was screened and removed. Each lane retained
its row result, wrote it after the ordered chunk solve, and synchronized the
subgroup before friction. Frozen sweep median improved from 140.144 to 137.216
microseconds with identical velocity/impulse bits. The matched short full-frame
means were 15.128 ms candidate versus 15.254 ms control (about 0.8%), too small
in this screen to justify retaining the added scheduling and synchronization.
The retained solver was restored; no full-minute or low-substep improvement
is claimed. Artifacts: `/tmp/colibri_store_coalesce_{control,candidate}.{json,npz}`
and `/tmp/colibri_store_coalesce_short{,_control}.{json,npz}`.


Additional frozen-sweep screens (all removed from the working solver):

- Full-warp structure-of-arrays normal rows: median 139.264 us versus
  140.144 us control. All four saved velocity/impulse arrays match bitwise;
  the small difference does not justify retaining the layout change.
- Skip velocity updates for exactly zero normal impulse changes: median
  139.120 us versus 139.264 us fresh control, again identical saved outputs.
  No convincing gain; removed.
- PhysX-style precomputed angular normal Jacobians with reassociated normal
  speed and torque arithmetic: median 174.080 us versus the retained roughly
  139 us sweep. Removed before trajectory validation because it is slower.
  This arithmetic experiment is not evidence of numerical equivalence.

Artifacts are `/tmp/colibri_fullwarp_soa_candidate.*`,
`/tmp/colibri_zero_delta_{candidate,control}.*`, and
`/tmp/colibri_jacobian_candidate.*`. None changes substeps, contact frequency,
iterations, or retained physics. Further candidate edits now live in the
isolated `/tmp/newton-phoenx-optimization` checkout so interactive use of the
main checkout cannot accidentally load experimental solver code.

A new retained-code validation corrects the measurement window: 60 warmup
frames followed by 3,600 measured frames (3,660 saved states). Reproduce with:

```bash
uv run --no-project .venv/bin/python -m local_studies.colibri.profile_temporal_performance --frames 3600 --warmup 60 --validate --output /tmp/colibri_retained_full_measured_minute
```

On this run, mean physics time is 17.2778 ms (57.9 FPS), median 17.2559 ms,
and p95 19.7752 ms. This is slower than the earlier roughly 15 ms runs;
those earlier measurements should not be treated as guaranteed throughput.
No competing compute process appeared in the GPU process query. Rendering
is disabled. Fresh contact, assembly, and powered-drive checks pass. Peak
penetration stays at 0.470375 mm and peak joint anchor error at 0.630504 mm.
The extra measured frames expose a larger maximum joint-axis error:
0.0125450 rad at frame 3603, GearedSpinner/WingLinkArcLeft. The first 3,600
pose and velocity states match `/tmp/colibri_contact32_minute.npz` bitwise.
Powered support motion is not reported as creep. The 9.4 ms objective is
still unachieved.


Fully unrolling the 32-lane normal-row source loop also loses: frozen median
147.424 us versus 141.312 us fresh control, with all saved velocity/impulse
bits identical. The isolated candidate was restored to the retained code;
no larger generated kernel was kept. Artifacts:
`/tmp/colibri_unroll32_{candidate,control}.{json,npz,log}`.


The matching motor-off validation also measures 3,600 frames after 60 warmup
frames, using the same command with `--motor-off` and output prefix
`/tmp/colibri_retained_motor_off_full_minute`. It passes fresh-contact,
assembly, and unpowered support checks. Mean time is 15.8777 ms; this is a
different, unpowered workload, not a powered throughput improvement. Peak
penetration is 0.240596 mm, joint anchor error 0.267415 mm, and joint-axis
error 0.00430454 rad. Its first 3,600 pose/velocity states match the previous
motor-off reference bitwise. Relative to saved state index 120 (the first
sample strictly after two simulated seconds), peak horizontal base motion
is 1.2163 um and final displacement is 0.1281 um; peak rotation is
5.8820e-6 rad. These drift values depend on the stated reference sample and
do not indicate a physics change from the earlier bitwise-identical run.


A fresh Nsight trace confirms one physics graph replay per frame: 120
`cudaGraphLaunch` calls for 120 measured frames. Rendering is disabled.
The 5,760 biased sweep calls are graph kernel nodes (48 substeps per frame),
not separate graph launches. Graph launch API time averages 0.244 ms/frame;
the temporal biased sweep still accounts for 55.6% of recorded kernel time.
Artifacts: `/tmp/colibri_retained_fresh_trace.{nsys-rep,sqlite}` and
`/tmp/colibri_retained_fresh_stats.csv`.

An explicit 192-register ceiling was evaluated in the isolated temporal
sweep and retained. The compiler selects 146 registers/thread for the biased kernel rather
than the control's 128; Nsight reports zero local-memory bytes/thread for both.
A 96-register ceiling was neutral in the frozen screen. At 192, frozen median
is 135.168 us versus roughly 139 us retained, and matched 180-frame means are
15.0121 ms candidate versus 15.4095 ms control, with exact trajectories.

Full measured-minute candidate/control/candidate times are 14.8859 / 15.1226 /
14.8667 ms. Both candidates pass the existing 1% speedup gate with every saved
pose/velocity bit and quality diagnostic identical. Candidate motor-off states
also match the full-minute unpowered reference bitwise. All 11 focused tests
pass, covering physical momentum, recorded wrench, full/partial warp tails,
ragged temporal scheduling, joint springs, and force reporting.

In the matched Nsight runs, the biased-sweep median falls from 156.0465 to
152.591 us (mean 169.4096 to 165.3156 us), with exactly 5,760 calls each.
Trace artifacts: `/tmp/colibri_register192_trace.{nsys-rep,sqlite}` and
`/tmp/colibri_register192_stats.csv`. Validation artifacts:
`/tmp/colibri_register192_{minute,minute_repeat,motor_off}.{json,npz,log}`
and `/tmp/colibri_register_control_minute.{json,npz,log}`. The independent
old-code repeat (`/tmp/colibri_register_control_minute_repeat.*`) measures
15.0232 ms and fails the preselected 1% performance gate against the first
control (1.00661x). Both candidate runs beat both controls, spanning roughly
0.9–1.7% improvement depending on the pairing. This is a modest gain; it does
not close the 9.4 ms goal. No arithmetic, contact policy, or solver counts
change. Reproduce the gate with:

```bash
uv run --no-project .venv/bin/python -m local_studies.colibri.check_temporal_performance --baseline /tmp/colibri_register_control_minute --candidate /tmp/colibri_register192_minute /tmp/colibri_register192_minute_repeat --minimum-speedup 1.01
```

Replacing the candidate prefixes with the independent old-code repeat fails.

These runs intentionally use the pre-tail-filter scene in the isolated
checkout; the requested adjacent-feather exclusions are not solver speedups.


After applying the compiler setting to the main checkout, all 15 current-scene
assembly/filter and 120 Hz gear-contact checks pass with the requested tail
filters enabled. Changed-file hooks pass. Required repository-wide hooks
still report pre-existing unrelated issues; their unrelated edits were
restored. The setting changes only the temporal sweep's CUDA register ceiling;
other dispatch paths, including those used by the earlier Kapla/G1/DR checks,
are not modified.

### Rejected cached-friction transform-load removal

The biased temporal sweep reconstructs body transforms before friction even
when prepared friction rows are active. An isolated candidate constructed them
only for the uncached branch; contact arithmetic, order, impulses, and solver
work were unchanged. Frozen candidate/control/candidate medians improved from
133.120 to 131.072/131.072 microseconds, and 1,200-frame candidate/control/
candidate means were 13.9163/13.9525/13.8687 ms. All 240 saved validation
poses and velocities matched byte-for-byte, with identical penetration and
joint metrics.

The required 3,600-frame measured-minute comparison reversed the result:
candidate 14.0894 ms/frame versus control 13.9723 ms/frame, about 0.84% slower.
Reject the candidate despite its frozen-kernel result. A follow-up specialized
the biased cooperative kernel at compile time, removing the unused uncached
branch and transform live ranges entirely. It reproduced the 131.072 us frozen
median, but its 3,600-frame mean was 13.9679 ms versus 13.9723 ms control,
effectively identical. That version was also restored.

No source or changelog fragment is retained. Artifacts use
/tmp/colibri_cached_pose_* and /tmp/colibri_cached_static_*. This does not
advance the 9.4 ms throughput gate.

### Rejected color-local friction phase

A color-local prototype solved ordered normal rows with the retained full-warp
path, stored each independent endpoint state, synchronized the block, and then
used scalar lanes for the per-contact friction continuation. Per-contact
normal-before-friction order and paired impulses were unchanged; color
independence made the cross-contact scheduling legal. Frictionless contacts
returned before the second body load.

The frozen outputs remained byte-identical, but median sweep time increased
from roughly 133 us to 143.360 us. The additional endpoint store/reload and
mid-color barrier cost more than the scalar-lane utilization recovered.
Reject this layout; any future friction parallelism must keep endpoint state
resident rather than externalizing it between phases. Artifact:
/tmp/colibri_color_friction_candidate_frozen.

### Rejected cooperative friction endpoint updates

A full-warp cached-friction prototype retained sequential patch/anchor order
and lane-zero trial arithmetic, but split the two independent paired-body
velocity updates between lanes zero and one before broadcasting the results.
Frozen velocities and impulses remained byte-identical. The added shuffles and
full-warp live state increased median sweep time from roughly 133 us to
176.128 us. The prototype was removed without full-frame testing. Artifact:
/tmp/colibri_friction_endpoints_candidate_frozen.


### Retained three-color mass-splitting slabs

The settled single-world graph has 19 colors but at most eight contacts in a
color. Four-color slabs expose only five sweep blocks, leaving most of the RTX
PRO 6000 Blackwell idle. Two-color slabs measured 10.9441 ms/frame over 360
frames, but failed the existing crank-tracking check (mean -3.1787 rad/s versus
the -3.4907 rad/s target). Reject that setting without relaxing the quality
gate.

Three-color slabs retain all 24 substeps, solver iterations, contacts, and
paired impulse arithmetic while exposing seven independent sweep blocks. A
3,600-frame powered validation measured 12.4727 ms/frame versus the current
13.9723 ms/frame control. It retained crank tracking and support, with 0.607 mm
peak penetration, 0.543 mm peak anchor error, and 0.0113 rad peak axis error.
The control peaks were 0.470 mm, 0.631 mm, and 0.0125 rad respectively. A
separate 1,200-frame repeat measured 12.3772 ms/frame.

The required 3,600-frame motor-off run passed with 0.200 mm peak penetration,
0.328 mm peak anchor error, 0.00358 rad peak axis error, and no support failure.
Thirteen focused temporal/contact momentum tests and 42 subtests pass. This is
a Colibri scene configuration change; global PhoenX scheduling defaults and
other examples are unchanged. Artifacts use `/tmp/colibri_slab{2,3}_*`.
