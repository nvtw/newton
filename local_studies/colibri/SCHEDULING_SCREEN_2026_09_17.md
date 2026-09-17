# Single-world scheduling screen, 2026-09-17

The 9.4 ms/frame full-minute physics goal remains open. These are frozen
kernel screens, not end-to-end speedups. The scene uses the original contact
policy (before the adjacent-tail filtering change), one world, 60 warmup
frames, and unchanged solver counts on the RTX PRO 6000 Blackwell.

Each sample restores 400 state arrays, then times one biased sweep using GPU
events. Restore copies are outside the timed interval. Discard 20 warmups
and measure 200 samples. Check velocity, angular velocity, contact impulses,
and joint impulses bitwise against the original sweep.

| Scheduling | Mean sweep time (microseconds) |
| --- | ---: |
| Control, component experiment | 137.196 |
| Independent connected groups | 125.418 |
| Control, block-size experiment | 136.906 |
| 32 threads | 301.757 |
| 64 threads | 196.573 |
| 128 threads | 144.799 |

All candidates passed the frozen bitwise check. Smaller blocks preserve
row coverage by adjusting contact and joint strides; they are rejected.
The connected-group screen partitions each color slab into disjoint body
components and preserves row order within each component. Its 8.6% kernel
saving excludes component construction, computed on the CPU for this
upper-bound experiment. It does not establish a full-trajectory gain and
is not retained in production. A GPU scheduling implementation would need
to justify its setup cost and complexity first.

Raw results: `/tmp/colibri_component_frozen.log` and
`/tmp/colibri_blocks_frozen.log`. Experimental drivers and kernel copies:
`/tmp/colibri_profile_components.py`, `/tmp/colibri_component_sweep.py`,
`/tmp/colibri_profile_blocks.py`, `/tmp/colibri_sweep_block{32,64,128}.py`.
Both drivers run against `/tmp/newton-phoenx-optimization` via `PYTHONPATH`,
using Newton's venv Python and `--output /tmp/<report>`.

A suspected frozen-benchmark kernel lookup issue was also checked. Both
omitting and explicitly passing the default world count intercepted the
sweep successfully in fresh processes. Both passed the restored-state
bitwise check. The proposed tooling change was unnecessary and removed.
Results: `/tmp/colibri_factory_before.log` and
`/tmp/colibri_factory_after.log`.

## GPU construction follow-up

An isolated prototype now rebuilds components on the GPU once per contact
refresh. One thread owns each slab's union-find; four blocks solve its
components with unchanged row ordering. This is research code, not enabled
in the production solver.

- Matching-policy short control: 15.041695 ms/frame.
- Dense GPU component prototype: 14.466245 ms/frame (1.03978x).
- Dense full minute, 3600 measured after 60 warmup: 14.527612 ms/frame.
  All 3660 poses and velocities, and all recorded quality metrics, match
  `/tmp/colibri_register192_minute` bitwise/exactly. Speedup against that
  earlier full-minute reference is 1.02466x; contemporaneous repeats remain
  necessary. Peak penetration is 0.470375 mm.
- A sparse variant indexes parents by existing body-copy slots, avoiding
  bodies-times-slabs allocation. Its short run measured 14.622051 ms/frame
  (1.02870x against the same short control), with exact trajectory and
  quality parity. Its full-minute and motor-off checks are still pending.

Nsight on the sparse variant captured 5760 biased sweeps over 120 frames.
Their median was 139.673 microseconds, compared with 152.591 microseconds
in the earlier retained-control trace. These captures are not interleaved
repeat measurements and do not establish the final speedup by themselves.

Artifacts: `/tmp/colibri_components_{live,original_control,minute,sparse}`
JSON/NPZ reports; `/tmp/colibri_components_sparse_trace.nsys-rep` and
`/tmp/colibri_components_sparse_stats.csv`. Drivers:
`/tmp/colibri_components_live.py`, `/tmp/colibri_components_sparse_live.py`;
GPU builders: `/tmp/colibri_gpu_components.py`,
`/tmp/colibri_gpu_components_sparse.py`. Run with the isolated checkout on
`PYTHONPATH` and Newton's venv Python. For a module-based control invocation,
set the working directory to the isolated checkout too: otherwise the main
checkout takes precedence and silently uses its newer collision policy.
The mismatched `/tmp/colibri_components_control` run is excluded.

Before retaining: repeat full-minute control/candidate timing, validate the
sparse variant for a full powered and unpowered minute, verify scheduling
and momentum regressions, and check generality and setup/memory cost.
The 9.4 ms target remains unmet.

### Sparse full-minute validation

`/tmp/colibri_components_sparse_minute`: 14.538932 ms/frame over 3600
measured frames after 60 warmup. All 3660 states and quality metrics match
the retained reference exactly (1.02386x against the earlier reference).
The sparse builder's Nsight median is 65.3085 microseconds per refresh,
240 calls over 120 frames. It runs once per collision refresh, not once
per substep.

`/tmp/colibri_component_momentum_covered.log`: four existing tests pass.
Two exercise the experimental temporal scheduler, with 16 intercepted
sweeps each, checking static/dynamic contact impulse wrenches against
physical momentum changes and captured replay. The two ordinary solver
conservation tests do not use the temporal scheduler and are not evidence
of the new scheduling path. The temporary hook covers serial/cooperative
and force-recording factory variants explicitly.

A fresh full-minute control is running as
`/tmp/colibri_components_control_minute`. A further candidate reuses
existing joint/contact slot caches instead of performing repeated slot
searches: `/tmp/colibri_gpu_components_cached.py` and
`/tmp/colibri_components_cached_live.py`. It has not yet been measured.

### Fresh control and cached-slot guard

The fresh full-minute control completed at 14.672049 ms/frame. Against
that reference, the dense and sparse prototypes improve by only 1.00994x
and 1.00916x respectively, while preserving all saved states and quality
metrics. These small gains require better-controlled repeat measurements;
the earlier 2.4% estimate is not confirmed by this control.

The first cached-slot prototype failed the penetration gate at frame 7
(1.389416 mm). Investigation showed that the generic constraint slot cache
is populated for soft constraints, not rigid joints. The experiment had
therefore omitted joint connectivity. This failed prototype was never
installed in production. The corrected variant reuses contact slots and
keeps the valid body-copy lookup for joints. Its 240-state short comparison
passes bitwise trajectory and exact quality parity (1.02870x against the
short control), recorded in `/tmp/colibri_components_cached_corrected`.

`/tmp/colibri_components_paired.py` now runs the corrected candidate and
control sequentially for each frame, alternating which goes first, with
both motors off. It checks every body's pose and velocity bitwise each
frame and runs penetration, assembly, joint-error, and support-stationarity
checks. The full-minute result is pending in
`/tmp/colibri_components_paired_motor_off.{log,json,npz}`. This test keeps
the collision policy, refresh frequency, substeps and iterations fixed.

### Alternating-order decision

The full unpowered comparison completed with every pose and velocity
bitwise equal across all 3660 steps, and all quality checks passing.
Control: 13.586743 ms/frame; corrected cached-slot candidate: 13.678358
ms/frame (0.99330x, approximately 0.7% slower). Peak penetration:
0.240596 mm; maximum joint anchor error: 0.267415 mm; maximum axis error:
0.00430454 rad. The support-stationarity test passed unchanged.

Do not retain the component scheduler: the powered gain against a fresh
control was below 1%, and the alternating unpowered comparison regressed.
The frozen-kernel gain does not justify extra GPU scheduling and scratch
storage. Production solver code remains unchanged by this experiment.

## Uniform joint backward-solve screen

A separate frozen-sweep candidate kept backward substitution and impulse
arithmetic active in all eight joint lanes, with only lane zero writing
accumulated impulses and body states. All checked outputs remained bitwise
identical, but mean sweep time increased from 136.454 to 137.161 microseconds
(medians both 137.216). Reject it. Artifacts:
`/tmp/colibri_joint_uniform_frozen.{log,json,npz}`; experimental sources
`/tmp/colibri_joint_uniform.py`, `/tmp/colibri_sweep_joint_uniform.py`,
`/tmp/colibri_profile_joint_uniform.py`.

## Collision phase attribution

CUDA events around the collision pipeline and solver calls, inside the
captured graph, measured the following GPU fractions over 120 frames after
60 warmup frames. This is a short instrumented phase diagnostic, not a new
full-minute throughput baseline. The one-world run uses the original
contact policy; replicated runs include adjacent-tail collision filters.

| Worlds | Collision | Solver including preparation/integration |
| --- | ---: | ---: |
| 1 | 2.8% | 97.2% |
| 4 | 3.8% | 96.2% |
| 16 | 4.8% | 95.2% |
| 64 | 7.5% | 92.5% |

Artifacts: `/tmp/colibri_phase_times_{single,multi}.{log,json}`;
`/tmp/colibri_phase_times.py`. No competing compute process was present
when checked during the run. The solver remains the dominant target.

## Bounded joint back-solve follow-up

Replacing the dynamic backward loops by six guarded constant iterations
preserves descending solve rows and ascending subtractions within each
row. Frozen sweep: 136.655 -> 133.683 microseconds; all outputs bitwise
identical. Additional gather/forward/scatter unrolling only reached
133.243 microseconds, so retain only the minimal candidate for evaluation.

Alternating-order paired full-minute comparisons (3600 measured + 60 warmup):

| Mode | Control ms/frame | Candidate ms/frame | Ratio |
| --- | ---: | ---: | ---: |
| Powered | 17.306374 | 17.200335 | 1.006165x |
| Motor off | 16.053879 | 15.969297 | 1.005297x |

Every pose and velocity matched bitwise on all 3660 frames in both modes;
all penetration, assembly, drive/support and joint checks passed. Powered
one-second block savings averaged 0.106039 ms (standard error 0.022235 ms),
positive in 47 of 60 blocks. These alternating measurements use two live
models and are not replacements for standalone throughput baselines.

Artifacts: `/tmp/colibri_joint_{unrolled,bounds}_frozen.*`,
`/tmp/colibri_joint_unrolled_paired_{short,minute,unpowered}.*`.
Candidate helper: `/tmp/colibri_joint_unrolled.py`; sweep:
`/tmp/colibri_sweep_joint_unrolled.py`; full-frame adapter and driver:
`/tmp/colibri_joint_unrolled_live.py`, `/tmp/colibri_joint_unrolled_paired.py`.
No production change yet: broader regressions and standalone/Nsight
confirmation remain before deciding to retain it.

A frozen experiment omitting dynamic friction code failed the velocity
bitwise gate (`/tmp/colibri_normal_only_frozen.log`), so do not assume that
all dynamic contacts are frictionless. No friction behavior was changed
in production.

Nsight Compute 2025.4.0 was found under `/usr/local/cuda/`; it is not on
PATH. Both attempted launch orders failed during Python startup before
kernel profiling (`/tmp/colibri_ncu_sweep{,_direct}.log`). The driver also
reports `RmProfilingAdminOnly: 1`. No usable counter profile was obtained
and no driver configuration was changed. Nsight Systems remains available.

### Actual-helper validation

The minimal bounded-loop change was applied to the isolated checkout's
canonical `bilateral_joint.py` (not the working example). All 22 tests in
bilateral preparation, direct drive, D6 drive, color-group conservation,
and temporal contact-force suites pass (`/tmp/colibri_joint_unrolled_tests.log`).

The canonical helper's standalone full-minute run preserves all 3660
saved states and quality metrics exactly, but measured 16.735662 ms/frame.
It **fails** the speed gate against the earlier 14.672049 ms control;
do not use the earlier paired gain as proof that this standalone run improved.
The actual frozen kernel still measures 133.796 microseconds, consistent
with the wrapper experiment. A fresh full-minute unchanged control is
running as `/tmp/colibri_joint_fresh_control_minute`; the isolated helper
has been restored before that control. The main solver remains unchanged.

A preload of the system libpython allowed a minimal Nsight Compute Python
startup probe to exit successfully, but the full profiling launch still
failed with exit code 11 before profiling (`/tmp/colibri_ncu_sweep_preload.log`).
No usable Nsight Compute counter data is available. No Python packages or
driver settings were changed for these probes.


### Fresh control and Nsight confirmation

The fresh unchanged standalone control completed at 16.846104 ms/frame,
versus 16.735662 for the canonical candidate (1.00660x). Both include
3600 measured frames after 60 warmup frames. All 3660 saved states and
quality metrics match exactly. The earlier 14.672049 ms control is not a
contemporaneous comparison; timing conditions changed between batches.
The alternating powered and motor-off comparisons above independently
support a small improvement, not a large throughput gain.

Nsight Systems, 120 measured frames after 60 warmup frames, confirms
5760 main biased sweeps in both versions. Their median falls from
151.839 to 149.7115 microseconds (1.40%); mean falls from 158.5753 to
157.2770 microseconds (0.82%). Long outliers remain in both traces.
These kernel gains are not whole-frame speedups. Artifacts:
`/tmp/colibri_joint_{control,candidate}_trace.nsys-rep`,
`/tmp/colibri_joint_{control,candidate}_stats.csv`, and
`/tmp/colibri_joint_fresh_control_minute.{json,npz,log}`.

The minimal candidate is now applied to the main checkout for validation.
No substep, iteration, collision, material, drive, or constraint policy
changes are included. Main-checkout joint and momentum regressions pass (24 tests), including
replicated-world and shared-global-body contact momentum checks; see
`/tmp/colibri_joint_main_validation.log`. Focused Ruff lint and formatting
checks pass. The change is not committed; repository-wide hooks and the
performance regression gate remain before committing.
The 9.4 ms/frame target remains unmet.


### Cooperative backward-row experiment

Distributed descending backward-solve rows across the eight joint lanes,
broadcasting each solved value while retaining ascending inner subtraction
order. Frozen outputs remain bitwise identical. Against the minimal
bounded-loop candidate, the new variant is slower: 134.684800 versus
134.055361 microseconds (medians 135.168 versus 133.120). Reject it; the
additional lane communication does not pay for this six-row solve.
Artifacts: `/tmp/colibri_joint_backward_lanes_frozen.{log,json,npz}`;
`/tmp/colibri_joint_backward_lanes.py` and
`/tmp/colibri_profile_joint_backward_lanes.py`. No production change.

The final Nsight Compute bootstrap attempt also failed at Python startup
with exit 11; `/tmp/colibri_ncu_bootstrap.log`. No counter data or driver
configuration change resulted. Continue using measured timings and Nsight
Systems rather than inferring occupancy or register stalls from this failure.


Delaying the body-response load until after the cooperative nonleaders exit
also preserves frozen output bits but loses performance: 135.359360 versus
134.144481 microseconds. Reject this variant too. It may trade live values
for additional loads; there is no counter evidence establishing the cause.
Artifacts: `/tmp/colibri_joint_late_response_frozen.{log,json,npz}` and
`/tmp/colibri_joint_late_response.py`. Neither experiment changes the retained
minimal bounded-loop candidate or the 24-substep work configuration.


### Register limits after bounded backward substitution

Frozen sweep, 200 samples after 20 warmups per variant, identical restored
inputs and all outputs bitwise equal. Repeat reverses candidate order.
These are exploratory kernel measurements, not end-to-end speed claims.

| Maximum registers | Forward order mean us | Reverse order mean us |
| --- | ---: | ---: |
| 192 (current) | 134.678401 | 133.937760 |
| 128 | 133.506400 | 133.118400 |
| 160 | 133.731361 | 133.279520 |
| 224 | 134.120320 | 133.824641 |
| 256 | 133.986081 | 133.748480 |

128 is about 0.6–0.9% faster than 192 in these screens, enough only to
justify an alternating full-frame screen. No production register limit
has changed. Artifacts: `/tmp/colibri_registers_bounded_frozen.*`,
`/tmp/colibri_registers_bounded_reverse.*`, and
`/tmp/colibri_profile_registers_bounded{,_reverse}.py`.


The 128-register bounded-loop variant completes a powered full-minute
alternating comparison: 14.217925 -> 14.166496 ms/frame
(1.003630x). All 3660 states are bitwise equal; peak penetration
0.470375 mm, anchor error 0.630504 mm, axis error 0.012544996 rad, and
assembly/drive checks pass. This is only a 0.36% paired frame gain; no
production register setting has changed and standalone/repeat/unpowered
confirmation remains. Artifacts: `/tmp/colibri_register128_bounded_minute.*`.


### Inactive inequality iteration screen

An isolated early return after body-state conversion skips the body load
and zero-impulse scatter for inactive limits with zero friction, retaining
the required friction multiplier reset. Ball/universal rows with no
angular limits likewise skip the no-op body roundtrip. No production
change yet. Expanded frozen parity includes constraint multipliers and
body access modes, in addition to velocities/contact/joint impulses.

Two frozen measurements: 133.861761 -> 131.071040 us and
133.986560 -> 131.070720 us, all checked outputs bitwise equal. Short
alternating full frames (180 measured + 60 warmup): 14.531424 ->
14.347068 ms (1.012850x), all 240 states bitwise equal and quality checks
pass. Full-minute powered/unpowered, transition regressions and Nsight
confirmation are still required. Artifacts:
`/tmp/colibri_inactive_inequality_{frozen,state,short}.*`;
`/tmp/colibri_inactive_inequality.py` and associated sweep/live/paired scripts.


Powered full-minute paired confirmation for inactive inequality iteration:
14.334873 -> 14.175937 ms/frame (1.011212x).
All 3660 poses/velocities match bitwise; penetration 0.470375 mm, joint
anchor error 0.630504 mm and axis error 0.012544996 rad match control;
assembly/drive checks pass. Artifact:
`/tmp/colibri_inactive_inequality_minute.*`. Motor-off full-minute comparison
is running sequentially as `/tmp/colibri_inactive_inequality_unpowered.*`.
No production early-return change yet. This approximately 1.1% gain does
not meet the 9.4 ms/frame objective.


Motor-off full-minute paired confirmation completed: 13.064452
-> 12.956495 ms/frame (1.008332x). All 3660 states
match bitwise; support-stationarity checks pass unchanged. Peak penetration
0.240596 mm, anchor error 0.267415 mm and axis error 0.004304540 rad match
control. Artifact: `/tmp/colibri_inactive_inequality_unpowered.*`.
The early return is now applied to the main checkout for broader regression
validation (`/tmp/phoenx_inactive_inequality_tests.log`), not committed yet.
Standalone repeated throughput and Nsight confirmation remain outstanding.


The main-checkout inactive-inequality change passes all 28 tests in joint
position limits, Coulomb friction/drives, distance and velocity bounds, D6
dispatch, and temporal contact momentum (69.498 s). Artifact:
`/tmp/phoenx_inactive_inequality_tests.log`. Keep the change uncommitted
until standalone repeat/Nsight and performance acceptance are complete.


### Reuse resolved body slots (isolated screen only)

The common joint body loader resolves each body slot in the linear read
and again in the angular read. Reusing the existing
`read_angular_velocity_with_slot` helper preserves all frozen output bits.
Forward order: 133.486081 -> 133.120160 us; reverse order:
134.172160 -> 133.256802 us. This small kernel difference is not a proven
frame improvement; no production body-loader edit was made. Artifacts:
`/tmp/colibri_body_pair_slots_{frozen,reverse}.*`,
`/tmp/colibri_body_pair_slots.py`, and the corresponding joint/sweep/profile
wrappers. Assess end-to-end cost before retaining this optional follow-up.


### Combined standalone confirmation in progress

Runner: `/tmp/colibri_joint_combined_confirmation.py`. It snapshots the
main checkout's two pending helper edits, compares them against the
`5ccefea2a` versions of `bilateral_joint.py` and `joint_inequality.py`,
and runs control A, candidate A, candidate B, control B sequentially in
the original-policy isolated checkout. Each run uses 3600 measured frames,
60 warmup frames and `--validate`. It then collects 120-frame Nsight traces
for control and candidate, and restores the isolated files in `finally`.
No register-limit or slot-reuse experiment is included.

Control A completed at 14.307503 ms/frame. Candidate A is now running;
all remaining acceptance comparisons are pending. Artifacts use
`/tmp/colibri_joint_combined_*`; orchestration log:
`/tmp/colibri_joint_combined_confirmation.log`. Do not infer a combined
speedup from the earlier separate or paired measurements.


Combined standalone ABBA sequence completed:

| Run | Mean ms/frame |
| --- | ---: |
| Control A | 14.307503 |
| Candidate A | 14.052157 |
| Candidate B | 14.398756 |
| Control B | 14.522045 |

The existing saved-trajectory gate passes for each adjacent pair: 1.01817x
and 1.00856x, all states and quality metrics exact. Conditions drifted
across the sequence; candidate B is slower than control A, so do not
claim every cross-comparison wins or present the average as noise-free.

Nsight main biased sweep: 5760 instances in each trace; median
151.710 -> 145.679 us, mean 161.7902 -> 155.2003 us. Long outliers remain.
Artifacts: `/tmp/colibri_joint_combined_{control,candidate}_stats.csv`
and corresponding traces. This supports a modest combined improvement,
not achievement of the 9.4 ms/frame target. The runner restored the
isolated source files after completion.


Retention gate: both adjacent standalone pairs pass the existing
`check_temporal_performance` with `--minimum-speedup 1.005`, exact saved
trajectory and quality checks enabled. Comparing unchanged control B
against unchanged control A fails this improvement gate, as expected
(`/tmp/colibri_joint_combined_old_gate.log`). This is a measured performance
acceptance check, not a deterministic correctness unit test; control
variation and the paired/full-minute/Nsight evidence above remain essential.
Correctness suites for the two changes passed 24 and 28 tests respectively.
