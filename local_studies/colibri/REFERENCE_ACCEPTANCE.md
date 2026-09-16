# Colibri reference and acceptance checks

## Current acceptance status (2026-09-15)

The current experimental PhoenX fused bilateral/contact configuration passes
1200 frames (20 seconds) in two independent runs after the predictive contact
fixes. Every one of74 recorded contact/state fields matches bit-for-bit in
every frame. Peak fresh penetration is150.59um, joint anchor1.00122mm and
axis0.00723613rad. An isolated300-frame measurement gives22.464ms/frame
(44.515FPS), versus PhysX89.97–91.52FPS at matched temporal settings.
This establishes20-second stability and same-GPU determinism for the tested
configuration, not unbounded reliability or hardware-independent bit matching.
The public example is present; broader integration and checks remain pending.
Kamino and reduced PhoenX have not passed the full mechanism. Historical
screens below are superseded by the current results near the end.

- Kamino DVI/Schur: prefixes 1–14 pass 180 frames each at 120 Hz collision
  updates, four physics substeps per 60 Hz frame, 16 iterations, SOR 1 and
  no global damping. Maximum fresh penetration is 76.8 micrometers.
  Prefix 15 fails with 1.336 mm penetration. Matching PhysX's automatic
  offsets reduces the contact count but still fails (1.668 mm).
- PhoenX reduced: corrected-source prefixes through 14 pass short checks.
  The optional forest solver also passes the first large-gear stage, but
  prefix 15 still fails after the current-gap correction (1.709 mm at frame 4
  in the 10-substep, eight-sweep diagnostic). It is an experimental
  contact solve across independently factored articulated trees, not an
  accepted full-scene solver setting.
- PhoenX maximal/direct: a friction torque defect was reproduced and fixed
  using a common contact impulse application point. Faster-motion checks
  exposed stale contact geometry during velocity relaxation; current
  geometry is now refreshed, with conservation regressions passing.
- PhoenX experimental fused bilateral joint blocks: 18 joint semantic tests
  pass, including internal motor momentum, ball anchors and bounded drives.
  After the ordinary speculative mesh-contact fix, stage 8 passes 180 frames
  at unchanged 30 temporal substeps and one sweep per 120 Hz update. Maximum
  depth is 76.8 micrometers initially and 13.68 micrometers finally; previously
  it failed at frame 3 with 1.54 mm. Its 18.50 ms/frame is a partial-scene
  diagnostic. Stage 15 also passes 180 frames (maximum 164.6 micrometers).
  **Full 37-body scene passes 180 frames**, maximum 365.2 micrometers and
  final 60.5 micrometers, at **92.25 ms/frame (10.84 FPS)** after warmup.
  Report: `/tmp/colibri_bilateral_fused36_livegap_180.json`.
  Halving to 15 temporal substeps also passes all 180 full-scene frames:
  maximum 342.3 micrometers, final 74.4 micrometers, **47.77 ms/frame
  (20.93 FPS)**. Report: `/tmp/colibri_bilateral_fused36_livegap_15x1_180.json`.
  The prototype is disabled by default; longer checks and optimization remain.

## Primary matched-step comparison

Use **120 Hz collision updates and 30 temporal steps per update**, plus the
final velocity relaxation, for the primary comparison. The eight- and
15-step screens below are secondary speed/accuracy experiments, not matched
PhysX benchmarks.

After the real friction-anchor reset, full PhoenX passes 300 frames at these
settings: **61.079 ms/frame (16.37 FPS)**, maximum fresh penetration
**208.47 micrometers**, final **41.93 micrometers**. Source masses, SOR 1 and
zero external damping are retained. Report:
`/tmp/colibri_anchor_candidate_matched 30x1_300.json`.

Two new isolated PhysX runs start from rest with every body damping coefficient
zero, without profiling or contact reporting. They measure **11.115 and
10.927 ms/frame (89.97 and 91.52 FPS)**. Reports:
`/tmp/colibri_physx_clean_matched.json` and
`/tmp/colibri_physx_clean_quiet_repeat.json`. The repeated final state agrees
with the earlier no-damping control. Use these as the current matched-startup
reference; the historical 116.65 FPS result retains authored startup/damping.
The historical unsplit configuration above costs roughly 5.5 times as much;
the current combined preparation prototype reduces this to about 2.0 times.

An explicit `--contact-offset-map` diagnostic option accepts the measured
PhysX map for all 102 colliders, including its 20 mm ground-plane offset.
Early PhoenX baselines use authored offsets with a 1 mm fallback on
unauthored shapes. Current combined runs use the complete exact map.

### Measured source of the cost

The actual PhysX runtime uses `solveBlockUnified`, not the optional
small-island kernel found during the earlier source inspection. Its trace
contains eight batches times 31 sweeps times two collision updates per frame.
Across 30 captured frames, the main solve takes 83.647 ms total, approximately
**2.788 ms/frame**. Triangle/triangle detection takes approximately
**1.734 ms/frame**. Instrumented timings are not standalone FPS benchmarks.

PhysX's main kernel uses 128 registers/thread, zero local-memory spill storage,
64-thread blocks and 21–24 blocks. PhoenX's costly tail kernel uses 154
registers/thread, zero local-memory spill storage and one 256-thread block.
The optimized 15-step PhoenX profile spends 21.63 ms in its main sweep and
4.64 ms in ordered warm preparation; independent point geometry is only
0.168 ms. The rigid point loop already keeps each body pair's velocities
local, so removing repeated body loads is not a new optimization.

PhysX reports **2,103 normal points**, 944 friction anchors and 67 nonempty
collider pairs, with up to 174 points per pair. Only 59 reported points have
nonzero impulse in that snapshot. Thus its point count is comparable to
Newton's roughly 2,200, rather than orders of magnitude smaller. The reporting
overlay leaves final positions and velocities bitwise unchanged but adds
substantial reporting overhead; do not use that run's FPS. Evidence:
`/tmp/colibri_physx_contact_workload_summary.json`.

PhysX's body-copy solve scales each copy response by its active copy count
and averages the copy velocity changes by the reciprocal count. Algebraically,
this recovers the physical inverse-mass response to the summed impulses.
Consistent counts and balanced impulse points are essential. This motivates
an opt-in mass-split joint/contact prototype, not changing physical model masses.

## Source fidelity

The original file is `/home/twidmer/Documents/colibri/Colibri.usd`.

- Source gravity is **1 m/s²**, converted from 100 cm/s².
- All 83 cylinder prims are represented by **32-sided meshes**, retaining
  their radial phase and axial orientation. Reconstructed vertices agree
  within 37 nm. An axis-only orientation is insufficient.
- Physical masses match the PhysX reference within 4.3 ppm and local inertias
  within 7.4 ppm after the cylinder correction.
- Per-shape source SDF resolutions are retained. `sdf_resolution=0` selects
  these values rather than a uniform resolution.
- 52 colliders explicitly use 1 mm contact offsets; three use 2 mm.
  Unauthored offsets are calculated by PhysX from the minimum full local
  geometry AABB dimension and the collision timestep:
  `max(2 * dt² * max(gravity, 1 stage-unit/s²), 0.02 * min_dimension)`.
  Positive rest offset is added. The typical automatic result is 0.139 mm;
  the ground plane defaults to 20 mm. These values differ from the explicit
  1 mm fallback used in the current Newton diagnostics.
- The accepted PhysX runtime has 44 frictionless colliders and 57 with
  friction 0.5, matching the source material interpretation.
- All enabled source joints disable collisions between their two bodies.
  Newton includes all 1,244 implied collider pairs. No additional authored
  collision groups or filtered-pair relationships were found. Disabled
  Flower joints do not imply filtering.

The exact runtime offset map is saved in
`/tmp/colibri_physx_exact_contact_offsets_m.json`.

## PhysX performance reference

`benchmark_physx.py` loads the original USD through installed `ovphysx` and
steps at 120 Hz. Run it with Python 3.12:

```sh
uv run --no-project --python 3.12 --with ovphysx --with numpy python \
  local_studies/colibri/benchmark_physx.py --frames 300
```

Initialization must call a normal `step_sync(1/120)` **before tensor reads**.
Otherwise the SDK's automatic minimal-timestep initialization of the saved,
slightly misaligned joint poses produces enormous correction velocities.
Earlier reports obtained without this initialization are invalid.

The accepted five-second run is saved in
`/tmp/colibri_physx_reference_normal_first.json` and the matching NPZ:

- Mean completed physics frame: **8.573 ms**, or **116.65 FPS**.
- 95th percentile: 9.545 ms. Rendering is excluded.
- Maximum final linear speed: 0.146 m/s; angular speed: 7.67 rad/s.
- Maximum origin displacement from the initialized pose: 0.288 m.
- Re-evaluating its final poses using Newton's fresh SDF contacts gives
  **54.31 micrometers** maximum penetration. Initial maximum is 79.94 µm.

This run retains source damping and initial velocities. A separate five-second
control starts PhysX from rest with every rigid body linear and angular damping
coefficient set to zero. It remains bounded and its final fresh Newton SDF
penetration is **41.925 micrometers**. This excludes body damping and saved
initial velocities as explanations for the Newton failures. The control
observed 8.805 ms/frame, but concurrent GPU work makes that timing diagnostic
only. Results: `/tmp/colibri_physx_at_rest_no_damping.json` and
`/tmp/colibri_physx_at_rest_no_damping_contacts.json`.
PhysX is a performance and robustness reference, **not a conservation oracle**.
No other large GPU jobs were launched during the accepted timing window;
a pending agent compilation could have completed a tiny regression step.

## Newton diagnostic tools

From this worktree, use the shared Newton environment:

```sh
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python \
  -m local_studies.colibri.validate_phoenx \
  --mode reduced --body-count 15 --frames 180
```

`validate_phoenx.py` recomputes contacts independently of cached solver
contacts, checks collision filters, joint/motion checks and penetration,
and writes a unique JSON report and failure NPZ. Physics timings exclude
validation and rendering. Its `--substeps` means substeps per 120 Hz outer
update, so four means eight physics substeps per 60 Hz frame. This differs
from Kamino's example `sim_substeps`, which counts per 60 Hz frame.

`profile_phoenx.py` attributes eager CUDA kernel costs. It is neither an FPS
benchmark nor a physical acceptance test. Initial profiles show sequential
contact sweeps dominate, including in the one-body ground-contact case.

The passing experimental full-scene configuration can be reproduced with:

```sh
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python \
  -m local_studies.colibri.check_bilateral_pgs \
  --fused --body-count 36 --substeps 15 --iterations 1 --frames 180
```

`36` selects all mechanism bodies; the separate kinematic Flower makes 37
bodies in the model. This route uses local diagnostic installers and is not a
public solver option. The report excludes its first ten frames from warm
physics timing but validates every frame, including startup. Changes after
the saved baseline require revalidation.

Conservation and friction energy require separate isolated regression tests.
The full mechanism exchanges momentum with the ground and cannot establish
those invariants by itself. Retain SOR 1, physical masses/inertias, all real
contacts, and no artificial global damping when addressing failures.


## Collision cost and solver regressions

The complete source scene at its authored poses uses **0.429 ms per 60 Hz
frame** for two collision queries, including reduction, sorting and sticky
matching. There are 2,184 generated contacts. This measurement uses a warmed
CUDA graph and excludes dynamics and rendering; it does not establish cost
throughout motion. Report: `/tmp/colibri_collision_profile.json`.
Collision detection and reduction do not explain the measured solver cost.

Confirmed fixes are tested against physical invariants rather than agreement
with PhysX alone:

- Shared impulse application points prevent artificial friction torque.
- Current joint/contact geometry is refreshed for velocity relaxation.
- Reduced and ordinary mesh preparation now recompute the current separation
  during internal steps. A stale positive generation-gap floor let a cube
  travel 0.5 mm through its support between collision queries. The unchanged
  regression fails before removal and passes afterward.
- Coupled normal and metric friction projection fixes an ordinary rigid-contact
  case that gained kinetic energy without external work. Energy, momentum and
  friction optimality tests exercise the correction.

The initial correct friction implementation regressed Kapla from 23.53 to
213.80 ms/frame. Optimizations preserve the objective and tolerance. The final isolated
measurement is **33.650 ms/frame (29.72 FPS)**, still about 43% slower than
the old path. Report: `/tmp/kapla_rigid_metric_final_isolated.log`. The mass-split Kapla settling regression is now resolved by a real two-anchor
friction reset. Correct metric friction and tangent position recovery remain
enabled; no damping or assertion relaxation was added. Lifecycle, frame
objectivity, history/reordering, energy and momentum checks pass.

Kamino's new private normal projection primitive solves a frozen feasible
normal problem, but the complete biased source problem is locally infeasible.
Offline configuration recovery improves joint alignment while respecting fresh
contact bounds. Holding velocities unchanged during that recovery changes
angular momentum, so it must not be inserted as a physical production step.

A one-time contact-aware initial pose reconciliation does not resolve Kamino
stage 15: the unchanged source control fails at frame 3 with 1.311 mm depth,
and the reconciled initialization fails at frame 3 with 1.265 mm. Initial
joint mismatch alone is therefore insufficient to explain the runtime failure.

The first full fused solver profile totals 96.08 ms of instrumented CUDA
kernels. Two tail-fused prepare/solve kernels account for 80.70 ms; bilateral
block preparation takes 1.79 ms. Thus solver work dominates the passing
configuration too. Report: `/tmp/colibri_profile_fused36_livegap.json`.


## Work distribution audit and bounded prototypes

The current node-level Nsight trace measures the momentum-preserving body
integration at about **0.1765 ms/frame** (120 calls, 2.94 us/call), compared
with 42.78 ms/frame in fused constraint iterations and 8.51 ms/frame in cached
contact preparation. This two-frame capture establishes kernel attribution,
not an acceptance benchmark. Earlier event-timer body-integration figures
included queue gaps. Conservation itself is not yet an explanation for the
remaining speed difference. Artifact: `/tmp/colibri_phoenx_resource_nodes.sqlite`.

PhysX's contact report headers are not solver work items. The GPU preparation
selects one patch per descriptor and limits a patch to six points
(`constraintBlockPrePrep.cu:985`, `PxgCommonDefines.h:36`). Thus 2,103 reported
points require at least 351 contact descriptors. Newton currently processes
roughly 2,100 points in 38–39 body/material columns. Splitting these ranges
without deleting points is an experimental work-distribution change; finite
iteration ordering and convergence still require validation.

The coarse-column copy-aware bilateral prototype passes five seconds of the
full source scene, but costs **101.69 ms/frame (9.83 FPS)** at the matched
30 temporal steps per 120 Hz collision update. It is slower than the accepted
unsplit parallel-preparation route (**61.079 ms/frame, 16.37 FPS**). There
are only 49 active copy slots, at most two per body, and 19 overflow constraints
in three batches. It is not a performance improvement.

Six-point chunking is opt-in through `--contact-chunk-size 6`. Initial
conservation, energy, drive and coverage checks pass, but source stage 8 does
not yet pass fresh penetration checks. The first metadata correction extends
`pair_source_idx` along with the column ranges; after that correction stage 8
still fails at frame 89 with 1.269 mm penetration. Do not advance this variant
or quote its speed as a valid scene result until the failure is resolved.

A separate zero-friction fast path was array-exact on its acceptance cases
but slowed the matched full scene from 61.31 to 64.13 ms/frame. It remains
disabled by default. Avoid accepting a specialization based on arithmetic
counts alone.

Kamino's matched-step diagnostic is dominated by its cooperative sparse PGS:
60 calls take 8.039 seconds, 99.6% of summed GPU kernel time. A single warp
per world handles the active contact sweeps. The loops use active counts;
spare contact capacity is not the dominant runtime issue. This diagnostic
is not a full-scene stability or usable-FPS result. Artifacts:
`/tmp/colibri_kamino_one_frame.*`.


### Exact per-shape contact offset comparison

The accepted unsplit parallel-preparation route with measured PhysX offsets
fails at frame 297 (4.95 s), with 3.178 mm fresh penetration between TailRack
and TailPinion. It retains 30 temporal steps per 120 Hz collision update,
SOR 1 and no global damping. The 56.38 ms/frame timing is from a failed run
and must not be advertised as a stable performance result. The prior
five-second pass uses 1 mm fallback gaps on unauthored shapes; it does not
establish stability with the exact PhysX offset policy. Failure state and
report: `/tmp/colibri_phoenx_exact_offsets_30x1_300.npz` and `.json`.


### Normal-first contact optimization

A controlled matched 30x1 full 300-frame comparison passed with array-identical
final poses, velocities, impulses, friction anchors and derived rows. The
reference costs 62.6701ms/frame and the candidate 55.2611ms/frame (18.10FPS),
an 11.82% improvement. Both reached 267.295 um maximum penetration and 38.790 um
final penetration with 1 mm fallback gaps. The optimization avoids tangent
mobility work only when the admissible friction load is exactly zero; normal
and stored tangent impulse release are preserved. It does not reduce contacts
or weaken the friction solve. The 128-case exactness suite, focused energy,
momentum and anchor tests, and the 11,341-body Kapla settling test pass after
production integration. Reports: `/tmp/colibri_normal_first_reference_300.json`
and `/tmp/colibri_normal_first_candidate_300.json`, with NPZ state comparisons.

The smaller-group failure audit identified a coloring conflict before the
physical failure: two constraints sharing FrameGround received color 1.
Equal packed priorities are currently not ordered strictly by element ID.
A deterministic tie-breaking fix and fail-before regression are in progress;
until verified, the chunk results cannot diagnose physical convergence.


### Equal-priority graph coloring correction

The repeated-edge regression fails before the correction for JP, greedy,
speculative and endpoint-owner coloring: adjacent constraints can commit to
the same color when their packed priorities are equal. The corrected
comparison adds element ID as a deterministic tie-break; endpoint-owner
winner keys use 64 bits while preserving its existing unsigned priority order.
The new captured-repeat regression passes for all four algorithms, together
with 42 existing speculative, overflow, general and warm-start test methods.
The source chunk simulation must be rerun: earlier failures occurred with
invalid simultaneous endpoint writers and do not establish the convergence
of correctly scheduled chunks. No physical mass or impulse law changed.


### Launch size and zero warm-start work

With production normal-first and the priority tie fix frozen, matched full
300-frame runs pass with array-identical final poses, velocities, impulses,
friction anchors and derived rows:

- 256-thread tail: 55.0361ms/frame (18.17FPS).
- 32-thread tail: 53.5947ms/frame (18.66FPS).
- 32-thread tail plus exact-zero warm-start geometry skip: 52.4595ms/frame
 (19.06FPS).

All reach 267.295um peak penetration and 38.790um final penetration. These
use 1 mm fallback offsets; the exact-offset failure remains separate. The
launch-size and zero-warm-start changes remain local opt-in prototypes.
The threshold is reduced with the block size so larger colors retain the
existing head fallback; no columns are silently omitted. Zero warm-start
rows are skipped only when all three stored impulse components equal zero,
and nonzero impulses retain their accumulation order. Reports:
`/tmp/colibri_tail_launch_comparison.json` and
`/tmp/colibri_zero_warmstart_comparison.json`. Small speed differences merit
confirmation before any general default change.

The exact-offset tail failure reproduces at frame 297 after both production
fixes, with identical final poses and no endpoint-color conflicts throughout
the run. Late guide contacts are absent before fresh penetration appears;
the subsequent guide impact precedes the pinion failure. This shifts the
next diagnosis toward contact coverage between 120Hz queries rather than
the resolved coloring race. Snapshot transport estimates are not exact
closest-point distances and must not be confused with them.


## Current combined exact-offset result

The full mechanism with measured PhysX physical offsets, existing 5mm-capped
velocity-derived search extension, six-point contact columns and two columns
per Jacobi batch passes 1200frames/20s. Every frame checks joint attachment,
fresh penetration, endpoint-color exclusivity and contact ownership; physical
shape gaps remain unchanged. Maximum penetration 156.1um, final 60.9um,
maximum 1860cached points of 8192capacity. Artifact:
`/tmp/colibri_chunks6_batch2_exact_predictive_30x1_1200.json`.

An isolated 300-frame run gives 27.090ms/frame (36.91FPS). A subsequent
controlled scheduling comparison reduces 27.446 to 24.999ms/frame (40.00FPS)
by using one head launch per conditional-loop body plus batch-aware tail
eligibility. Final poses, velocities, all contact impulses, anchors and
derived rows are array-identical over that 300-frame comparison. Both
scheduling changes remain local. They also pass a separate 20-second run:
maximum joint-anchor error 0.946283 mm and axis error 0.0112952 rad across
1201 samples and all 38 joints. Report:
`/tmp/colibri_joint_accuracy_1200.json`.
Artifacts: `/tmp/colibri_combined_isolated_300.json` and
`/tmp/colibri_combined_head_launch_comparison.json`.

The warmed two-frame CUDA-node profile isolates the current baseline:
prepare fused/head 4.32/1.92ms per frame; iterate fused/head 6.84/3.68ms;
bilateral block preparation 1.90ms; body-copy broadcast 1.09ms; momentum
integration 0.203ms. Each head callback unrolls eight kernel launches, including
no-op launches after handoff; this motivated the exact-state scheduling test.
The profile is an attribution instrument, not an FPS benchmark.
`/tmp/colibri_combined_kernel_attribution.json` and
`/tmp/colibri_combined_captured_profile.sqlite` retain full details.

Offline source-joint residuals quantify the saved-state accuracy:
Phoenx final snapshot has 183.2um maximum anchor error and 1.111mrad maximum
axis error; the clean PhysX snapshot has 96.1um and 2.367mrad respectively.
These are different simulation phases (PhysX includes its warmup) and are
not a matched trajectory or peak-error comparison. All 38 source joints are
covered. Reports: `/tmp/colibri_phoenx_joint_accuracy_300.json` and
`/tmp/colibri_physx_joint_accuracy_final.json`.

### Copy-aware preparation and trajectory accuracy

Parallel point geometry with ordered copy-aware warm preparation reduces an
isolated comparison from 25.61 to 22.64 ms/frame (39.04 to 44.17 FPS).
The full scene passes 20 seconds with maximum fresh penetration 184.77 um.
Unlike the scheduling changes, the full trajectory is not array-identical.
A frozen-input audit locates the first difference in effective-mass rows:
222 entries differ by at most three FP32 ULPs (5.82e-11 absolute).
Headers, history, impulses and copy/body velocities initially agree exactly.
Both variants have maximum relative error 5.015e-7 against FP64 evaluation;
momentum and energy tests pass. This is accepted as a numerically equivalent
physics implementation, not an exact-state optimization. Canonical private
helpers and a mixed-copy regression are integrated. The supported constructor
option `parallel_contact_prepare=True` now selects canonical dispatch. A
subsequent 20-second canonical-helper run also passes: peak joint anchor
0.879583 mm and axis 0.00727713 rad across 1201 samples. Report:
`/tmp/colibri_canonical_splitprep_joint_accuracy_1200.json`.
Evidence: `SPLIT_PREPARE_AUDIT.md` and
`/tmp/colibri_splitprep_candidate_head1_1200.json`.

A separate no-damping PhysX 20-second trajectory has peak joint-anchor error
3.23195 mm and axis error 0.0924983 rad; the first five seconds peak at
0.458081 mm and 0.00959601 rad. An independent calculation using composed USD
frames and normalized Gf quaternion matrices agrees within 66.1 nm and
4.19e-7 rad. This excludes a measurement units/frame artifact, but does not
establish a PhysX solver bug. The first 2 mm anchor crossing is frame 1021.
These trajectory tests are accuracy evidence, not isolated speed measurements.
Reports: `/tmp/colibri_physx_joint_trajectory.json` and
`/tmp/colibri_physx_joint_trajectory_verified.json`.

### Rejected or unaccepted variants

Batch 1 passes prefixes 8/15 but fails the full scene at frame 30 with 2.093mm
joint-anchor separation. Joint-first priority does not fix this: its stage 15
fails at frame 15 with 2.258mm separation. Both retain the existing acceptance
thresholds and are rejected; the accepted configuration uses batch 2.

Kamino's frozen same-input layout replay is bitwise equal but slower:
126.067ms baseline versus 127.395ms transposed. The componentwise scatter
prototype changes tiny floating-point results even in an untouched compact
world. Both unaccepted production options were removed, with a local
reproduction patch retained in the Kamino study directory. Proven physics
fixes remain; no Kamino speed improvement is claimed.


## Constructor integration and slab experiment

`joint_solver="block_pgs"` now owns block data and callbacks through an
internal `BlockJointSystem`, with the existing direct strategy as default.
All 18 existing analytical drive/conservation assertions pass through this
constructor. Three production tests additionally cover bounded motor impulse,
six-component momentum, dynamic-row rebinding after property notifications,
and unsupported combinations. Disabling the row rebind reproduces failure.
`contact_chunk_size=6` and `parallel_contact_prepare=True` replace the old
process-global contact installers. Temporal relaxation and head/tail scheduling
in the studies still use local wrappers; full example integration is pending.

The local GPU slab scheduler passes exact schedule/copy membership tests and
actual callback momentum/energy/drive tests. Prefixes 8 and 15 pass 30 frames;
full Colibri passes 300 frames. However, width-eight slabs FAIL at frame 372
in the 1200-frame test with 1.1006 mm tail rack/pinion penetration. Topology and
ownership audits pass; joint peaks remain 0.5565 mm and 0.00685 rad.
The roughly 49 FPS diagnostic screen is NOT an accepted stable configuration.
Report: `/tmp/colibri_slabs8_full_1200.json`. Diagnose contact coverage versus
finite-iteration convergence before changing physical offsets or timesteps.


### Late public integration failure under investigation

The constructor-owned route passes 30 full-scene frames but fails the 1200-frame
check at frame 1133: Frame/GearedSpinner anchor separation 3.662 mm and fresh
gear penetration 1.2367 mm. Report `/tmp/colibri_public_solver_1200.json`.
Do not treat small analytical tests as full-scene acceptance. A one-frame
public versus current legacy comparison has bit-identical poses, velocities,
impulses, history, derived rows, graph arrays, and packed contact headers.
The legacy compatibility shim now uses the constructor-unified preparation,
whereas the earlier passing canonical-helper run used its prior installer.
A current legacy 1200-frame replay is running to isolate these changes.
The 40 FPS scheduling-only 20-second baseline remains the conservative
reference while the faster preparation integration is investigated.


### Repeatability and completed schedule integration

A current legacy 1200-frame repeat and a public 1200-frame repeat both pass
with bit-identical final poses, velocities, impulses, contact history, derived
rows, headers and graph arrays. Thus the earlier public failure cannot be
attributed to API integration alone. A further instrumented 1200-frame A/B
also passes both trajectories, but first differs in angular velocity at
frame 316 (1.79e-6 rad/s), followed by pose at frame 317. Generated normals
already differ earlier, while chunk headers and ownership agree. The late
failure remains a robustness issue, not an accepted deterministic outcome.

`velocity_relaxation="final_substep"` now replaces the local schedule wrapper
in public-route studies. Its test matches the prior independently scheduled
contact-star fixture exactly, including changing contacts, momentum and energy.
The initial test caught missed dispatcher count accesses; fixing those makes
both split and unsplit comparisons pass. Nine combined joint/chunk/schedule
tests pass after integration. Canonical head/tail factories now compare actual
overflow batch counts against available lanes; their ASTs matched the validated
prototype exactly before removing the local source clones. The old
`--batch-aware-tail` study flag is now a compatibility no-op. Head launch
unrolling remains at its default outside the local head-chunk study option.


### Proven allocation-dependent normal encoding

Predictive export reused an existing contact's already encoded normal, but
fresh allocation normalized the same normal again before octahedral encoding.
A frozen128-normal test reproduced differences for19 normals, up to5.96e-8.
Passing the original normal to storage while retaining normalization for
approach-speed calculations fixes the branch-dependent encoding. Four
CPU/CUDA × fast/deterministic tests fail before and pass after. This is a
confirmed determinism defect; full-scene determinism after the fix remains
unverified. It is not yet proven to explain the late physical failure.

Frozen slab-tail analysis separately shows132 raw predictive witnesses at
outer746,76 of which become penetrating. Every one of those76 is already
predictably closing at generation (initial gaps0.654–1.116mm, normal speed
-0.140 to-0.0968m/s, predicted end gap-0.342 to-0.003mm). The reduced16 retain
none of them. Adding8 coarse normal-octant guards passes generic coverage
cases but does not fix this source witness set; a60-degree normal difference
can occupy one octant. That intermediate experiment is not accepted. Reusing
the existing20 normal-bin depth slots for eligible predictive contacts is
being tested as a bounded coverage correction.


### Verified post-fix baseline

Reports `/tmp/colibri_fixed_contacts_reference_1200.json` and
`/tmp/colibri_fixed_contacts_repeat_1200.json` both pass. Hash directories
`/tmp/colibri_fixed_contacts_hash_reference` and
`/tmp/colibri_fixed_contacts_hash_repeat` contain1200 identical74-field
records (generated contacts at both120Hz updates, ingestion/history and body
states). The repeat summary has no first difference of any kind.
The isolated300-frame report `/tmp/colibri_fixed_contacts_isolated_300.json`
measures22.464ms/frame (44.515FPS); instrumented runs are not timing evidence.

The slab8 variant still fails after the normal-family fix, now frame448 with
1.3705mm Rack/MountRight penetration. Report
`/tmp/colibri_slabs8_normalcoverage_full_1200.json`.
It remains excluded; the validated baseline uses two-column overflow batches.


### Sixty-second stability and broader scenes

The post-fix two-column-batch configuration passes3600frames/60s with all
per-frame contact and joint checks. Peak anchor1.00122mm and axis0.0101094rad
across3601samples. Report `/tmp/colibri_fixed_contacts_3600.json`.
Timing from this concurrent correctness run is not a new FPS measurement.

An isolated existing Kapla11341-body/249027-point comparison tests head-loop
unrolling8 versus1:21.733ms/46.014FPS versus22.052ms/45.347FPS. All six final
body/contact arrays are bit-identical. The1.47% slowdown is small but merits
repeat timing before changing the shared default. Reports
`/tmp/kapla_head8.json` and `/tmp/kapla_head1.json`.

Existing G1 maximal/direct standing acceptance fails both tracked HEAD and
current code: mean tracking0.055366rad and0.052880rad respectively against
0.04rad. The current result is slightly better, but neither passes. An
unchanged reduced4-world zero-action run falls in both versions by2s; that
is not a learned balancing-policy test. A cached actual29DoF locomotion
policy has been found and is being tested with its existing configuration.
Do not claim overall G1 acceptance from constructor or short finite checks.


### G1 regression attribution and first exact optimization

The cached learned G1 controller passes 10 simulated seconds on current code.
Matched original/current isolated policy-step timing initially measured
30.470/47.918 ms, a 57.3% regression. Captured-node profiles identify added
mass factorization, response construction, and contact-row rebuilding;
see `G1_REGRESSIONS.md` and `G1_REDUCED_REFRESH_AUDIT.md`.

Removing body-space impulse-response construction when the active reduced
contact path does not consume it gives 48.143 -> 41.779 ms/policy step in
a fresh matched comparison (13.2% improvement). Recorded 20-frame body
pose/velocity trajectories are bit-identical, as is final base height after
130 steps. Reports `/tmp/g1_response_before.json` and
`/tmp/g1_response_after.json` with companion NPZ files. Necessary mass and
contact-geometry refresh remains. The remaining regression is still open.

A separate scheduling audit found reduced solves and maximal tree projection
could run on substeps with zero active relaxation iterations. Two CPU
regressions fail before the fixes and pass after: the reduced entry point
uses the active count and both unsplit dispatchers return before any
relaxation impulse when it is zero. Default G1 each-substep scheduling is
unchanged. Focused scheduling files pass Ruff.


### Public example defaults and complete scheduling checks

The actual public command (no study hooks) passes 300 frames with `--test`,
including the separate fresh-contact detector and inherited joint/state checks:

```sh
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m newton.examples phoenx_colibri --viewer null --num-frames 300 --test
```

Exit status 0; log `/tmp/colibri_public_default_300.log`. Its isolated
`--benchmark --num-frames 360` run reports 40.9 FPS (357 measured frames
in 8.73 s), log `/tmp/colibri_public_default_benchmark.log`. This is the
production default head unroll of 8; the 44.515 FPS study result uses 1.
Do not quote the latter as the unchanged public example's measured FPS.

All four scheduling tests, including GPU changing-contact comparisons, pass
in 2.312 s; log `/tmp/colibri_schedule_all.log`.

The same public command also passes `--num-frames 3600 --test` (60 simulated
seconds), exit status 0. Log `/tmp/colibri_public_default_3600.log`. This
uses production defaults, no study installation hooks, and all independent
post-step checks. The run overlapped other correctness tests; its elapsed
time is not performance evidence.


### Tail-first scheduling prototype (not yet integrated)

`tail_first_dispatch.py` starts each round with the existing fused small-colour
kernel. It arms the persistent head only when a large colour remains. This
preserves the colour and impulse order and avoids unconditional head launches
for scenes whose colours all fit the fused kernel. Generated experimental
functions live under `/tmp/colibri_tail_first_dispatch`; production files
remain unchanged.

Kapla correctness and isolated runs match all six saved body/contact arrays
bit for bit, including contact impulses and lambdas. Matched isolated results:

- Public Colibri, 357 measured frames: tail-first 43.6 FPS / 8.18 s;
  default 40.4 FPS / 8.84 s. Logs `/tmp/colibri_tail_first_benchmark.log`
  and `/tmp/colibri_head_first_repeat_benchmark.log`.
- Kapla, 30 warmup + 180 measured frames: tail-first 45.547 FPS,
  default 44.600 FPS. Reports `/tmp/kapla_tail_first_isolated.json`
  and `/tmp/kapla_head_first_repeat.json` with companion state NPZ files.

These are modest single paired timing results. Full public Colibri stability
with the prototype and additional dispatch regressions remain required before
production integration.

Tail-first prototype completes the actual public example's 3600-frame/60-second
`--test` run with exit status 0. Log `/tmp/colibri_tail_first_3600.log`.
All inherited joint/state checks and independent fresh-contact checks ran.
This overlapped other correctness work and provides no timing measurement.


### Integrated scheduling and prepare grid

The production solver now drains small colours first and sizes prepare lanes
to `min(128, contact_chunk_size or 128)`. Unchunked preparation retains128
lanes; the point kernel arithmetic is unchanged. Compatibility study entry
points are now no-ops for these integrated optimizations.

Combined1200-frame/20-second state/contact comparison matches all74 hashed
fields per frame exactly: `/tmp/colibri_compact_tail_hash/summary.json` has
no first difference and no state difference; scene report
`/tmp/colibri_compact_tail_1200.json` passes. The standalone tail-first
public3600-frame test also passes as documented above.

The preparation-grid production regression fails before in six narrowed
chunk configurations and passes after; all five preparation tests pass
(`/tmp/colibri_prepare_grid_production_{before,after}.log`). The all-small
scheduler test fails before with one executed head round instead of zero.
Mixed cloth coverage exposed an existing undefined-lever compilation path;
its static-branch repair and final scheduler validation are in progress.

Combined isolated benchmark reports44.2 FPS,357frames in8.08s
(`/tmp/colibri_compact_tail_first_isolated.log`). The earlier
`colibri_compact_tail_first_benchmark.log` overlapped a short gradient test
and must not be used as isolated timing evidence. The grid's incremental
speed gain is small; the main measured gain comes from tail-first dispatch.

## Current public driven-motion verification

Optional `--save-history` in `check_public_analytic_gradient.py` records
unique post-step poses and authored-joint twist statistics. A fresh public
1200-frame run passes every original check: peak depth0.21271mm,
anchor1.16513mm, axis0.0079581rad. Crank motion is10.76637 turns over
19.9833s, mean3.38517rad/s against target3.49066rad/s; no sampled stall.
Both wings span approximately1.5rad. Final q/qd are bitexact to the earlier
no-history public1200 run. See `MOTION_AUDIT.md` for evidence and limits.
This overlapping correctness run is not timing evidence.

Public-constructor slab8/envelope with canonical parallel preparation
passes3600frames: peakdepth0.13573mm, anchor0.4521mm, axis0.01595rad.
Two1200-frame repeats match all74 state/contact hashes and q/qd histories
exactly. Isolated paired330-frame timing (30warmup,300measured) gives
18.918ms/52.86FPS candidate versus22.147ms/45.15FPS public baseline,
a14.6% frame-time reduction. This remains a local experimental candidate;
contact-admission coverage limitations are unresolved generally.

## Minimum-substep experiment

The current no-envelope color-group schedule passes 60 simulated seconds
at10 temporal substeps per120Hz collision update (20 steps/renderframe),
SOR1, one biased iteration/substep, no global damping. N9 fails startup.
Peak fresh penetration161.18µm, anchor1.92263mm, axis.020112rad. Crank
motion31.806turns confirms this is not a stalled-mechanism pass. The
minimum has narrow attachment-error margin; it is a finite-duration test.

Isolated mean8.96868ms/frame (111.50FPS), p9510.23846ms. No static factor
optimization. This is not temporally matched to the prior PhysX30 result.
See `SUBSTEP_STABILITY.md` for thresholds, failure times of the public
schedule, exact command, reports and comparison caveats.

## Maintained implementation at the restored 30-step baseline

The public Colibri example now selects canonical color groups of size8.
Its default remains30 temporal steps per120Hz update, as requested after the
stability sweep. The actual public330-frame run passes and final q/qd match
`/tmp/colibri_public_slab8_rebased_isolated_330.state.npz` byte for byte.
Report `/tmp/colibri_public_integrated_groups30_330.json`.

Canonical slot-cache integration passes independent CPU/CUDA endpoint tests,
including pinned/out-of-range nodes and legacy partition equivalence.
Canonical fixed-bound block preparation passes84 frozen cases; the seven
preparation/group/blockpolicy tests pass. At30 steps, plain fixed-loop
preparation measured54.727FPS versus the generated-source study55.067FPS,
with identical final arrays. Production uses ordinary fixed loops.
