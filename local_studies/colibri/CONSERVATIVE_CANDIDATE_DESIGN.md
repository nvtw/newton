# Candidate admission versus contact activation

PhysX GPU sphere narrow phase receives contact distances and transforms,
not velocities (gpunarrowphase/src/CUDA/cudaSphere.cu:62-137).
It admits nearby geometry through the geometric contact distance. CPU
geomutils/src/contact/GuContactSphereSphere.cpp:44-62 makes the same distinction:
the inflated distance controls candidate admission, while the stored distance
is the true surface separation. TGS recomputes cached separation from temporal
body motion before unilateral impulse projection
(gpusolver/src/CUDA/solverBlockTGS.cuh:299-310).

This supports separating admission from activation. It does not prove that
the original Colibri USD uses speculative CCD or velocity-expanded PhysX
contact distances.

Current Newton predictive admission discards positive-gap witnesses that are
initially receding, even inside its velocity-expanded geometric search envelope.
Joint/contact forces can reverse those velocities during a 120 Hz collision interval.

The local conservative_mesh_candidates installer selects the existing ordinary
reduced mesh/SDF kernel while leaving CollisionPipeline speculative search on.
The kernel receives the same expanded shape_gap search array and the same
separate authored shape_base_gap. Ordinary normal/spatial/voxel reduction retains
geometric witnesses regardless of initial velocity sign. Contact distances,
model shape gaps, collision margins and solver equations are unchanged.
The existing 255-per-pair export budget remains. This first prototype is limited
to reduced mesh/SDF candidate generation plus the shared reduced export writer.
Primitive direct writers retain their original policy; reduced buffered paths
also pass through the changed final geometric-admission writer.

The envelope bounds current speed and angular speed. It does not guarantee
coverage under arbitrary acceleration, especially from zero initial speed or
motion exceeding the 5 mm search cap.

Required acceptance:
- Isolated receding candidates have zero normal/tangent impulses.
- Paired internal velocity reversal activates them only when the solver's live
  geometric gap predicts crossing within its current microstep.
- All six momentum components remain conserved; contact impulses do not add
  kinetic energy. Reversal impulse work is accounted for separately.
- Compare frozen failing source witnesses, normal coverage, row counts,
  hash failures, allocation high-water and existing contact capacity.
- No accepted timing claim until an isolated benchmark; no canonical policy
  change before these physical and capacity tests pass.

## Bounded findings

The two-body rotated-box test admits dormant candidates with exactly zero
impulse. A paired velocity reversal preserves all six momenta and does not
add contact kinetic energy. It nevertheless FAILS the strict stopping test:
three retained support points lie entirely on nonpositive local Y; subsequent
rotation produces 4.05 mm oriented-box overlap. This is a real coverage failure,
not merely a center-distance proxy error. No criterion was relaxed.

Frozen source queries at outer745 retain33 rows instead of13. Two newly
retained rows initially recede at +0.0980/+0.0993 m/s with gaps1.025/0.943 mm,
then transport to -0.190/-0.375 mm at the captured post pose. At746 the
counts are18 original versus48 envelope rows. This proves useful additional
coverage only; it does not establish an altered trajectory is stable.

Artifacts: /tmp/colibri_slab_tail_envelope_frozen_coverage.json and
/tmp/colibri_slab_tail_envelope_retained_reversal.json.

## Source trajectory screen

The local slab8/envelope route passes300 and1200 frames (20 simulated seconds),
crossing the prior448-frame failure. All original fresh collision, joint,
slab assignment, color and copy-count audits remain enabled. Peak fresh depth
is105.37 micrometers, final10.88 micrometers; peak joint anchor0.481 mm and
axis0.011327 rad (including the authored starting pose). Maximum cached points
2357 stays below8192. Actual shape_gap arrays remain bit-identical; maximum
search extension is the configured5 mm. Solver equations, source normal
gradients,120 Hz narrowphase and30 temporal steps per update are unchanged.
No isolated performance claim is made for these correctness runs.

The screen establishes useful scene behavior, not a general contact coverage
guarantee: the independent rotated-box fixture still fails its stopping test.
Canonical promotion therefore remains out of scope.

Report: /tmp/colibri_slab8_envelope_1200.json. Reproduce with:

    uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.check_slab_envelope --body-count 36 --frames 1200 --substeps 30 --iterations 1 --fused --mass-splitting --contact-chunk-size 6 --max-colored-partitions 8 --mass-splitting-batch-size 2 --contact-offset-map /tmp/colibri_physx_exact_contact_offsets_m.json --output /tmp/colibri_slab8_envelope_repeat.json

## Revalidation with analytical SDF gradients

After the canonical analytical local-cell SDF gradient correction, the same
local slab8/envelope route passes300 then3600 frames (60 simulated seconds).
Peak fresh depth157.56 micrometers, final10.83 micrometers; peak joint anchor
0.5678 mm and axis0.011420 rad. All original topology, joint and per-frame
fresh collision audits pass. Maximum2391 cached points is below8192; physical
shape gaps remain bit-identical and search extension remains capped at5 mm.
These runs overlapped other correctness tests: recorded runtime is not an
isolated benchmark. No collision or solver production policy was changed.

Report: /tmp/colibri_slab8_envelope_analytic_3600.json. Source provenance:
/tmp/colibri_slab8_envelope_analytic_sources.json. The independent rotated-box
coverage failure was observed on the earlier sampler and remains a documented
limitation; this scene success does not establish a general coverage guarantee.

## Isolated timing and repeated trajectory

On matching330-frame runs with30 warmup frames excluded (300 timed), the
public baseline measured22.0435 ms/frame (45.36 FPS) and the local slab8
envelope measured21.1985 ms/frame (47.17 FPS). Both use30 temporal steps
per120 Hz collision update, source offsets and the same harness physics-only
timer; fresh checks remain outside timing. This is a3.8 percent frame-time
improvement, still below60 FPS.

Two subsequent20-second hash runs match exactly:1200 q/qd states and all74
per-frame hashes including both generated-contact boundaries, histories,
prepared rows and impulses. See /tmp/colibri_slab8_envelope_hash_b/summary.json.

A local public-constructor/parallel-prepare overlay is being tested separately.
Its first frozen preparation input is bit-identical to serial slab preparation;
only effective masses differ by at most3 ULP. An independent FP64 audit of5430
directional masses bounds both routes at4.04e-7 relative error. All other
first-preparation fields remain bit-identical.

## Preferred simpler control after corrected normals

Public constructor + parallel point preparation + slab8 WITHOUT the local
conservative admission overlay passes300 and3600 frames (60 seconds).
Peak fresh depth111.68 micrometers, final10.94 micrometers; peak anchor0.5479 mm
and axis0.011832 rad. Maximum1934 cached contacts, unchanged physical gaps,
all original color/copy/joint/penetration checks pass. This control uses the
existing public initial-twist predictive admission and corrected analytical
SDF normals. It removes the generic admission-policy limitation from this
solver-scheduling candidate. It does not establish coverage guarantees for
other scenes.

Report: /tmp/colibri_public_slab8_noenvelope_3600.json and final state
/tmp/colibri_public_slab8_noenvelope_3600.state.npz.

The envelope/public-parallel route independently passed60 seconds and exact
20-second state/contact-hash repetition. Its paired isolated timing was
18.9181 ms (52.86 FPS) versus public baseline22.1469 ms (45.15 FPS).
Do not attribute this isolated number to the newer no-overlay control before
its own timing. Current envelope-route eager profile attributes8.67 ms to
biased slab iterations,2.58 ms to joint block preparation,2.45 ms to ordered
slab warm starts, and only0.344 ms to parallel contact geometry.

## Post-relaxation-rebase acceptance and static preparation

After the ordinary relaxation lever/mobility correction, no-overlay public
slab8 passes3600 frames: peak depth218.26 micrometers, final14.04 micrometers;
peak joint anchor0.6933 mm and axis0.012918 rad. Maximum1917 cached points.
Two1200-frame repeats match every q/qd state and all74 hashes per frame.
Artifacts: /tmp/colibri_public_slab8_rebased_3600.json and
/tmp/colibri_public_slab8_rebased_hash_a, hash_b.

The subsequent isolated330-frame comparisons each exclude30 warmup frames:
public baseline22.2382 ms (44.97 FPS), no-overlay slab18.9351 ms (52.81 FPS),
slab with local static bilateral preparation18.3317 ms (54.55 FPS). The static
trial saves0.6034 ms/frame; all seven final body/contact arrays match byte for
byte against the dynamic-preparation slab control. All330-frame physical
checks pass. Source-equation equivalence is additionally covered by the root
agent's84 frozen block cases. No global collision admission change is involved.

Latest timing/state artifact: /tmp/colibri_static_slab8_rebased_isolated_330.json
and .state.npz. Static wrapper: local_studies.colibri.check_static_slab.


## Rotated-box coverage cause and minimal correction

The exact independent fixture is `test_conservative_mesh_candidates.py`.
Current source reproduced the original3-point support polygon and4.0525 mm
SAT overlap. A raw unreduced geometric-envelope query produced the same three
points, ruling out normal/spatial bins, ranking, export deduplication and
reducer capacity. No hash insertions failed.

The upstream cause is static endpoint ownership: an edge minimum at a corner
was discarded if another edge was designated that vertex's owner. The owner
edge need not select that corner as its own minimum. Disabling only ownership
metadata in a local control restored the missing positive-Y corners and passed
stopping, energy and all six momentum checks.

The minimal canonical correction removes the two ownership rejection clauses
in reduced/unreduced mesh kernels. It changes no SDF search tolerance, query
budget, gap, normal or solver equations. Existing geometric reduction handles
duplicate witnesses. Raw candidates increase3->8 and retained contacts3->5 in
this fixture. The original three-case test now passes: dormant contacts exert
zero impulse, the velocity reversal conserves momentum and loses contact
energy, and minimum SAT separation is+15.51 micrometers. This fixes the stated
fixture; it does not establish a general motion-envelope coverage guarantee.

Independent public `newton/tests/test_sdf_contact_coverage.py` fails before
and passes after on all six CPU BVH / CUDA texture cases across unreduced,
reduced and deterministic reduction paths. Both worktrees have the same
minimal patch and test; lint is clean. Logs:
`/tmp/sdf_contact_coverage_before.log`,
`/tmp/sdf_contact_coverage_after_final.log`,
`/tmp/dvi_sdf_contact_coverage_after.log`,
`/tmp/conservative_fixture_fixed.log`.

## Current bounded audit after endpoint ownership fix

The original conservative stopping fixture now passes after fixing mesh-edge endpoint ownership; six independent CPU/CUDA endpoint coverage variants also pass. A local public group4 geometric-admission run is being checked through 75 seconds against an endpoint-only public control, which has already passed that interval. This does not justify changing the default admission policy.

The current installer is deliberately not suitable as a canonical API: it intercepts factories globally, affects reduced buffered export paths, and leaves direct primitive writers on the original velocity filter. A clean future opt-in must thread one explicit candidate-admission policy through CollisionPipeline, NarrowPhase, mesh reduction, buffered export, and direct primitive writers. It must preserve the separate search gap versus authored physical gap, true witness separation, and current-step unilateral activation. The writer contract must describe geometric admission consistently; globally claiming exact initial-velocity admission would be wrong for this policy.

Required additional coverage before promotion includes zero-impulse receding candidates, reversal stopping with six-momentum/energy checks, deterministic output across direct/reused allocation, capacity/reclamation under dense separated geometry, and normal-family retention. The existing 255-per-pair reducer and global contact budget must remain enforced. The speed-derived capped search envelope is still not a proof against arbitrarily large acceleration or zero-speed impulses, so this option must not be advertised as full CCD.

The bounded post-endpoint-fix public group4 geometric-envelope run passed all4500frames (75s): peak fresh depth344.80micrometers, final14.10micrometers, peak joint anchor0.70463mm and axis0.012723rad. Maximum2753cached contacts remained below8192; authored shape gaps stayed byte-identical and search extension remained capped at5mm. Artifact: /tmp/colibri_canonical_geometric_endpoint4500.json. Timing was not isolated. Endpoint-only public control independently passed75s, so this result does not establish that the broader admission policy is necessary. The overlay remains local and public defaults are unchanged.

## Canonical opt-in validation

CollisionPipeline and NarrowPhase now accept speculative_contact_velocity_filter=False; the default remains True. The option covers geometric broadphase bounds, primitive admission, mesh inline reduction, buffered reduction, pipeline export and the standalone simple writer. Broadphase fills the same scalar shape-search envelope in every direction rather than rejecting initially receding pairs. It does not increase that capped scalar envelope or alter stored physical gaps.

All82 tests in the new generation/physical fixtures plus existing speculative-contact suite pass on CPU/CUDA. An expanded three-test admission rerun also passes, including the standalone NarrowPhase simple writer. Dormant receding candidates have zero impulses; paired reversal stops within the original10micrometer penetration bound with all six momenta and non-increasing contact energy. The default filtered control still misses those initially receding contacts, demonstrating the causal difference.

Frozen source8147 canonical deterministic replay: default filtered reduced15 points, zero transported penetrators, minimum+549.8micrometers; geometric reduced39 points includes one missing-flank witness transported to-348.4micrometers. Geometric raw726 points include54 such penetrators, minimum-423.5micrometers. Both repeats match exactly. These post distances transport the stored witnesses to the captured post pose; they are not a new signed-SDF query. Artifact: /tmp/colibri_public8147_admission_coverage.json. No default change or full-scene stability claim follows until the independent long canonical control finishes.
