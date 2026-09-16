# Tail contact coverage at exact authored offsets

## Reproduction

Source37-body maximal bilateral route,30 temporal steps per120Hz collision update,
SOR1, no ether damping, source SDF resolution and32-sided cylinders. Measured
PhysX per-shape contact offsets are applied without altering physical geometry.

The ordinary static-search run fails at frame297 with3.177954mm fresh
TailRack/Pinion penetration. Instrumented trace reproduces final q array-exact.
All298 actual per-frame colored endpoint audits show zero conflicts.

Artifacts:
- /tmp/colibri_tail_trace_30x1_exact.trace.npz:64 outer-step ring, generation and
  postsolve q/qd, cached shape IDs/witnesses/normals/impulses.
- /tmp/colibri_tail_trace_30x1_exact.trace.analysis.json: fresh pinion geometry.
- /tmp/colibri_tail_trace_30x1_exact.trace.rack.analysis.json: fresh all-rack geometry.

## Missed guide approach precedes pinion failure

| Generation step lacking new guide row | Next-query guide | Future witness gap before | Gap after8.33ms | Summed offset |
|---|---|---:|---:|---:|
|589|LeftStandoffB|408.84um|−179.44um|277.78um|
|590|TailMountRight|294.99um|−675.53um|277.78um|

The before value transports the newly generated material witness pair backward
using its future fixed normal. It is not a claim about exact previous closest
distance. At step591 guide impulse reaches0.004064Ns. At593 the pinion pair is
absent during the step and appears3.178mm deep afterward.

Cached and independent fresh guide penetration agree at the generation poses.
An initial message incorrectly compared pinion-only depth to global guide depth;
that interpretation was retracted. There is no demonstrated false sticky gap here.

## Cylinder33 loses an existing nearby feature during reduction

At saved generation592, explicit-pair raw detection produces11 points. One lies
on the opposite rack depth edge at269.61um gap. Standard reduction keeps4 points,
all on the other depth edge, and omits this point. In the next update the opposite
edge is deeply penetrating. Thus this instance is not missing broadphase pair
detection. Raw and reduced artifacts are /tmp/colibri_cylinder33_{raw,reduced}_592.npz.

The omitted point is initially separating at about 1.07 mm/s; motion reverses
inside the temporal interval. Predictive reduction at this same saved pose also
keeps four points and omits this edge. It is therefore not evidence that the
predictive selector directly rescues this particular feature.

For the earlier LeftStandoffB guide, generation-time contact velocity instead
predicts 1.111 mm travel and 702.50 um penetration. Earlier admission of that
approach is a plausible explanation for avoiding the later failure chain; the
full-run comparison alone does not isolate every intervening contact.

Existing rotating-leading-feature tests cover the general reducer behavior,
not this initially receding Cylinder33 feature.

## Existing predictive search path

CollisionPipeline already accepts speculative_contact_gap_max and collide(dt=...).
It expands search bounds using COM-correct shape-origin linear velocity and
angular speed times shape radius, then admits extra points with true signed-gap
and contact-point closing-velocity checks. Predictive reduction retains leading
features separately. Authored shape_gap and physical margins remain unchanged.
None remains the default.

Optional cap5mm, exact offsets,120Hz/30x1 passes300frames:
- max fresh depth271.20um, final66.75um;
- max observed cached points1868, capacity8192;
- max observed search extension5mm;
- authored gap arrays asserted unchanged everyframe.
Report /tmp/colibri_speculative_exact_5mm_300.json.
Timing was under correctness-job overlap and is not a benchmark.
The independent 1200-frame (20 simulated second) extension also passes:
max fresh depth 349.53 um, final 48.10 um, maximum 1868 cached points.
Report: /tmp/colibri_speculative_exact_5mm_1200.json.

The combined chunk-six, mass-split batch-two route passes 1200 frames with
max depth 156.1 um, final 60.9 um, and maximum 1860 cached points. Coloring,
point ownership and joint checks pass throughout. Report:
/tmp/colibri_chunks6_batch2_exact_predictive_30x1_1200.json.

The new physical two-sphere speculative-impact regression passes all six
momentum components at every step and non-increasing energy, with unchanged
authored gaps. Its zero-motion case produces no extra contacts or search
extension. Five existing rotating-leading-feature tests pass on CPU/CUDA.

## PhysX attribution

PhysX simulationcontroller/src/ScCCD.cpp:64–94 adds linear and angular motion to
contact distances only under eENABLE_SPECULATIVE_CCD. GPU contact preparation
also explicitly checks that flag. GPU TGS contact preparation uses invStepDt for
positive separation bias (contactConstraintBlockPrep.cuh:926–941), consistent with
a temporal contact solve; selecting TGS alone does not demonstrate automatic
velocity envelope inflation. A composed USD traversal found no authored
attributes containing 'ccd'. This Newton option is an explicitly selected
robustness feature, not a demonstrated explanation of the unchanged PhysX run.
