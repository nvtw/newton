# Colibri stationary-base requirement

The user clarified that Colibri is a stationary sculpture: its base must not
move relative to the ground. Assembly-relative escape checks do not establish
this requirement. The prior 300-second trajectory passes its recorded joint,
penetration, and escape checks but FAILS the intended stationary-sculpture
behavior. Its 98.6414 FPS is a performance measurement of that failing physical
configuration, not a fully accepted Colibri result.

## Direct current-trajectory measurement

Input: `/tmp/colibri_cooperative_prepare_velocity1_long18000.npz`.
Output: `/tmp/colibri_current300_base_drift.json`.
FrameGround translation is measured against its first saved frame, rather
than the pre-step initial pose. Rotation is the quaternion geodesic angle
`2*acos(abs(dot(normalize(q), normalize(q0))))`, not an Euler yaw estimate.

| Time | Horizontal base displacement | Orientation change |
| --- | --- | --- |
| 1 s | 2.021 mm | 0.016230 rad |
| 10 s | 14.044 mm | 0.009339 rad |
| 30 s | 44.005 mm | 0.070720 rad |
| 60 s | 88.716 mm | 0.158028 rad |
| 120 s | 179.093 mm | 0.348329 rad |
| 300 s | 437.326 mm | 0.859425 rad (49.24 degrees) |

Base height changes only about 0.66 micrometers between first and last saved
frames. This is sustained horizontal sliding and rotation, not a falling
assembly. The earlier 0.40 m figure was dynamic assembly COM displacement.

## Priority and acceptance

Standing user constraint: speed or FPS improvements must never sacrifice
simulation quality, behavior, or correctness. Reject an optimization that
violates this constraint, even if it improves timing. A faster trajectory
that fails base stationarity is not an accepted optimization.

Verify the composed USD base rigid/kinematic/fixed-joint intent and actual
ground support geometry/materials/filters. Then record native per-substep
ground/kinematic reactions, FrameGround twist, and support friction residuals.
Separate physical external momentum transfer from internal momentum defects
and finite-iteration contact error. Keep SI units, source offsets, SOR 1,
and no global damping. Do not change source anchoring merely to conceal
a friction or solver defect. No assertion threshold is loosened to accept
this trajectory. The next physical acceptance must explicitly include
absolute base stationarity relative to the ground.

## User-confirmed support model

The user explicitly confirmed that ground friction must hold the sculpture
in place. FrameGround remains dynamic; introducing a fixed joint is not a
solution to this requirement.

## Existing controlled iteration comparison

The one- and four-final-velocity-iteration runs have identical recorded source
and asset SHA256 maps, both at 30 temporal substeps per 120 Hz collision update,
without the local search floor. At 60 seconds, FrameGround horizontal drift
is 88.716 mm versus 85.781 mm; at 240 seconds it is 354.349 mm versus
343.441 mm. Rotation at 240 seconds is 0.698440 versus 0.567866 rad.
Additional final velocity iterations modestly reduce creep but do not resolve
it. This does not exclude convergence errors in the biased pose-advancing
sweeps; the final velocity relaxation occurs after those integrations.

Evidence: `/tmp/colibri_base_velocity_iterations_comparison.json`, using
`/tmp/colibri_spatial_priority_velocity1_long18000.npz` and
`/tmp/colibri_spatial_priority_velocity4_clean14920.npz` with their recorded
`.source.json` manifests. No new simulation or settings change was made.

## Biased-iteration intervention

A new local runner changes only `solver_iterations`; the base stays dynamic,
friction/source offsets unchanged, 30 substeps per 120 Hz collision update,
SOR 1, one final velocity iteration, no global damping.

Ten-second runs give maximum absolute horizontal FrameGround displacement
from the pre-step initial pose:
- Two biased iterations: 1.686506 mm, final rotation 0.0125425 rad.
- Four biased iterations: 0.811037 mm, final rotation 0.00922774 rad.

Both retain unchanged source/asset hashes and pass the earlier joint/depth
screens, but neither is accepted as a stationary-base solution. The trend
strongly localizes sensitivity to the pose-advancing biased solve rather
than the final velocity correction. More solve work is a diagnostic, not a
matched-budget speed improvement.

Reproduce with `check_base_biased_iterations --iterations 2 --frames 600
--output /tmp/colibri_base_biased2_600` from this worktree using the usual
`uv run --no-project ... -m local_studies.colibri.` module prefix. Use 4 for
the second control. Outputs include `.npz`, `.json`, and `.base.json` with
full source/asset hashes and base-motion measurements.

## Confirmed coupled-support friction-load defect

The native last-substep capture at frame 330 is byte-identical to the full
330-frame reference trajectory. In its biased solve, loaded base support
points 21/23/24/25/26/27 have normal impulses
(10.185, 39.173, 19.907, 33.616, 21.947, 6.562) micro-N-s. The current
`_friction_normal_lambda` subtracts isolated-row recovery estimates
(40.707, 42.535, 43.272, 40.832, 40.458, 38.164) micro-N-s. Every result
is clamped to zero, eliminating friction capacity while the base slides.
Final relaxation supplies friction, but the biased velocity has already
advanced the pose. Copy averaging also reopens the friction equation: at
point 27 final relaxed copy slip is 45.87 micrometers/s, versus 825.61
micrometers/s after averaging, despite an interior friction ratio.

The row-local recovery subtraction is not a valid coupled decomposition.
An independently constructed two-support model with SPD normal response
A=[[1.01,0.99],[0.99,1.01]], gravity velocity 0.001 m/s, and uniform
recovery target 0.02 m/s exposes the same defect even at full convergence.
With actual default soft-contact coefficients, each total normal impulse
is 0.01018168046 N-s; its physical gravity component is 0.00048484193 N-s
and coupled recovery component 0.00969683853 N-s. The current isolated-row
estimate is about 0.0186475 N-s, larger than the total, so both friction
loads become zero. This is independent of finite-iteration residual.

Native phase accounting finds no corresponding internal momentum leak in
this snapshot: unaccounted linear impulse <=3.37e-10 N-s, angular impulse
<=4.37e-11 N-m-s, and net phase work-balance error <=3.84e-11 J. Ground
reactions are included; Flower contributes zero in this captured substep.
These are phase-local checks, not an all-time conservation proof.

Artifacts: `/tmp/colibri_support_phases330.audit.json`, its capture and
trajectory companions, and `/tmp/colibri_two_support_load_separation.json`.
The old row-local subtraction is invalid. A replacement must define a
consistent contact law: either explicitly separate physical and pseudo
recovery channels, or use the actual applied normal impulse in the existing
single-velocity stabilized Coulomb law. See the formulation distinction below.

## Formulation distinction and rejected recovery-free control

The current biased normal impulse is applied to real body velocities, not a
separate pseudo-velocity channel. In that formulation, using its total
accumulated normal impulse in the Coulomb cone is internally consistent;
zero friction from pure recovery is not a universal requirement. Earlier
notes that categorically excluded this option were too restrictive.
A two-channel formulation instead requires independently computed physical
normal loads and separately accounted geometric corrections. The two-support
regression's unbiased-load gate specifically tests that latter interpretation.
Both formulations invalidate the current isolated-row subtraction.

Local PhysX source corroborates the single-channel convention:
`physx/source/lowleveldynamics/src/DyTGSContactPrep.cpp` lines1530–1571
apply the biased normal update to actual velocities and accumulate it;
lines1643–1654 use that total for static/dynamic friction bounds. This is
a formulation reference, not proof of PhoenX conservation or accuracy.
The candidate must retain paired common-point impulses, dissipative
tangential work, bounded recovery, correct sliding/sticking, and cross-scene
regressions. Total work can include normal recovery and must be accounted.

A causal control disabling only negative numerical normal recovery (positive
speculative targets and physical PD terms retained) failed after 0.15 seconds:
HummerBody/Cylinder30–CamFollowerBody contact penetration reached1.092mm,
exceeding the unchanged1mm limit. It is rejected, not extended or promoted.
Artifacts: `/tmp/colibri_no_negative_recovery330.json` and companions.
The next local experiment retains recovery and uses total actual normal
impulse for the friction cone. No production physics changes yet.

## Local total-applied-normal candidate: improvement and failure

`total_normal_friction.py` replaces only the load helper, process-locally,
with `max(lambda_n,0)`. Recovery, friction coefficients, paired impulse
application, 30 temporal substeps, one biased iteration and one final
velocity iteration stay unchanged. No production file is modified.

The first330frame run passes the earlier geometry/joint checks at97.95FPS.
Maximum initial-pose-relative base translation is1.969mm versus7.685mm
for the unchanged330frame reference. Four focused energy/momentum/normal-first/
contact-bias tests pass; a test-loader typo in the first combined command
was corrected separately and its failed log retained.

The60second extension passes3601 geometry/joint checks at95.74FPS:
peakpenetration0.326085mm, anchor0.621586mm, axis0.0152093rad. However,
absolute base drift reaches9.49274mm and rotation0.284733rad (16.314degrees),
versus roughly88.7mm/9.05degrees for the old formulation. Translation improves
strongly while rotation worsens. This is NOT an accepted stationary-base
solution and does not justify a300second extension or production promotion.
The next phase audit must quantify friction torque and copy averaging under
the corrected candidate itself.

Artifacts: `/tmp/colibri_total_normal330_base_comparison.json`,
`/tmp/colibri_total_normal3600.json`, `.npz`, and `.candidate.json`.

## Matched PhysX reference also moves

The saved matched at-rest/no-global-damping PhysX repeat has13.7655mm
horizontal FrameGround displacement and1.18997degrees orientation change
from its saved post-first-step pose to its roughly5.5second final pose.
The PhysX harness stores positions in centimeters; this measurement applies
the same0.01 conversion as the harness. Only endpoint poses are available.
This benchmark does not establish that PhysX holds the sculpture stationary
in the matched modified setup; it does not excuse PhoenX creep or change
the user's friction-supported stationarity requirement.
Evidence: `/tmp/colibri_physx_matched_base_motion.json`.

## Ordinary coloring controls (correctness first)

The local total-normal friction candidate was held fixed for both controls.
Temporal settings remain 30 substeps per 120 Hz collision update, one biased
iteration and one final velocity iteration, SOR 1, and no global damping.

- Sequential group width 256 removed copy averaging (194 active colors,
  one copy per participating node at failure). It failed at 0.4333 s with
  1.709 mm gear penetration. It did not reach the timing warmup boundary.
- Ordinary coloring with cap 8 and splitting only overflow failed at
  1.1833 s: Frame/CamWheelBottom anchor separation reached 2.341 mm.
  Its 46.18 FPS measures a failing configuration and is not accepted.
  Successful saved poses already reached 1.615 mm base displacement.

Evidence: /tmp/colibri_overflow_initial_controls.json and the referenced
trajectory artifacts. Saved histories exclude the failing post-step pose.
These failures do not establish a mass-splitting implementation bug or
validate the grouped production configuration. The next control raises
the ordinary color cap to the currently supported maximum, 63.

The cap-63 control also failed: at 0.2833 s Frame/CamWheelTail axis
misalignment was 0.041932. Its captured graph had 593 rows: 347 in
63 ordinary colors and 246 in overflow. The largest body copy count was
98. The unchanged-production-friction paired control failed even earlier,
at 0.15 s with 1.032 mm gear/cylinder penetration. Neither reached timing
warmup. Artifacts: /tmp/colibri_total_normal_ordinary63_120.json and
/tmp/colibri_original_friction_ordinary63_120.json, with ownership sidecars.

Source audit clarifies that current overflow-only topology still scales
ordinary-row inverse mass by the total number of copies when a body also
participates in overflow. Regular rows use the shared partition-zero copy,
and the whole sweep is averaged afterward. This preserves impulse
accounting but does not give the regular pass full physical-mass coupling.
The intended alternative needs a real phase boundary: solve ordinary rows
on physical bodies, broadcast their resulting velocities to all overflow
copies, then solve and average overflow. Simply changing head count caches
would be incorrect. This is the next implementation direction; not yet a
validated fix or a claim of a momentum leak in the old implementation.
