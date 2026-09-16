# General patch friction acceptance

The patch representation must preserve the original contact forces and their
moments, not just the sum of friction capacities. For fixed normal impulses,
each contact retains its own Coulomb disk `|t_i| <= mu_i * lambda_n_i`.
Equal and opposite impulses act at the same world point. No extra torsional
capacity is assigned to a point contact.

The independent reference suite `test_friction_wrench_generality.py` passes
four tests covering 1, 2, 4, 127, 128, 129 and 513 original points (up to 1,539
after subdivision), arbitrary normals and mixed friction coefficients:

- Splitting a point's load among colocated points preserves the wrench set.
- Rigid coordinate changes transform force and torque and preserve work.
- A single point cannot resist pure rotation about its own normal.
- A raised horizontal force requires the correct normal-load transfer; frozen
  equal support loads leave an unbalanced torque even with feasible friction.

These are offline reference tests, not evidence that the live solver has met
the requirements. Existing reference tests also reject equal-load two-anchor
compression that overstates a particular yaw capacity by five times.

## Implementation boundary

The proposed small planar sticking solve is a conditional acceleration. Its
acceptance requires checking every original point cone and the resulting
velocity, wrench and coupled normal/joint/drive equations. A failed proposal
means UNKNOWN and uses the original general contact solve. It must not be
interpreted as proven sliding or used to erase material history. Nonplanar,
rank-deficient and unsupported cases retain their physical modes through the
fallback. Contact count must not cause silent truncation.

The fixed-normal-load reference alone cannot certify a live coupled solve.
Tangential impulses can change normal velocities, normal loads and joint
reactions. Unchanged biased point targets can also be incompatible with a
three-degree-of-freedom rigid slip field.

## Current evidence

Projecting only the numerical recovery targets onto a compatible field gave
2.345 micrometers/second late drift versus 2.446 for the corrected native
baseline in the ten-second two-body test. This is not a drift fix.

The literal two-anchor diagnostic achieved much lower creep, but its force
capacity approximation is not acceptable as the general physical model.
The source-coefficient sixty-second run measured 0.0165 micrometers/second
over seconds 10–60 and about 27 FPS for two bodies. These figures do not
describe a validated general implementation or the full Colibri sculpture.

Removing observation-only kernel launches from that diagnostic produced
26.943 FPS versus 26.952 FPS, with byte-identical pose, velocity and time
histories across 600 frames. This shows no meaningful speed gain from those
launches; solver ledger writes were deliberately retained. Artifacts:
`/tmp/colibri_physx_patch_observer_timing600.*`.

## Coupled normal/friction follow-up

A captured final-velocity-pass state admits a full normal-and-friction sticking
contact solution under the original circular cones and native joint/drive
response. This preserves the existing hard-joint residual; it does not remove
that residual. The live absolute-joint gate later exposed this limitation.
The initial offline optimization served only as a feasibility oracle. A
six-component FP32 weighted-wrench proposal also passed the independent frozen
physical audit: normal residual 7.28e-10 m/s, tangent residual 2.83e-10 m/s,
positive normal impulses and all point cones interior. Held speculative rows
retain their impulses and positive predictive slack; they are not claimed to
have been re-solved for complementarity during a pass that skips them.

This combines exact point wrenches into six coordinates; it does not replace
original contacts with two force anchors. It is still a sufficient proposal,
with rejection falling back to the original solve. Unchanged biased friction
targets remain incompatible in this snapshot, so this evidence applies only
to the unbiased final phase.

`test_full_wrench_physics.py` independently exercises the actual FP32 proposal:
normal-load transfer, a 400:1 body pair, arbitrary normals/nonplanar points,
decisive members at indices 128 and 512, and individual-cone rejection even
when the requested resultant wrench fits. Five CPU tests pass. Native GPU
response and live trajectory integration remain separate acceptance gates.

## First live control and joint residual finding

The first full-wrench final-pass test rejected all 1,200 proposals and retained
byte-identical poses, velocities and times versus the corrected native baseline.
It therefore provides no creep improvement. The rejection reasons include an
existing absolute hard-joint velocity error that the homogeneous contact response
preserves. The independent snapshot audit checked its increment, not the full
absolute hard-joint equation; those are different acceptance claims.

`direct_equality.py` skips unbiased solves below a scaled residual threshold of
1e-4. The unbiased hard-joint target is zero. A separate zero-threshold control
is testing this quality loss; the new contact acceptance tolerance stays strict.

Separately, zeroing only the biased tangent recovery targets increased five-to-ten
second drift to 3.96653 micrometers/second versus 2.44593 baseline, at 20.08 FPS.
It is rejected as a fix. Artifact: `/tmp/colibri_native_zero_tangent_recovery600.*`.
