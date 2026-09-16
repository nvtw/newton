# Frozen global joint solve conditioning

A CPU diagnostic assembles all188 joint rows from the physical post-relax
state in /tmp/colibri_support_relax330.npz. Contact impulses remain fixed.
With J the saved Jacobian, W the physical inverse mass/inertia and R the
original drive compliance, it solves (J W J^T + R) delta = -residual after
diagonal scaling. It uses every row with no mode truncation or regularizer.

The direct normal-matrix solve is rejected: maximum equation residual is
1.03444e-6 in mixed translational/angular units, exceeding the1e-8 reference
criterion, and the largest generalized impulse coefficient is9543 in mixed
SI units. Resulting body increments reach0.07642 m/s and3.02950 rad/s. This
is not a live trajectory or a contact-admissible correction.

To separate conditioning from cancellation while forming the normal matrix,
the diagnostic also factors each physical W block and examines the singular
values of the row-normalized factor [J sqrt(W), sqrt(R)]. Its largest value
is2.504; its smallest values are2.532e-5,8.932e-9,1.621e-10,4.444e-12,
2.731e-14 and1.509e-16. Thus the several almost-zero normal-matrix eigenvalues
cannot all be assumed to represent removable redundancy. Some correspond to
small, resolved physical response directions. A reliable augmented/primal
solve would need to preserve these directions and independently certify any
pure multiplier nullspace before reduction.

Artifacts: /tmp/colibri_global_joint_probe.npz stores J,W,A,rhs,delta,initial
velocity,compliance and normal-matrix spectrum; the matching .json stores
the factor conditioning and rejection evidence. Production is unchanged.

## Localization of small directions

A left-singular-vector audit of the unsquared, row-normalized factor locates
all six smallest directions in the wing linkage loops: Arc/Straight,
HummerBody/Straight and GearedSpinner/Arc joints on both sides. The drive
coefficients are almost absent (norm1.24e-10 for the8.93e-9 direction).
This points to closed-loop conditioning rather than singular drive compliance.
The audit does not establish that those small directions can be discarded.
The complete weighted joint participation and RHS projections are in
/tmp/colibri_joint_small_modes.json.

## Superseded wrong-sign physical audit

This section used the opposite correction sign and is retained as a rejected
control. Its contact/energy values do not describe the intended correction.
The corrected results are below.

The subsequent unsquared high-precision reference satisfies the equations
without removing resolved modes, but the independent physical audit rejects
its use as a correction. Linear momentum plus prescribed reaction changes
by less than3e-19 N s; angular momentum changes by up to3.64068e-4 N m s.
The stored positions are COM-in-world (solver_kernels.py state conversion),
so this is not a body-origin versus COM convention error.

Among70 eligible non-speculative relaxation points, the maximum normal
residual increases from0.0375384 to0.0445590 m/s; tangential residual changes
from0.0154903 to0.0132060 m/s with contact impulses held fixed. Kinetic energy
increases3.84271e-4 J. The midpoint impulse-work identity holds to4.88e-19 J,
but this algebraic identity does not establish admissible motor/contact work.

Reproduce: uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python
-m local_studies.colibri.audit_global_joint_response. Report:
/tmp/colibri_global_joint_physics_audit.json. The large cancelling dual
reactions may amplify small Jacobian torque defects; their cause is under
independent investigation. No production change was made.

## Torque defect diagnosis completed

Independent rowwise audit exactly reproduces the rejected angular momentum
change from the saved Jacobian's linear-row torque coefficients. Force pairs
and angular-only torque pairs cancel exactly. Linear-row equivalent lever
imbalance is at most11.41 nm. This is consistent with independently rounded
FP32 COM levers despite the existing shared-midpoint construction.

The native accumulated impulses weight the same defect to only6.83e-13 N m s;
the high-precision oracle's huge cancelling reactions amplify it to3.64e-4.
Thus the global correction is rejected, but the audit does not identify a
missing midpoint implementation or a significant native-step momentum error.
See GLOBAL_JOINT_TORQUE_DEFECT.md and /tmp/colibri_joint_torque_defect.json.

## Corrected sign and geometry-consistent reference

An independent gate now evaluates positive_native_residual + J*delta_v +
R*delta_lambda before any physical audit. The old wrong-sign control fails
with residual2.76331977; the previous audit lacked this check and exited zero.
Correctly signed original and reformed references pass at2.22e-16 and4.23e-16.

Reforming linear rows at the same inferred shared pivots in higher precision
changes J by at most5.70e-9, removes three independently rounded lever
artifacts (exact dual nullity1→4), and retains the remaining resolved modes.
This is explicitly a geometry-consistent changed operator, not bit identity
with the saved FP32 Jacobian.

| Correctly signed reference | Max angular momentum change [N m s] | Delta KE [J] | Max contact normal residual [m/s] | Max tangent residual [m/s] |
|---|---:|---:|---:|---:|
| Before correction | — | — | 0.0375384 | 0.0154903 |
| Original stored geometry | 3.64068e-4 | -1.243806e-4 | 0.0518598 | 0.0271791 |
| Reformed shared-point geometry | 1.36e-20 | -1.192092e-4 | 0.0518981 | 0.0258386 |

The reformed correction preserves linear momentum to1.63e-19 N s and the
midpoint-work identity to5.96e-19 J. Neither joint-only correction is accepted
because held contact residuals worsen. A coupled reference may reuse the
consistent geometry, but no live solver change follows from these results.

Reports: /tmp/colibri_global_joint_corrected_physics.json and
/tmp/colibri_global_joint_reformed_physics.json. audit_global_joint_response.py
accepts explicit --operator, --correction and --output paths.
