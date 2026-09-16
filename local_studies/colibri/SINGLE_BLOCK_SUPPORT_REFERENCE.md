# Single-block support reference

Local implementation: `single_block_support_reference.py`.
Artifacts: `/tmp/single_block_support_reference.json`, per-phase NPZs,
and `/tmp/single_block_support_momentum.json`.

## Actual matrices

The biased two-body case has 52 base/ground points plus 15 Frame/ground
points: 67 points, 201 Coulomb rows, twelve body velocity components,
five hard hinge rows, and one compliant drive row. The actual drive
coefficient 6281.44016238 is retained.

The base normal Jacobian uses only vertical translation and two tilt
rotations, so its rank is three. Only 28 base normal rows have native
compliance. Thus its 52-row normal operator has rank at most 31, and its
156-row friction operator has rank at most 34. Adding diagonal terms to
enable full Cholesky would change the equations.

The five hard hinge rows are independent (singular values 1.369 to 1.464).
Joint-constrained mobility therefore has rank seven without a numerical
rank cutoff. The six-row hinge/drive Schur matrix is SPD, minimum
eigenvalue 46.22, and eligible for Cholesky.

## Implemented native CUDA reference

One 128-thread block factors that six-row system, solves all 202 biased
right-hand sides, and reconstructs the complete twelve-component response.
The relaxation case uses 88 right-hand sides for 29 eligible contacts;
38 speculative points retain their native captured impulses. No point is
discarded by geometric reduction.

Maximum original linear equation residual is 4.55e-13 and physical response
difference from NumPy is 8.38e-13. No copy averaging, arbitrary regularization,
or rank truncation occurs.

The kernel also executes a fixed sliding-face solve selected by the accepted
CPU relaxation reference. All 29 contact laws, including inactive points,
are checked: natural-map residual 1.03e-17, joint/drive residual 4.16e-17,
CPU velocity difference 2.15e-16. Energy change -1.68166e-8 J matches midpoint
work within 6.62e-24 J. Actual stored levers plus ground reactions give zero
linear momentum error and 9.19e-13 N m s angular error.

## Limitations and next gate

The selected face is diagnostic, not an online mode-selection algorithm.
The full biased 67-point outer solve remains rejected at residual
0.00390765. No live stationarity or speed claim follows.

A practical block needs opening/sticking/sliding outer iterations and
certified impulse redistribution in redundant directions. Fixed sliding
directions generally produce a nonsymmetric matrix; Coulomb is not thereby
an SPD quadratic program. Blocks sharing Frame or another dynamic body
still need coupled iteration or scheduling. Color by dynamic-body ownership,
not individual point.

Do not promote the reference-selected face or failed biased outer solve.
The reusable validated piece is the small physical response setup.


## Online biased outer solve: subsequent accepted frozen gate

The earlier rejected 201-variable solve is preserved. A new local CPU outer
solver, `solve_support_online_continuation.py`, obtains activity without an
oracle: solve the convex frictionless normal problem, polish that seed, then
solve at actual friction with nonnegative normal impulses and dynamically
selected contact activity. Opening points remain explicit zero and all 67
contact equations are checked before accepting. Activity thresholds are
branch choices, not operator rank truncation or deletion of physical modes.

The simplest accepted control uses `--direct --gpu-response`: the response
matrix and affine baseline come from the native single-block kernel, then
the CPU outer uses 14 normal seed iterations, four normal polish evaluations,
and 37 nonlinear evaluations at actual friction. No intermediate friction
stages are needed; a separate homotopy control reaches the same response.
Final all-point natural residual is 1.30e-14. Actual original-equation audits
check normal complementarity, both tangent components and friction disk,
maximum dissipation, joint compliance, full impulse response, ground
reactions, and energy/work. See
`/tmp/colibri_support_online_direct_gpu_response_physical.json`.

Eleven points are loaded at the root: ten sliding and one sticking. The
corresponding 33-variable Newton matrix has condition number about 54 and
is nonsymmetric; use pivoted LU or QR rather than forcing Cholesky.

This remains a frozen CPU-outer diagnostic, not a complete GPU contact solver.
The selected points follow from the equations and current iterate; no saved
accepted solution is used as input. All 67 points stay in acceptance checks.
The previous failed unbounded solve carried negative normal impulses and
unneeded open-point variables; the physical normal seed plus activity
selection provides a materially better starting problem.

Crucially, the accepted biased root does not keep the base still:
base velocity is about (-0.750, 4.412, 17.545) mm/s. Delta energy is
+0.21377 mJ, with +0.22417 mJ from hard-joint position recovery and negative
contact normal/tangent and drive delta work. This is not unexplained energy
creation, but it prevents a stationarity claim or immediate live promotion.
Stored FP32 lever-arm momentum residual is 1.11e-10 N m s; it is reported,
not projected away. Coupled live recovery, neighboring states, and
production-suitable outer globalization remain necessary gates.


## Short live reference gate

`two_body_coupled_live.py` first replays 330 native total-normal frames and
requires exact q/qd bytes against the captured two-body reference. It then
runs eager native integration with contact generation at 120 Hz and thirty
substeps per interval. Every substep rebuilds actual joint geometry, inertia,
bias targets, contact response, and the single-block GPU factor. CPU contact
mode selection uses the online method above. Total impulses replace their
previous values once; corrected physical velocities are broadcast to every
copy before subsequent native phases. Warm history is retained by the
native replay and contact matcher.

The first run exposed a local empty-contact array-shape bug after thirty
successful biased phases. The corrected runner handles a joint-only
relaxation phase with zero eligible contacts. This is required when the
base lifts and its speculative contacts remain inactive.

The corrected two-frame run is `/tmp/colibri_two_body_coupled_live2_fixed.json`
and matching NPZ/candidate fingerprint JSON. All 120 biased plus four relax
phases pass: maximum original contact residual 1.41e-14, joint residual
2.25e-15, ideal-correction momentum/reaction residual 1.05e-10 N m s, and
midpoint-work mismatch 6.08e-20 J. These mathematical gates occur before
final FP32 storage; they do not establish bitwise conservation after casting.
Reference runtime for the corrected continuation is 1.336 seconds, not a
production speed result. The isolated frozen outer cost measured earlier
was 71 ms; many later live phases have no loaded contacts and are cheaper.

Recovery target magnitude decreases from .0870 to .0003146 (about 277x).
The base rises .415 mm; vertical COM speed peaks at 22.7 mm/s and becomes
-1.985 mm/s after 33.3 ms. Horizontal COM speed is still about 4.1 mm/s
while airborne. All four final relaxation phases have zero eligible
contact rows. This demonstrates decay of initial joint recovery error,
not settled base stationarity. A longer reference through recontact,
with explicit FP32 storage checks, is the next meaningful gate.

No production source, source geometry, damping, anchoring, friction,
collision frequency, or drive settings changed.

## Longer reference and nonlinear transition limits

The 60-frame attempt did not complete. The first run stopped after 13
collision intervals on a two-loaded-point friction transition; adding a
nonsymmetric Newton/Armijo fallback fixes that saved operator. The next
run, `/tmp/colibri_two_body_coupled_live60_newton.json`, stopped after
33 intervals (0.275 simulated seconds), with 1033 accepted phases. It
therefore establishes touchdown but does not establish settled stationarity.

These runs now independently audit actual FP32 stored velocities and
impulses, in addition to the ideal correction. Accepted phases have maximum
stored contact residual 8.92e-10, joint residual 1.27e-8, momentum/reaction
residual 4.99e-10, and midpoint-work discrepancy 1.19e-11 J. Native
integration momentum discrepancy is at most 2.92e-10. No final stationary
base claim follows from those local gates.

The second rejected operator has 17 loaded points; the two sticking
points have a smallest physical tangent response singular value of
2.30e-9, which is retained. A generic friction-boundary search along the
weak physical direction, followed by nonsymmetric polishing, reaches
all-point residual 2.26e-9 without rank truncation or added compliance.
Disabling this fallback reproduces residual 1.60955e-6. Four CPU tests
(including empty-contact and both transition regressions) pass:
`uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python
-m unittest local_studies.colibri.test_coupled_support_transition
local_studies.colibri.test_coupled_support_empty -v`.

This is not a production solver proposal. The saved hard phase takes
0.366 CPU seconds: 21 normal-seed iterations, 86 least-squares evaluations,
eight Newton steps with backtracking, and 52 hybrid evaluations. The
report is `/tmp/coupled_support_boundary_regression.json`. Counts of
least-squares evaluations in older live reports omit Newton line probes
and hybrid work and must not be interpreted as total operator evaluations.
A small, robust GPU outer-work bound is not demonstrated. The single-block
linear response remains useful; nonlinear contact globalization is the
unresolved production requirement. No further GPU live run was launched
while the friction-history correction has priority.
