# Small physical-space Coulomb solve — frozen reference

This is a CPU diagnostic, not production code or an accepted live stability result.
It preserves the assembled contact rows, all five hard hinge rows, the finite-gain
sixth joint row, normal compliance, friction coefficients and source targets.

## Equations

Exact joint condensation leaves seven physical velocity coordinates for the two
free bodies. With `P = G Gᵀ`, `F = C G`, `v = vbar + G u`, contact consistency is
`u = Fᵀ lambda`; contact gradient is `g = F u + rhs`.
The factor uses the nullspace of the five independent hard joint rows and a
7-by-7 finite-drive response factor. It does not truncate small contact modes.

For a proposed active contact face:

- Compliant normal: `lambda_n = -g_n/gamma`, checked nonnegative afterward.
- Hard normal: retain its multiplier and impose `g_n = 0`.
- Sliding tangent: `lambda_t = -mu*lambda_n*g_t/|g_t|`.
- Sticking tangent: retain its two multipliers and impose `g_t = 0`.
- Inactive contact: zero impulse, with its original normal inequality rechecked.

This is the non-associated Coulomb law, not an associated cone energy minimum.
Eliminating soft normals and sliding tangents leaves a small nonsymmetric Newton
system. The actual transition needs nine unknowns; accepted original biased and
relaxation faces need ten and eight respectively. The local helper refuses more
than 32 unknowns rather than silently introducing a large dense solve.

## Face selection without an oracle

`physical_face_active_set.solve(a)` starts with two original-order metric PGS
sweeps. It tries the current natural-map face and a coupled normal-complementarity
proposal with friction held fixed. The latter is only a numerical proposal.
Its present CPU implementation uses a bounded SLSQP normal solve, including the
original compliance. It still constructs a contact-normal matrix; replacing this
reference with a fast native implementation is separate work.

A negative trial normal multiplier triggers at most three contact-unloading
pivots. No omitted row disappears from acceptance: every original contact is
checked after the solve, and modes are reclassified on subsequent outer steps.
Each accepted impulse update lies in its normal/friction cones and decreases the
original all-row natural-map squared norm. Failed face solves are only proposal
sources; none are applied directly as physical impulses.

Bounds: two seed sweeps, six outer updates, 100 normal-QP iterations per proposal,
12 Newton steps and 20 backtracks per face, three unloading pivots, 20 final merit
backtracks. These are reference bounds, not a GPU performance claim.

## Measured frozen results

| Frozen operator | Seed / selection | Final original natural-map error |
| --- | --- | ---: |
| Original frame330 biased, 67 points | Two sweeps, two safeguarded outer updates | 3.33e-16 |
| Original frame330 relaxation, 29 points | Two sweeps, one safeguarded outer update | 4.88e-18 |
| Current live rejection after 0.1 s, 65 points | Two sweeps and a nine-unknown face solve | 1.59e-13 |

All three pass the original normal, cone, joint and physical scatter checks. For
the original two phases, joint error is at most 1.48e-15 and body impulse-response
error at most 1.74e-18. Original biased work/kinetic change is +2.137496e-4 J;
relaxation is -2.397904e-7 J. Equality of work and kinetic change is an accounting
check, not a claim that biased recovery conserves energy.

An earlier all-sticking proposal was incompatible with the biased targets. The
accepted biased solution actually has ten sliding contacts and one sticking
contact; the accepted relaxation solution has one sliding contact. A general
all-sticking shortcut would therefore change this transient's physics.

Artifacts:

- `/tmp/colibri_physical_face_active.json` and phase solution NPZ files.
- `/tmp/colibri_physical_face_active_without_pivots.json`: preserved failed relax control.
- `/tmp/colibri_physical_face_live_rejection.json`: two-step correction of failed 32-sweep output.
- `/tmp/colibri_physical_face_live_shortseeds.json`: zero/two/four-sweep seed controls.

The original accepted-face-only experiment is explicitly labeled
`reference_face_only` in `/tmp/colibri_physical_face_support.json`; it is not used
by the safeguarded active-set wrapper. Independent neighborhood and live checks
remain necessary. No production defaults or physics parameters were changed.
