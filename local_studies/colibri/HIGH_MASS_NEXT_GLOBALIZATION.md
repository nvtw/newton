# Next bounded high-mass convergence experiment

## Evidence reviewed

This is a CPU-only proposal, not a new accepted solver. No production changes
or GPU runs were made. The ordinary400:1 unsplit8x8 benchmark remains unresolved:
post-rebase top RMS speed0.142421m/s, supported-load error10.2953%.
Static-support copy grouping cannot fix this unsplit island.

The expensive120-frame relaxation oracle proves substantial improvement is
possible, but is not a production strategy:960 accepted relaxations,147 discrete
mode fallbacks,615CPU seconds in retained generic solves plus enumeration.
Whole-window RMS0.021033m/s and last-half RMS0.008680m/s do not establish exact
rest; biased integration/recovery still moves the bodies.

Current geometry-consistent bounded replays are more relevant to an implementable
solver:
- Original:789/960 accepted;171failures all exhaust eight face visits.
- Certified objective-tie seed:797 accepted, gains32 but loses24.
- Original-first numerical proximal continuation:802 accepted, gains80 but loses67.
- The cheap natural path already accepts576inputs. Preserve that path unchanged.

Only six original failures include a response-null rejection. Broad failure is
therefore not solved by lowering a rank cutoff.300_2 is especially decisive:
the physical factor minimum singular value is0.01736; a physical root exists,
yet the non-oracle interface heuristic already selects its correct face and
still fails at residual1.40765 with a negative sliding multiplier.
More accurate factorization or choosing that face alone is insufficient.

Evidence:
- /tmp/high_mass_consistent960_summary.json
- /tmp/high_mass_consistent960_failure_classes.json
- /tmp/high_mass_tied960_summary.json
- /tmp/high_mass_adaptive960_summary.json
- /tmp/high_mass_300_2_diagnosis.json and _rank_diagnosis.json
- /tmp/high_mass_interface_move.json
- BOUNDED_ISLAND_COULOMB_PLAN.md

## Why the current globalizer stalls

The current augmented Newton unknowns include physical response, every impulse,
and one multiplier per sliding contact. Face merit can decrease while normal
impulses become negative, sliding multipliers become negative, or supposedly
sticking impulses leave their disks. The original-law audit correctly rejects
these roots. Natural-map Newton also approaches switching boundaries with
successive very small Armijo fractions; existing derivative checks do not
indicate a missing derivative.

There are two distinct difficulties: selecting a useful contact face and
reaching its physically admissible root.300_2 isolates the latter. Enlarging
the face queue, restoring load-dependent recovery, relaxing KKT acceptance,
or selecting the saved oracle root would avoid rather than solve that issue.
Fixed-response gauge redistribution alone is also insufficient:362_5 has a
certified incompatible sliding-direction equality at its rejected velocity.

## Proposed change: eliminate linear face equations before angular Newton

Keep the cheap natural solve. For a bounded fallback face obtained only from
current impulses/velocities and the existing deterministic mode queue, represent
each sliding impulse by

    lambda_t,k = -mu_k * lambda_n,k * d(theta_k)
    d(theta) = (cos(theta), sin(theta)).

Let lambda=T(theta)z, where z contains active normal impulses and sticking
tangent impulses. Let H select active normal and sticking tangent velocity
equations. With physical response factor F=J*M^(-1/2), solve

    [ I      -F^T*T ] [ y ] = [    0   ]
    [ H*F       0  ] [ z ]   [ -H*rhs ].

Then g=F*y+rhs. The nonlinear equations are only

    d_perp(theta_k)^T * g_t,k = 0.

At acceptance additionally require d(theta_k)^T*g_t,k>=0, nonnegative normal
impulses, inactive normal separation, all sticking disks, and the unchanged
original natural-map residual. Thus normal response remains non-associated
Coulomb; this is not an associated cone quadratic program.

Differentiate the linear equations using their existing factorization to
obtain angular Jacobians. Each angular trial re-solves the linear equations,
so the nonlinear iteration cannot spend its budget correcting arbitrary
normal/sticking equation errors or an unphysical independent sliding multiplier.
This is a different parametrization/globalization experiment, not evidence
that convergence is guaranteed.

The expensive reference_coulomb_modes_general.py already uses angular
parametrization, but its exhaustive face enumeration, SciPy hybrid solver,
and rcond=1e-11 solve on A[eq]*T are NOT acceptable production ingredients.
Reuse the mathematical parametrization only, with unsquared physical response,
bounded deterministic trials, and no oracle mode or accepted impulse input.

## Actual current-seed assembly

CPU assembly using only the existing non-oracle interface seeds gives:
-300_2: three angular unknowns,27x27 linear system; resolved tail0.003116.
-244_0: three angles,27x27; resolved tail0.001808.
-245_1: two angles,29x29; resolved tail0.002612.
-263_2: three angles,27x27; resolved tail0.002531.
-362_5: two angles,29x29; resolved tail0.005823.

The matrices also have six or nine numerically null singular values. These
values are observations, NOT permission to drop modes. Redundant sticking
point wrenches must be certified as pure impulse gauges using the shared
physical pivots/body-pair wrench map, full original equations, and unchanged
physical response. Preserve the support-line free twist in362_5. Any ambiguous
response-bearing null direction aborts the trial. If the selected impulse
representative violates sticking disks, only a certified zero-response
redistribution may repair it; otherwise reject the face. Minimum norm alone
does not prove disk feasibility.

Full spectra, current-seed angles, and input-derived modes:
 /tmp/high_mass_variable_projection_design_audit.json.
No root was solved or accepted in this assembly audit.

## Bounded reproducible experiment and stop criteria

First use the five cases above, including known successes as regression controls.
Inputs are the consistent operators and saved non-oracle interface states:
 /tmp/high_mass_consistent960_CASE.npz
 /tmp/high_mass_interface_move_CASE.npz.

Use only their A/J/W/rhs/initial/seed and the already recorded candidate modes;
do not read any accepted oracle solution. Initialize angles from the current
candidate tangential velocity. Zero-slip directions retain mapped prior contact
direction or a deterministic local basis, never a future accepted direction.

Keep the original maximum76factorization budget including the cheap12-step
natural path: every angular backtracking trial consumes the remaining64budget.
Derivative multiple-right-hand-side solves reuse each factor and are counted
separately. Do not hide extra factorizations inside line search. No extra
physics iterations, contact drops, altered mass/friction, recovery or compliance.

Require full original KKT<=1e-8, cone/normal checks, reaction-balanced P/L and
midpoint-work identity, nonpositive unbiased contact work, and unchanged input
hashes. Reject unresolved rank/gauge cases rather than flooring physical modes.
Intermediate numerical states are never scattered. A production fallback would
apply accepted J-transpose impulse changes once or run native relaxation,
never both.

If the five cases do not improve without losing existing successes, stop:
do not increase the face budget or run all960. If they pass, replay all960 and
report gains AND losses, worst costs, and all invariant checks. The seed itself
currently costs enumerated normal masks; that cost must be removed/replaced by
native mapped impulses or a bounded pivot seed before claiming practical GPU
viability. The new scheme is at most a candidate small-island fallback for
unbiased relaxation, not a proposed replacement for the biased contact law.
