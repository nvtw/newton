# Replace repeated tree solves with a grouped contact solve

## Decision

Replace the **contact solve inside each biased temporal substep**, retaining all
physical contact points. Do not add another final-only pass. The existing
23.9%-accepted final correction stops admissible frozen states but worsens late
motion: 2.6905 µm/s versus 2.2589 µm/s with direct relaxation skipping disabled.
The biased substeps have already integrated their displacement before that pass.

The promising compression is the exact sum of point wrenches into six coordinates
per body, **not compression of force capacity into two anchors**. Individual
normal loads, friction cones and application points remain authoritative.

## Where the current work goes

These are source-derived counts, not measured kernel-time attribution. The public
example reports 60 frames/s, two 120-Hz contact generations per frame, 30 temporal
substeps per generation, one biased sweep per substep, and one final velocity sweep
per generation: **60 biased + 2 unbiased contact sweeps per displayed frame**.

At a frozen 67-point two-body state:

- `refresh_maximal_contact_mobility_kernel` uses only lane zero, visits each point
  serially, and computes six pair cross-mobility entries. The measured 60 refreshes
  give 24,120 scalar cross-mobility evaluations per frame.
- `iterate_maximal_contact_runs_kernel` visits up to 4,154 point rows and evaluates
  three relative velocities per visit. Speculative rows still require traversal;
  their unbiased impulse update is skipped.
- Each nonzero point correction invokes a complete articulation response and
  writes body velocities plus compliant-drive momentum. Tree traversals and
  block synchronizations repeat for every point. A two-body tree occupies only
  two body lanes in a 64-lane block; the point update itself is lane-zero work.
- Contact response uses small cached matrices, but repeatedly materializes their
  effect through the whole tree. The resulting cost is not explained by the
  number of physical body degrees of freedom.

`maximal_contact_gs.py` contains the serial refresh, point loop and
`_apply_accumulated_impulse`; `maximal_contact_response.py` contains the repeated
upward/downward response. The warmed profiler now confirms the dominant costs below. Removing optional observer launches changed the hybrid
26.95 FPS result negligibly, so observer overhead is not the main opportunity.

## Replacement at fixed geometry

For a complete small connected component, retain the native physical response
operator Wc, including the hard joints and finite compliant drive. For two bodies
this is a 12-coordinate body operator; a certified reduced representation has the
six root coordinates plus the hinge coordinate. Do not infer rank by discarding
small eigenvalues of squared mobility.

Keep every point's original Jacobian C_i and its three impulse coordinates.
Accumulate body wrenches q = sum(C_i^T lambda_i), then evaluate v = v_free + Wc q.
A change to one point can update the small cached component velocity directly.
It does not need a fresh upward/downward tree traversal and global body scatter.
A grounded body's full 6x6 self-response already exists in
`response.mobility[articulation, lane]`; cross-body blocks are also required when
several bodies in the component have contacts.

The first implementation should reproduce the existing sequential point solve
inside this cache, then do the necessary local convergence work before one paired
scatter per component. It may perform more local algebra sweeps while removing
thousands of tree propagations and synchronizations. Count that work explicitly;
this is an algorithm replacement, not a claim of equal iteration cost or a
promised FPS multiplier. Do not form a dense 3N-by-3N contact matrix.

## The grouped equations must include normals and sliding

For the native soft-normal branch, the fixed-point normal equation has the form

    g_n = C_n v + b_n + gamma_n lambda_n

with the **actual existing** softness/PD coefficients. Complementarity requires
lambda_n >= 0, g_n >= 0, lambda_n*g_n = 0. Speculative and final-relax rows retain
their distinct hard/held policy. For sticking, the original tangential gradient
is zero and each point remains inside its own static disk; sliding retains the
native non-associated Coulomb law and distinct dynamic coefficient.

The weighted six-wrench proposal is a cheap sufficient hard-normal sticking path,
not the biased soft-contact solution. In the biased phase, normal load cannot be
freely redistributed merely because its resultant balances: each compliant row
has its own normal equation. Eliminate compliant active normal rows using their
actual gamma, or solve their coupled active equations with the small body
operator. Keep hard speculative rows explicitly. A bounded active-face/Newton
update must check the original normal/tangent equations and every point cone;
UNKNOWN retains the original solve, never invents a static threshold or clipping.

Our prior biased target audit also found incompatible material recovery targets.
A faster exact solve cannot be assumed to cure inconsistent targets. Coherent
history needs a separate physically justified treatment, especially for new
rolling contacts. Likewise the native joint-recovery velocity is stripped before
biased contacts and restored before integration: the intended contact residual
must be specified against the velocity actually integrated. Any RHS/order change
must be explicit and separately validated, not hidden inside performance work.

## Implementation and acceptance boundary

1. Gather geometry, original point metadata and conditioned component response
   once per required geometry refresh. Preserve source material parameters.
2. Run the local coupled normal/friction solve in FP32, with scaled small systems
   and explicit failure flags. No artificial compliance or physical mode cutoff.
3. Validate all current normal equations, predictive slack/held rows, point cones,
   stick/slip equations, hard targets and compliant-drive equation. Validate
   actual rounded scatter, not just the independent FP64 oracle.
4. Commit total-to-increment impulses once, including native drive impulse
   accounting and equal/opposite common-point wrenches. Rejection preserves body,
   impulse and history bytes. Avoid double-counting the point rows replaced.
5. Compare late integrated displacement, force/torque and work budgets, and full
   source/trajectory provenance before measuring full-scene throughput.

Bounded scope should be topology/size-based complete components, never a Colibri
body name or ground-only production special case. Larger components use the
existing ordered block solve until a validated replacement is available. A
component boundary that omits incident contacts or joint reactions cannot be
claimed globally converged.

## Warmed measured profile

Root profile `/tmp/colibri_no_skip_warm_profile_stats.csv`, 30 displayed frames:

| Kernel | Calls/frame | Mean call | GPU time share |
| --- | ---: | ---: | ---: |
| conditioned contact iterate | 62 | 443 µs | 57.9% |
| contact mobility refresh | 60 | 122 µs | 15.5% |
| conditioned contact warm start | 60 | 52.6 µs | 6.6% |

These three consume about 80% of GPU time. Direct equality solves are only about
1.7%; collision is not the dominant cost in this profile. Warm start also invokes
the tree response per nonzero point, so grouped wrench accumulation should replace
that path as part of the same architecture. It cannot simply discard warm impulses.

Measured baseline is approximately48.8 ms/frame. Removing all three costs is not a
realizable zero-cost replacement or an FPS forecast; it identifies the correct
macro-level target. Pointwise cross-mobility refresh should be replaced by cached
small component response and point maps, not retained alongside the grouped solve.

The matched isolated PhysX two-body timer measured 201–213 FPS (about 4.7–5.0
ms/frame), versus 20.49 FPS here: roughly a tenfold gap. Even an impossible
zero-cost replacement of these three kernels leaves roughly 10 ms/frame of
remaining work. Therefore a successful contact replacement is necessary but is
not by itself evidence that the complete PhysX performance target is met. The
full 37-body all-awake PhysX control measured 119.67 FPS (8.356 ms/frame); compare
full scenes separately from the two-body diagnostic.

The first cached implementation is in `cached_component_live.py`. Its initial
dense independent-body response failed the independent angular-momentum and
hard-joint precision gate, despite small trajectory differences. The replacement
is being revised to use shared root and generalized coordinates with native
kinematic reconstruction. That failed gate is not waived for speed.
