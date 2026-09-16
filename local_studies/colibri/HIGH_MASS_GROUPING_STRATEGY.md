# High-mass convergence versus support-star grouping

CPU-only audit; no production edits or GPU runs.

## Different bottlenecks

The 400:1 fixture has a static ground, a 1 kg lower box and a 400 kg upper
box. The benchmark explicitly disables mass splitting. Its frozen operator
has two contact columns: ground/lower and lower/upper. The frame245/substep1
relaxation includes three ground points and five upper-interface points, with
12 dynamic velocity DOF. Thus it already has one physical velocity per body.
Combining only the static support manifold into one copy cannot fix this case.

Colibri's saved ground snapshot instead has 52 candidate points and three
FrameGround copies. Several loaded, non-speculative points have interior
friction impulses but nonzero slip in the final averaged state. Grouping
the support star can remove this specific averaging dilution. It does not
prove that coupling to the rest of the mechanism will converge in one sweep.

## Quantified cross-interface mode

For pure vertical translation, the exact normal mobility is
[[1,-1],[-1,1+1/400]]. Gauss-Seidel retains 400/401 of the slow error each sweep:
98.02% after 8 sweeps and 85.23% after 64. Approximately 1845 sweeps are needed
for a hundredfold reduction in that idealized mode.

The actual frozen normal matrix gives a 0.997589 linear iteration spectral
radius even when each entire body-pair block is solved exactly. This is a
cross-interface mode, not merely redundant points within one manifold.
The computation uses pseudoinverse only to analyze the exactly redundant
pair block; it is not a proposed mode truncation or production solver.

The full 24-row response has 12 physical velocity modes and 12 redundant
impulse modes. Its smallest positive eigenvalue is approximately 0.0003027;
the normal-only matrix also has a genuine 6.34e-6 mode. Removing such small
modes would alter physical response and is not justified.

## Actionable complementary strategy

A small connected contact-component solve spanning BOTH interfaces can
eliminate this slow propagation mode while preserving physical masses and
J-transpose impulses. The existing independent non-associated Coulomb oracle
already proves a valid solution exists for this difficult frozen state:
normal residual4.75e-10 m/s, friction residual3.34e-9 m/s, zero cone violation,
and negative physical contact work(-2.231 J). The prior root failure was
active-mode identification, not physical infeasibility.

A production experiment should therefore target bounded connected components
with a small number of dynamic DOF, not a high-mass-only special case or merely
one manifold. Preserve actual normal complementarity and the friction disk;
an associated cone QP changes the velocity law. Rank-aware impulse feasibility
must retain all original contacts and genuine physical modes. Semismooth
natural-map iterations with robust sticking/sliding mode handling are the
existing reference's promising route; exhaustive1680-mode enumeration and
SciPy are diagnostic tools, not a production implementation.

Start with the existing eight-point two-interface fixture and verify the
original Coulomb residual independently, reaction-balanced six-momentum and
nonpositive unbiased contact work. Then run the actual 400:1 8x8 load/RMS
benchmark. Do not claim the one-frame reference establishes settled accuracy:
the live extension still stopped safely on its next difficult mode, later
solved only offline. Biased recovery remains a separate problem.

For articulated Colibri, any analogous collective block must include the
relevant joint response or operate within its constrained mobility, rather
than treating a joint-connected dynamic body as immovable. Support-star
grouping is a useful inexpensive scheduling improvement first; connected
physical response is a complementary convergence strategy if residuals remain.

## Artifacts

- /tmp/high_mass_support_grouping_audit.json: reproducible frozen spectra.
- /tmp/high_mass_relax_245_1_mode_audit.json: independently accepted Coulomb state.
- /tmp/high_mass_relax_live_completed_audit.json: eight accepted solves in one frame.
- HIGH_MASS_QUALITY_AUDIT.md: previous matched quality and stopped-run limitations.

## Bounded GPU design informed by the completed ten-frame oracle

The updated oracle completes80/80 relaxation solves. Thirteen need discrete
mode search; three accept the previous mode first. This supports persistent
contact-mode guesses, but physical point matching must map them through
contact replacement/reordering. Reusing bare row indices may only be a guess;
it must never authorize a physical result without recomputing residuals.

An initial local prototype can limit eligible components by computational size
(e.g. at most12dynamicDOF and at most9contacts), leaving all other components
on the existing solver. This is strategy selection, not contact deletion.
For the actual9contact case, a27x27 FP64 response requires5832bytes; a second
dense factor buffer plus vectors remains suitable for one cooperative block.
The component owns all endpoints or applies its exact joint-constrained
mobility; splitting only one supporting interface leaves the slow mode.

Use the current natural-map analytic Jacobian and non-associated normal law.
Sliding-mode KKT matrices are generally nonsymmetric, so an SPD-only Cholesky
shortcut is inappropriate. A pivoted small QR/complete-pivot solve with
explicit redundant-impulse handling is a safer first diagnostic. A numerical
rank decision may identify redundant equations only if the discarded residual
and the full original physical map pass; never cut genuine low-mobility modes.

Candidate schedule:
1. Gather matched point modes, true J, M^-1 and current RHS into shared memory.
2. Test the previous active/stick/slip face and solve its coupled equations.
3. If invalid, run a bounded semismooth correction with residual globalization;
   use violated complementarity/cone conditions to prioritize neighboring
   faces. Keep deterministic alternate direction seeds for near-zero slip.
4. Evaluate every original point's normal, friction, cone and reaction/work
   conditions. Scatter J-transpose impulses only after acceptance; intermediate
   trial directions never touch live velocities.
5. Queue unresolved components for further bounded work, preserving the current
   valid solver result. Do not silently claim convergence when the budget ends.

The rare enumeration branches expose the main unresolved design issue:
a single local root can settle on the wrong active/sliding face even for six
contacts. Porting dense linear algebra alone will not reproduce the oracle's
robustness. First replay all80frozen operators and deliberately wrong mode
seeds, with the same independent acceptance, before any live GPU trajectory.
Then compare the400:1 load/RMS window at unchanged work; the present ten-frame
CPU settling result is evidence of physical solvability, not usable throughput.
