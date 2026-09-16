# Heavy-on-light convergence comparison

Both cases use the existing `bench_high_mass_ratio` recipe, SOR 1, one velocity
iteration after each substep, 240 settling frames and 120 sampled frames. The
single measured frame is only required by the runner; its FPS is not a valid
performance result. No solver files or physics defaults were changed for this
comparison.

HEAD is the existing isolated `f0073be5332b1caf155d54a448943f924d856ef7`
archive used for G1. Its fixture requires the same `add_shape_plane(xform=...)`
keyword compatibility migration already present in the current fixture. A
process-local wrapper applied only this API adapter.

| Version | Ratio | Substeps × iterations | Top RMS speed (m/s) | Bottom RMS speed (m/s) | Top load error | Minimum center separation (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HEAD | 100 | 8 × 8 | .004667 | .001856 | .006286% | .999563 |
| Current | 100 | 8 × 8 | .000009304 | .000007023 | .00009333% | .999574 |
| HEAD | 400 | 8 × 8 | .018278 | .028877 | 3.2707% | .998036 |
| Current | 400 | 8 × 8 | .153862 | .062448 | 9.8164% | .997078 |
| Current | 400 | 8 × 16 | .016367 | .012616 | .9420% | .998270 |
| Current | 400 | 40 × 32 | .00007923 | .0003112 | .002082% | .999932 |

All runs remain finite with the stack intact, but that boolean alone hides the
substantial unresolved oscillation. Current 100:1 improves; current 400:1 has a
material low-work convergence regression. Higher work approaches the supported
load and rest state, but does not erase that low-work regression.

Artifacts:

- `/tmp/colibri_shared_high_mass_quality.json` (parent's current 8×8 run)
- `/tmp/colibri_head_high_mass_quality.json`
- `/tmp/colibri_current_high_mass_convergence.json`
- `/tmp/run_high_mass_compat.py`

## CPU source attribution

The recipe uses box/plane primitives, sticky matching, point friction with μ=.5,
no joints, no mass splitting, no contact chunks, and no parallel prepare. The
new final-only relaxation option is not selected.

The removed `load_boost` multiplied only tangent position-recovery bias by a
load-dependent factor between 1 and 4; it never multiplied the normal bias.
Restoring it would change friction recovery, not repair physical normal mobility.

The positive generation-gap clamp removal is inactive here: its shape predicate
accepts only mesh/heightfield/tetrahedron, excluding both box and plane.

Relevant changes are material friction-anchor persistence/reset; corrected
common-point impulse levers; and normal-first projection with the full tangent
mobility metric. The latter changes finite-iteration trajectories while retaining
physical masses and dissipative Coulomb friction. For witnesses separated only
along the normal, moving their levers to the common midpoint leaves normal
angular Jacobians unchanged, but changes tangent response.

A two-normal translational model already explains poor conditioning: with bottom
mass 1 and top mass M, its mobility is [[1,-1],[-1,1+1/M]]. Serial GS contracts by
M/(M+1) per sweep (400/401 for the heavy case). This explains why extra work can
help; it does not attribute the new-versus-HEAD difference by itself.

Next bounded diagnostic proposed to parent: compare HEAD/current 400:1 8×8 with
zero friction only as a causal control, then freeze full-friction rows if the
normal-only paths match. No frictionless result would count as acceptance of the
original scene, and no load-dependent correction is to be restored.

## Additional causal controls and independent correction

Matched μ=0 400:1 controls both ejected the light lower body and lost the stack;
these cannot serve as settled acceptance results. Current preparation/history
with the HEAD projection substituted process-locally gave top RMS speed .0729 m/s
and roughly 6.6% load error at 8×8. This is between full HEAD and full current,
so projection changes alone do not explain the entire trajectory difference.
No old projection or load boost was restored in production.

A frozen current full-friction state at frame 240 also gave decreasing kinetic
energy with both projection equations over eight unbiased physical sweeps.
That snapshot does not establish convergence of the biased temporal simulation.
Artifacts: `/tmp/high_mass_mu0_{head,current}.json`,
`/tmp/high_mass_current_legacy_projection.json`, and
`/tmp/high_mass_frozen_operator.{json,npz}`.

The source audit found an independent conservation defect: ordinary rigid
relaxation used COM-relative levers prepared before pose integration. After two
free bodies moved, opposing impulses therefore acted at different world points.
A new regression failed with angular-momentum error .05000028 kg·m²/s in one
relaxation update, for both split-enabled and unsplit worlds. It passes after a
central once-per-substep rebase preserving the prepared world impulse point and
refreshing the corresponding effective masses. Rotated anisotropic bodies,
independently calculated mobility, linear/angular momentum, and unchanged
collision witnesses/material friction anchors are checked. The single pair does
not exercise overflow copies. Seven existing contact history/projection/chunk
regressions also pass.

This correction does **not** resolve the high-mass low-work convergence issue:
post-fix 400:1 8×8 top RMS speed is .142421 m/s and supported-load error 10.2953%.
The 100:1 case has .001193 m/s top RMS speed and load error 4.46e-8 (fraction).
See `/tmp/high_mass_rebased_quality.json`. The conservation fix has independent
mathematical evidence; the heavy-stack convergence regression remains open.

## Tangential-recovery and phase trace

Removing only tangential positional recovery, while retaining μ=.5 and the
current physical contact projection, worsened 400:1 8×8 to .2871 m/s top RMS
speed and approximately 11.9% supported-load error. This diagnostic is not an
accepted configuration (`/tmp/high_mass_no_tangent_recovery.json`).

A passive eager trace of frames 240–255 records all 128 biased solve phases and
128 relaxation phases. Normal fixed-point residuals include the actual existing
soft-contact coefficients (.9417003989 mass coefficient and .0582995452 impulse
coefficient at idt=480). After eight sweeps, the maximum residual is 1.120 m/s
and the median per-phase maximum is .104 m/s. The worst phase has only 9.23e-6
m/s tangent bias, versus 1.30 m/s normal bias. Relaxation residual maxima reach
.204 m/s (median .0507 m/s). This supports finite-work coupled normal
nonconvergence rather than tangent recovery as the primary failure mechanism.

All relaxation phases reduced kinetic energy in this trace (total -.948 J).
Biased solves had net -16.21 J kinetic-energy change, although some recovery
phases added energy (sum of positive changes +3.89 J). Per-row bias/impulse dot
products are only correlation diagnostics, not exact work attribution across
warm-start and projection phases. Material anchors changed 168 times during
prepare/solve and never during relaxation. Friction saturation was uncommon.
See `/tmp/high_mass_recovery_trace.json` and `trace_high_mass_recovery.py`.

## Packed-row regression discovered and corrected

The first ordinary-rebase wiring gathered canonical contact state into packed
solve rows before relaxation. For colored-row Kapla, that replaced live solved
impulses with stale canonical history. The packed-history regression failed
before correction (11 seeded impulses replaced by zero). Rebase now operates
on the active solve container and packed header map directly, without gathering.
Both conservation and packed-history tests pass.

The matched 11,341-body Kapla check (6×10, 30 warmup + 180 frames) recovered:
mean/max drift .000748/.01385 m versus prior .000769/.01899 m; mean/max speed
.00624/.04271 m/s versus prior .01053/.07302 m/s. Maximum angular speed is .626
versus .621 rad/s. These concurrent correctness runs do not establish FPS.
Artifact: `/tmp/kapla_rebased_packed_fixed.json`.


## Frozen collective Coulomb reference

`reference_high_mass_connected_normals.py` uses corrected frame244/substep0,
eight points and twelve dynamic DOF, retaining the physical J M^-1 J^T,
soft normal diagonal, recovery targets and friction .5. Exact normal active-set
enumeration reduces residual 1.120 -> 4e-15 m/s; eight extra scalar sweeps reach
.478 m/s. Matrix condition is 126.7. J-transpose scatter balances momentum plus
static-support reaction to 8.9e-16 linear and 1.2e-7 angular; common world points
differ by 3e-8 m. This does not justify a live normal-only replacement: alternating
normal blocks and scalar tangents still leaves .0300 m/s after64 sweeps.

A24-variable non-associated Coulomb natural-map solve reaches1.10e-4 m/s residual,
normal residual6.75e-9 m/s and no friction-disk violation. Normal closure excludes
friction multipliers; this is not an associated cone QP. Kinetic energy decreases
9.0422 ->8.8045 J and reaction-balanced angular error is1.4e-7. Strict friction KKT
acceptance FAILS despite optimizer termination. The seven-active sticking matrix
has six null directions and recovery RHS projection1.09e-4 m/s into that nullspace.
All127 sliding-face refinements reject; an additional all-eight-normal search
rejects255 faces. Local nonlinear root-search failure is NOT proof of Coulomb
infeasibility; incompatible sticking targets do not prove incompatible geometry.
No production edits or GPU runs. Artifacts:
`/tmp/high_mass_connected_reference.{json,npz}` and
`/tmp/high_mass_connected_all_normal_reference.{json,npz}`.


### Recovery isolation and Jacobian verification

The same frozen operator with only tangential recovery removed still stalls at
1.07e-4 m/s. Removing all recovery, while retaining friction .5 and the original
soft normal diagonal, passes the complete non-associated Coulomb KKT check:
normal residual 4.86e-16 m/s, friction residual 7.20e-16 m/s, disk violation
2.78e-17 Ns. Six active contacts slide with positive multipliers. The analytic
Jacobian agrees with centered finite differences to 4.04e-11 relative error.
Reconstructed physical free energy is 9.31634 J and final energy 9.07914 J;
contact work is -0.23720 J. The earlier snapshot-before energy is already after
its biased solve and must not be treated as the free-energy baseline.
Linear/static-reaction balance is 9.22e-16; angular balance is 1.04e-7.

Four bounded continuation controls (zero tangent recovery) pass at normal-bias
scales .01 and .1, with residuals 1.42e-15 and 7.78e-13 m/s, but reject at .3 and
1.0. These local mode searches do not prove infeasibility at stronger recovery;
they show that simply seeding from the unbiased solution does not solve the
biased system. No live physics was changed. Exact reproduction uses
`--recovery none`, or `--recovery normal-only`, with `--output` for separate
artifacts; `--seed` and `--normal-scale` reproduce the continuation controls.
Results: `/tmp/high_mass_coulomb_no_recovery.{json,npz}`,
`/tmp/high_mass_coulomb_no_tangent_recovery.{json,npz}` and
`/tmp/high_mass_coulomb_continuation_*`.


## Exact existing-relaxation oracle

`capture_high_mass_relax.py` captures frame 244/substep 0 immediately after
integration and the canonical force-point rebase. `--relax` in the reference
retains the existing positive-bias point skip, fixes those skipped impulses,
and uses zero normal compliance, exactly matching this phase. Seven of eight
points participate. Actual scalar output leaves .204194 m/s normal residual;
the collective reference reaches 2.55e-9 m/s normal and 1.82e-9 m/s friction
residual with no disk violation. Physical free energy 9.38167 J becomes4.73074 J
(scalar output8.98540 J). Linear reaction balance3.6e-15, angular1.59e-6;
the latter reflects original FP32 world-point mismatch at larger54.6 Ns loads.
No poses change. `/tmp/high_mass_relax_comparison.json` records both outputs.

A process-local one-frame oracle (`check_high_mass_collective_relax.py`) retains
all biased solve/integration steps and replaces only unbiased relaxation after
strict KKT acceptance. Redundant sticking impulses initially defeated the root
solver; solving sticking equations and finding a Coulomb-feasible distribution
in their nullspace fixes that particular mode. This is impulse feasibility,
not an associated cone velocity law. The next live retry accepts substeps0/1
at <=2.93e-9 KKT, but stops at substep2 (.0389 m/s) without applying the rejected
solution. Enumerating all normal faces finds no accepted all-sticking solution
for that third snapshot; sliding mode handling remains necessary. No completed
frame, reduced oscillation, or live convergence is claimed. All production
physics/defaults remain unchanged; these references are CPU diagnostics.
Artifacts `/tmp/high_mass_relax_live_*.{json,npz,log}` preserve the stopped run.


### Third relaxation snapshot solved offline

An explicit semismooth Jacobian for the unilateral normal projection and the
Euclidean tangent-disk projection resolves the previously rejected third
snapshot. This map has the same non-associated Coulomb fixed points; normal
closure still excludes tangential friction multipliers. The warm impulse seed
stalls, but a zero-impulse restart converges in seven evaluations. Independent
original-metric verification gives normal2.13e-10 m/s, friction2.67e-11 m/s,
and disk violation2.67e-16 Ns. Only two sliding contacts carry appreciable load
(6.67e-5 and3.11e-5 Ns); old large warm impulses must be released.
Physical contact work is -3.41e-8 J, with zero linear reaction imbalance and
2.38e-7 angular imbalance from original FP32 endpoint geometry.

`coulomb_semismooth.py` is now an offline fallback in the reference, and the
original independent KKT acceptance remains authoritative. Failed analytic
mode-trial diagnostics are explicitly named as trials, not final multipliers.
`/tmp/high_mass_relax_live_2_accepted.{json,npz}` records the accepted result.
The one-frame live retry has NOT been rerun yet; this establishes progress on
its third frozen snapshot, not completion or live stability.


### Complete one-frame oracle acceptance

The next bounded live retry completed frame244 at400:1 with all eight existing
biased8-sweep solves and position integrations unchanged. Each unbiased
relaxation uses the CPU oracle only after independent KKT acceptance. Another
least-squares mode stall at substep4 was resolved offline by the same analytic
semismooth Jacobian with an unconstrained hybrid root search; the projection
still enforces unilateral impulses, and final physical checks are unchanged.

All8 replacements pass: maximum FP64 KKT2.93e-9 m/s, applied FP32 normal
residual4.42e-9 m/s, FP32 disk rounding at most9.2e-8 Ns. Every relaxation has
negative physical contact work (sum -13.57 J); linear/static-reaction balance
<=1.1e-14 and angular balance<=1.59e-6, reflecting the original FP32 endpoint
mismatch. Poses are not changed by the replacements. Final light/heavy speeds
are .0681/.1176 m/s, compared with .3668/.2066 before the first replacement.
Final center-height separation is .997910 m; all state is finite.

This establishes accurate contact convergence across ONE complete frame, not
settling, load accuracy, longer-term robustness or usable speed. No production
changes. `/tmp/high_mass_relax_live_completed_audit.json` records all substeps;
`/tmp/high_mass_relax_live_final.npz` records the final state. The CPU SciPy oracle
is deliberately not a production dependency or performance proposal.


### Ten-frame extension rejected safely

The requested ten-frame extension completed frame244 and the first substep of
frame245, then rejected frame245/substep1 at .024776 m/s KKT residual without
applying the candidate. Nine preceding relaxation solves passed. The frame244
benchmark load measurements are1125 N top /1123 N ground versus3924/3934 N
expected; the transient is not settled and load accuracy is not established.
No ten-frame stability/convergence claim is made. Per-phase physical P/L/energy
is recorded in `/tmp/high_mass_relax_live_phases.json`; changes in P/L alone
include external ground reactions and are not labeled conservation errors.
Unique snapshot `/tmp/high_mass_relax_live_before_245_1.npz`, reference output
`/tmp/high_mass_relax_live_245_1.*`, and outcome
`/tmp/high_mass_relax_live_10_outcome.json` preserve the rejection. Production
is unchanged. A more robust non-associated mode search is still needed before
this CPU oracle can establish longer-term settling behavior.


### Second-frame rejection is a mode-identification failure

Frozen frame245/substep1 contains eight participating points; points4/7 have
exactly identical three-row physical Jacobians and RHS. Their aggregate cone
is mathematically identical at common friction .5; equal splitting of an
aggregate impulse preserves both individual cones and the original J-transpose
physical impulse. Aggregation alone does not solve the failed root, and the
bounded normal-face search finds no accepted all-sticking solution.

`reference_coulomb_modes.py` independently enumerates normal active sets and
stick/sliding-direction modes on this seven-aggregate problem. For fixed
sliding directions, impulses satisfy linear normal/sticking equations; a small
root solve enforces sliding direction alignment and nonnegative dissipation.
All original complementarity/cone conditions are checked after each candidate.
An accepted mode appears after1680 trials (~1.1 s CPU): aggregate normals
(2,3,4,5), direction-constrained contacts(2,3,4). Some of the latter have nearly
zero slip, so the mode labels do not imply substantial physical sliding.

Expansion to all eight original points gives normal4.75e-10 m/s, friction
3.34e-9 m/s, zero cone violation, and contact work -2.231 J. Reaction-balanced
linear error2.0e-15, angular8.84e-8. This identifies a failed active-mode guess,
not physical inconsistency. Accepted output/audit:
`/tmp/high_mass_relax_245_1_mode_solution.npz`,
`/tmp/high_mass_relax_245_1_mode_audit.json`. No additional live retry yet;
the ten-frame run remains incomplete and production is unchanged.

## Ten-frame physical relaxation continuation completed

The local generalized mode fallback now completes frames244-253: 80/80
unbiased relaxation replacements pass the independent original non-associated
Coulomb check. Normal KKT max8.759e-9m/s, friction KKT max2.987e-9m/s.
Applied FP32 normal residual max1.010e-8m/s; disk rounding max2.753e-7Ns.
Reaction-balanced linear error max7.11e-15 and angular max1.592e-6, reflecting
the original FP32 endpoint mismatch. Poses are unchanged by each replacement.

Every actual relaxation decreases kinetic energy (sum-24.508J, largest
change-0.004198J). Every reconstructed-free-velocity contact-work check is
also negative. The latter's sum is NOT temporal energy loss because cached
impulses are subtracted independently at each solve.

Final top/bottom speed is1.47e-13/1.31e-11m/s; final loads3924.00049/3933.81030N
match3924/3933.81N. Final center separation.998694m still reflects the untouched
biased integration/recovery. Ten frames is only1/6second; no long-run resting
accuracy or performance claim follows.

The helper used13 discrete-mode fallbacks among80 solves. Three succeeded on
the immediately previous mode's first trial. Worst accepted search used12576
trials;251_5 required a deterministic pi/2 change to its direction seed after
two unsuccessful seed families. This is bounded diagnostic enumeration,
not a GPU performance solution. Generic solver diagnostics alone consumed
about96CPU seconds, excluding failed preliminary solves and enumeration.
Existing biased8x8steps, physicalmasses, friction.5 and sourcegeometry remain
unchanged. No production solver edits.

Artifacts: /tmp/high_mass_mode_continuation_completed_audit.json,
high_mass_mode_continuation_live_{frames,summary,phases}.json and the per-substep
NPZs. Reproduce with check_high_mass_mode_continuation --frames10. The saved
245_1 physical-mode seed is only a solver guess; all outputs are independently
revalidated before scatter. Failed trials are never applied.

## Completed 120-sample window

The extension completes all 120 samples (frames 244-363), with all 960
relaxations independently accepted. Maximum normal/friction residuals are
9.30e-9 / 5.28e-9 m/s. Every actual relaxation decreases kinetic energy;
reaction-balanced linear/angular errors remain below 7.11e-15 / 1.59e-6.
FP32 application has maximum normal residual 1.01e-8 m/s and friction-disk
rounding of 2.75e-7 Ns.

| Window | Top RMS speed | Bottom RMS speed | Top mean-load error | Plane mean-load error |
| --- | ---: | ---: | ---: | ---: |
| All 120 samples | .021033 m/s | .011677 m/s | .02148% | .01887% |
| Last 60 samples | .008680 m/s | .007289 m/s | 1.4018% | 1.4017% |

The very small whole-window mean-load error includes cancellation of the
initial transient and later load excursions. The last-half values therefore
matter: this is improved convergence, not exact static equilibrium. Biased
solve/integration remains unchanged and continues to move the bodies between
the exact physical relaxation solves. Minimum center separation is .997791 m;
maximum top-body XY drift over the recorded substeps is .022792 m.

For context, the earlier current 400:1 8x8 baseline had top RMS .142421 m/s
and mean-load error 10.2953%. This comparison is diagnostic: the oracle starts
after the baseline has already developed motion and is not a new production
benchmark configuration. No existing acceptance tolerance was changed.

There were 147 discrete-mode fallbacks among 960 solves. Retained generic
reference calls alone cost about 615 CPU seconds, excluding failed preliminary
calls and discrete enumeration. This is unsuitable for production but now
provides 960 validated operator/solution fixtures for developing a smaller
robust solver. No solver runtime was changed.

Artifacts: /tmp/high_mass_window120_audit.json and
/tmp/high_mass_window120_live_{frames,summary,phases}.json. Per-substep NPZ
files preserve every frozen operator and accepted solution. Reproduction:

    python -m local_studies.colibri.check_high_mass_mode_continuation \
      --frames 120 --output-prefix /tmp/high_mass_window120_
