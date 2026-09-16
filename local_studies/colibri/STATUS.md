# Colibri validation status

## Current acceptance target

Source geometry and drives in SI units, dynamic FrameGround, separate kinematic Flower, SOR 1.0, no global or authored damping. Refresh narrow-phase contacts at120Hz. Current optimization is explicitly at30 internal substeps per collision update and one biased iteration; the lower-substep comparison is a separate study. Authored contact offsets:52 shapes at1mm, three hypocycloids at2mm, rest offsets zero; remaining offsets automatic in USD. Comparison fallback is1mm; uniform1cm remains an experiment, not a validated default.

## Latest validated optimization (2026-09-16)

The canonical cooperative joint RHS path passed the complete 18,000-frame
Colibri comparison: all ten saved arrays are byte-identical to the previous
300-second trajectory. Kapla's six arrays and G1's two 20-step controller
histories also match their respective baselines byte-for-byte. The runner
verified that the three affected production source files stayed unchanged.
Report: `/tmp/colibri_cooperative_canonical_validation.json`.

The long Colibri run measured 11.18856 ms/frame, or 89.377 FPS, excluding
rendering and physical audits. This approaches the clean matched PhysX
reference of 89.97–91.52 FPS; it does not establish equal physical accuracy.
The further cooperative forward-substitution optimization is now integrated
and validated: all four canonical cases pass with unchanged source hashes,
including all ten saved Colibri arrays over300 seconds. This run measured
11.082846 ms/frame (90.2295 FPS), p95 12.31738 ms, excluding rendering and
physical audits. Report: /tmp/colibri_cooperative_forward_canonical_validation.json. See COOPERATIVE_RHS_INTEGRATION_REVIEW.md.

Five-minute motion analysis finds no sampled crank stall, but mean crank
speed is -3.17343 rad/s versus the -3.49066 rad/s finite-gain target. Observed
gear rotation ratios remain consistent across the trajectory. These are
motion observations, not independent tooth-engagement or drive constitutive
validation. See LONG_GEAR_MOTION.md.

Neither solver meets the complete target. Free-assembly support drift,
production high-mass-ratio convergence, Kamino full-scene acceptance, and
final integration remain unresolved. The 400:1 CPU coupled-contact reference
now completes 120 samples/960 accepted relax solves; this does not fix the
production solver's approximately 0.142 m/s RMS at the original 8x8 budget.
Kamino's frozen split-position feasibility experiment is not promoted:
its explicit energy redistribution and unrefreshed contacts do not establish
a physically admissible live correction.

## Current unintegrated solver studies

These studies do not change the live 30-substep configuration or its FPS.

- Geometry-consistent joint tree elimination reduces the frozen global
  factor to a46x46 mass matrix and48x14 QR. Four saved poses retain the
  original joint equations and physical response. A dedicated QR prototype
  measures74.6us versus145.4us for the vendor QR component; geometry, mass
  assembly, contacts and scatter are excluded. See TREE_GPU_FACTOR_FEASIBILITY.md.
- The original biased closed-loop recovery targets can be incompatible
  with the exact joint nullspace. Do not drop those equations. A final
  unbiased relaxation experiment can retain the native biased path.
- Accurate joints alone do not solve contact convergence.64 pointwise
  sweeps still leave normal/tangent residuals0.01161/0.00642m/s. Solving the
  normals together followed by tangents also fails because of strong
  normal–tangent coupling. See TREE_CONTACT_SWEEPS64.md.
- The bounded, geometry-consistent high-mass contact prototype accepts
  789/960 saved inputs with unchanged physical gates;171 remain rejected.
  This is separate from the earlier expensive960-accepted external reference.
  The new method is not ready for GPU or live integration. Results:
  `/tmp/high_mass_consistent960_summary.json`.
- An independent storage audit of those789 accepted FP64 roots finds
  realized residuals up to6.08e-6m/s after ideal accumulation and FP32
  impulse/velocity storage. Conservation, work and friction-disk changes
  fit explicit quantization bounds. This does not weaken the FP64 root
  gate or validate actual native scatter. See CONTACT_FLOAT32_STORAGE_AUDIT.md.

## Earlier integration results (2026-09-16)

Neither solver satisfies the complete robustness target.

The public Phoenx example now defaults to geometric candidate admission,
group4/grid64,30 substeps per120Hz contact update, one biased iteration,
SOR1 and no global damping. The global CollisionPipeline default is unchanged;
`--velocity-filtered-candidates` selects the old example policy for diagnostics.

This configuration PASSES300 simulated seconds with the same penetration,
joint, finite-state, speed and tail-support limits, plus a base-relative0.5m
escape check appropriate to the free assembly. Peaks: penetration0.753578mm,
anchor0.938996mm, axis0.028822rad. Original absolute travel exceeds0.5m at
203.45s (peak0.65637m); this remains explicitly reported rather than erased.
All12206 prior pose/velocity/timestamp frames match byte-for-byte, including
the old stopping pose as the next continuation frame. Public default330 smoke
and isolated timing trajectories also exactly match this validated prefix.
Four policy/bounds regressions pass; rigid-motion invariance fails before fix.

The rigid-only coloring specialization is now integrated. Generic eight-node
rows remain supported, and the rigid policy scans only its two endpoint slots.
CPU/CUDA regression and generic multibody tests pass; integrated public330
matches all ten saved reference arrays exactly. Isolated prototype pairs:
76.982→78.632FPS and76.462→80.935FPS. Timing magnitude varies; a fresh isolated
integrated-default measurement remains pending. See RIGID_COLOR_ROWS.md.

Before this coloring specialization, isolated synchronized physics-step timing was77.3636FPS, mean12.92598ms,
p9514.10250ms,30 warmup and300 measured frames, excluding rendering/audits.
Artifact: `/tmp/colibri_geometric_default_isolated330.json`.
Five-minute result: `/tmp/colibri_public_geometric_relative18000.json`.

Support accuracy remains unresolved: three independently solved base copies
leave residual slip after averaging. A loaded interior-friction point changes
from0.05885mm/s before averaging to0.97207mm/s afterward; see
GROUND_COPY_RESIDUAL.md. The five-minute pass establishes bounded mechanism
operation for that trajectory, not accurate static support or all-scene solver
acceptance. The previous velocity-filtered mode failed tail joint checks at
124.1333s, before the geometric policy became the example default.

- Smaller groups2/3 are rejected: axis failure at55.55s and penetration failure
  at44.933s respectively. The example remains group4.
- Lower-substep group8 study:10 is the lowest tested one-minute passing count;
  9 fails. The default remains30 as requested. See SUBSTEP_STABILITY.md.
- Kamino DVI after the shared endpoint fix, intended joint recovery0.01,
  still fails stage15 atframe3:1.474mm penetration, unconverged residual0.02746,
  crank speed-0.455rad/s. The earlier pre-fix run failed at1.177mm penetration.
  It remains unstable and too slow; no full-scene acceptance claim.
- New multi-group conservation tests pass and fail when the required contact
  lever refresh is deliberately disabled. Exact scheduling/coloring changes
  preserve tested state arrays. The400:1 collective Coulomb reference subsequently completed a120-sample
  window; production convergence remains unresolved (see latest status above).

Current evidence: COLOR_GROUP_SIZE.md, CURRENT_PROFILE.md,
HIGH_MASS_QUALITY_AUDIT.md; Kamino recovery details are in the DVI worktree's
local_studies/colibri/KAMINO_RECOVERY.md.

## Historical early results

Neither solver has passed the current complete acceptance target. Earlier anchored Kamino36-body scene passed10 simulated seconds at20substeps/16iterations, about2.05 isolated FPS. Those results do not establish stability of the corrected free-base scene. Free-base full Kamino passed3seconds in some runs, but a40-substep run failed around5seconds and a batched partial-assembly run failed atstage23. More substeps alone are not a solution.

Phoenx maximal anchored full scene passed3seconds after contact ownership and normal correction fixes. The free-base full scene at20 contact refreshes/frame,1 internal substep,16iterations failed atframe146. At120Hz contact refresh,10 internal substeps, source offsets and1mm fallback it fails aroundframe3. Reduced mode still fails on the first hypocycloid stage. No usable robust FPS claim yet.

## Confirmed fixes under validation

- Single-world ordinary contact sweeps were processing contacts already owned by the joint Schur solve; both dispatch paths now respect ownership.
- Near-blocked contact penetration recovery is projected through the joint constraints; cached normal loads are released when a contact becomes blocked.
- Direct Schur cancellation is filtered relative to the large terms being subtracted.
- Free-tree contact factorization read unprepared legacy joint anchors; current pose/model joint geometry is now gathered directly.
- Reduced dynamics incorrectly integrated kinematic free-joint trees. Reduced and direct contact helpers also incorrectly included kinematic physical inverse mass and applied impulses to them. Regressions demonstrate these errors.
- Reduced internal contact row velocities now use the same generalized Jacobian used for mobility, avoiding subtraction noise near blocked directions.

## Historical contact fidelity experiments

Source uses SDF mesh collisions even for cylindrical shapes. An optional64-sided mesh-cylinder path now reproduces that representation. Distance-field padding covers the configured detection gap. Contact buffers are enlarged for mesh-cylinder manifolds. Kamino external120Hz contact path now retains separated speculative contacts; its previous default culled these, defeating early detection from larger gaps. Full validation pending.

## Workspace

Kamino: /tmp/newton-colibri-dvi on dev/dvi_next_round.
Phoenx: /tmp/newton-colibri-phoenx on twidmer/rooftop-optix-usd, with pending uncommitted merge of origin/main. No commits made. Preserve unrelated changes in the original workspace.
