# Reduced Colibri physics audit

## Acceptance status

The source-correct reduced scene is not physically stable. Prefixes 1–14 passed staged checks; stage 15 is the first confirmed fresh-contact penetration failure. Acceptance uses gravity 1 m/s², authored 32-sided cylinders and SDF resolutions, free base, source contact offsets with 1 mm fallback, no damping, SOR 1, and detection at 120 Hz.

Accepted original-USD PhysX final poses evaluated using Newton fresh geometry have maximum penetration 54.31 µm (initial 79.94 µm). Reduced Colibri failures of 1–2 mm are material.

| Stage-15 forest variant | First failure |
| --- | --- |
| 10 internal steps × 8 sweeps, current speculative gap | Frame 4: 1.709 mm gear penetration |
| 30 × 1, relaxation only after last internal step | Frame 10: nonfinite; prior peak penetration 0.833 mm |
| 10 × 8, relaxation only after last internal step | Frame 3: 1.402 mm penetration |
| 10 × 8, tangent positional bias removed diagnostically | Frame 3: 1.477 mm penetration |

Unique reports: /tmp/colibri_reduced_forest15_*_20260915.json. None supports full-scene stability or FPS claims.

## Confirmed fixes

- Exclude kinematic articulation trees from ownership and preserve their exported coordinates.
- Kinematic endpoints have zero inverse mobility and receive no impulse.
- Compute internal contact mobility and velocity in generalized coordinates; retain genuinely small positive mobility.
- Project normal positional recovery through constrained mobility; retain physical velocity effective mass.
- Refresh response after integration before relaxation; avoid applying warm starts twice.
- Solve normal first, include its tangential velocity effect, then solve the full tangent mobility block.
- Minimize tangent kinetic energy within the Coulomb disk; radial clamping an anisotropic unconstrained solution is incorrect.
- Use current witness separation during internal steps. A stale initial mesh gap formerly allowed a cube with 0.5 mm clearance to penetrate 0.5 mm despite ten internal updates.
- Apply paired impulses at a common point to conserve angular momentum.

Regressions cover energy, linear/angular momentum, sustained self-contact with an internal motor, kinematic immobility, near-blocked contacts, and closing speculative mesh contact. The motor fixture clears body forces each captured step.

## Ordinary rigid contacts and performance

The same uncoupled friction update affected ordinary bodies without joints. An actual solver regression increased energy from 0.75 J to 0.75037265 J. The coupled metric fix passes energy and momentum checks.

FP32 is used only for well-conditioned mobility; FP64 retains small positive eigenvalues. Normalization and RHS use selected precision. Safeguarded Newton accepts either convergence side; previous-impulse KKT data only initializes its search. Expanded tests retain original energy/KKT tolerances across rotations, scales, conditioning boundaries and tiny mobility.

Actual Kapla root counters show mean sliding iterations 5.524 → 2.895 with the coherent initial guess. Histograms: /tmp/kapla_friction_root_histograms.json.

Initial all-FP64 fix: 213.8 ms/frame versus 23.53 ms legacy on 11,341-body Kapla. Final optimized isolated timing is 33.6504 ms/frame (29.72 FPS), about 43% slower than legacy, with 246,882 contacts. Log: /tmp/kapla_rigid_metric_final_isolated.log. This is a Kapla result, not Colibri.

## Outstanding regressions

- The earlier full Kapla settling regression is resolved by the material-anchor reset described below; its unchanged test now passes with coupled friction and positional recovery enabled.
- Packed versus forced-serial self-contact bitwise velocity equality differs after fresh-response correction. Precision alignment needs audit; assertions were not loosened.
- Forest is optional and does not make full Colibri stable. It packs physical block-diagonal tree mobility and avoids repeated cross-tree traversals; closure/contact convergence remains unresolved.

No commits made by this agent.

## Sticky friction findings

Removed a dimensionally invalid normal-load multiplier from tangent positional recovery. Uniformly scaling both mass and normal load formerly changed the same 10 µm recovery velocity from 0.000808 to 0.0016 m/s; the fixed constant ERP gives 0.0008 m/s for both. The dedicated actual-prepare regression passes. Kapla settling still exceeds its unchanged threshold (median RMS 0.007867 m/s vs 0.007).

The broken-anchor branch reads the same already sticky collision witnesses twice. Clearing saturated tangent impulses therefore leaves the broken positional drift active. A local fail-before actual-prepare test clears the impulse but retains 0.0008 m/s bias. Fresh witnesses remain in collision sorting scratch, while matcher history has already saved sticky witnesses before solving; editing live contacts alone cannot persist the reset.

A local two-material-anchor prototype preserves collision witnesses, starts both friction anchors at the common world contact point, and transports them in their respective body frames. Four CPU geometry checks pass: rotated reset, global frame objectivity, endpoint symmetry, and recovery of only motion after a break. Production integration now uses existing rigid-unused persistent rows 6–11, initializes unmatched references at the common contact midpoint, copies matched history independently of impulse warm-start, and scatters resets from colored solve order back to canonical rows. Separated contacts reset their reference instead of accumulating a spring while airborne.

Two actual regression tests fail before and pass after: broken-reference recovery and captured colored scatter → history snapshot → reversed matched IDs. Further checks pass for rotated world frames, subsequent motion after break, 100 ordinary contact updates conserving P/L, six maximal/direct conservation fixtures, and the unchanged full 11,341-body Kapla jitter test. Constant ERP and correct metric friction remain enabled; damping remains zero.

## Separate full-scene route

The parent and Schur agent's fused bilateral D6/ordinary-contact route has passed the full source-correct scene. That result is separate from reduced coordinates. After ordinary current-gap and load-multiplier fixes, 15 internal steps at 120 Hz passed 180 frames (maximum 265.5 µm, final 67.5 µm). Preparation optimizations are being measured by those agents; this document makes no reduced stability claim.

After anchor integration, the primary matched full-scene run (30 internal temporal steps per 120 Hz collision update, 300 frames) passes: max penetration 208.47 µm, final 41.93 µm, mean 61.079 ms / 16.37 FPS. Before anchors: max257.9 µm, final38.8 µm,61.85 ms. Report: /tmp/colibri_anchor_candidate_matched30x1_300.json. The process loaded solver_phoenx_kernels dba43d6 and bilateral_joint4ebbe0f before the subsequent mass-splitting callback prototype edits.
