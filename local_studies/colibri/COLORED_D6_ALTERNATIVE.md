# Colored local D6 alternative (experimental)

No canonical solver files are changed. The primary comparison is between:

1. Current joint-conditioned contact owner: each contact applies an articulated response.
2. `colored_d6_alternative`: existing full-coordinate bilateral block PGS and ordinary point contacts in the same color schedule.
3. `colored_d6_scalar`: FP32 prepared endpoint response rows, ordered scalar joint impulses with immediate endpoint velocity updates, followed by the ordinary contact schedule.

The third route adopts the GPU PhysX D6 *local row architecture*. It does not claim to reproduce every PhysX TGS detail. It retains Newton's original joint coordinates, hard-row bias, native implicit drive diagonal/reference, contact law, thirty substeps, and source masses. It does not introduce PhysX lever slop, drive approximations, mass splitting, damping, or over-relaxation. PhysX row preprocessing and body motion accumulation remain separate comparisons.

## Why the prior fast path failed

The isolated native implicit drive is already analytically correct. The historical frozen two-body contact operator demonstrates that later local contacts reopen the drive equation. In the new FP32 audit, the first biased joint solve leaves a drive residual of 2.31e-9; after the contact sweep it is 0.2223. At 32 complete sweeps the residual remains 0.1014 and the original contact natural-map residual is 0.01835. The corresponding relaxation values after 32 sweeps are 0.001315 and 0.0001994.

These are historical frozen equations, not current corrected friction trajectory measurements. The independent impulse-response velocity ledger differs by at most 3.31e-7 over the recorded biased sweeps and 1.20e-8 for relaxation. An earlier diagnostic with tuple-assignment aliases failed this ledger and was rejected; no production defect was inferred from it.

## Intended convergence experiment

First measure the current corrected live colored block and scalar routes with their actual callback ownership recorded. A fast trajectory is not accepted unless joint, contact, drift, and analytical drive checks pass. If ordinary sweeps fail, increasing the number of cheap local sweeps is a numerical work comparison, not a physical parameter change, but must be reported explicitly. Scalar row order alone is not expected to repair the demonstrated global coupling; prior joint-last and interleaved frozen variants also failed complete KKT checks.

For a substantial architectural improvement, the useful unit is a local joint/contact neighborhood that keeps endpoint velocities resident through multiple sweeps and writes them once. All original point cones/application sites and joint equations must remain. Coloring provides disjoint neighborhood ownership; arbitrary shared endpoints cannot be fused without an explicit conflict schedule. A two-body neighborhood is a bounded prototype, not yet a general multi-body implementation. This is compatible with the independent coupled physical-space work, but no feasibility gate may be replaced by an assumed all-stick face.

## Remaining gates

- Actual scalar callback analytic drive and internal momentum gate.
- Corrected ordinary contact callback provenance, graph execution, and current short trajectory.
- Matched timing scope and long drift/angle checks.
- General joint types, bounded drives, shared-body chains and multi-world support before production consideration.

## Current corrected live measurements

The actual CUDA scalar callback passed the closed two-rotor drive check:
relative angular velocity 0.006994935684 versus implicit oracle 0.006994934295
(error 1.39e-9 rad/s), exactly zero internal angular momentum in that fixture.
This proves the isolated drive row, not every contact/joint phase.

Matched 60-frame startup controls (30 warmup, same corrected friction/history):
block 94.45 FPS; scalar 120.65 FPS. Scalar had worse transient error: peak joint
anchor 58.95 micrometers versus block 77 nanometers; maximum base displacement
1.007 mm versus 0.1125 mm. Broad geometry checks passed, but neither startup
measurement establishes resting quality.

The block route completed 600 frames at 93.93 FPS, with the correct final hinge
angle 18.20732788 degrees. It nevertheless creeps 62.6724 micrometers/s over
5–10 seconds. Final hard joint residual is 1.17e-11 and drive residual 1.84e-10.
Therefore the current failure is not established by a final joint-equation
violation. No variant is promoted.

All six saved native row wrenches have zero net force. Their maximum world
angular defect per unit impulse is 5.59e-9 meters, consistent with FP32 shared
midpoint arithmetic; accumulated defect is at most 8.65e-13 Nms in the block
sample. An independent FP32 source/scatter audit of 200 rotated unequal-mass
cases with 58-micrometer anchor separation passes: maximum scatter momentum
error 1.93e-10 Ns and angular momentum error 2.78e-10 Nms. This latter test is an
offline source-equation reference, not an actual six-row kernel dispatch.

## Accepted current phase trace

`/tmp/colibri_colored_block_dispatch_trace600.phases.npz` is byte-identical to
the uninstrumented run for all 600 saved poses, velocities, and times. All six
phase counters are 36000; canonical contact storage flags are explicitly false.
An earlier trace hooked unused legacy methods and had unwritten biased/relax
buffers; those phases were rejected and replaced by actual dispatcher hooks.

Actual color order is contact columns 1 and 2, then joint 0. The biased phase
leaves joint hard residual 7.03e-10 and drive residual 4.13e-9 *before position
integration*. Contact natural-map residual is 0.0169438 m/s. Nineteen interior
loaded tangent rows have maximum velocity-plus-target residual 0.0007833 m/s;
maximum tangent target is 0.0006976 m/s. Base x velocity is -86.425 micrometers/s
at integration. Final relaxation closes joint equations again but leaves contact
residual 7.30e-5 m/s and base x velocity -72.104 micrometers/s.

The historical J-then-C frozen test and the current C-then-J live trace describe
opposite schedules. Both expose unresolved joint/contact coupling; their
individual residual values must not be conflated.

A bounded FP32 replay of the **current** frozen C-then-J equations reduces the
biased contact residual from 0.0169438 to 0.0160832 after 4 extra sweeps, 0.0130035
after 16, and 0.00608981 after 64. Relaxation decreases from 7.299e-5 to 6.050e-5,
3.667e-5, and 2.574e-5 respectively. All original rows, friction, and soft normal
coefficients remain. This does not provide a cheap convergence remedy; a coupled
local solve or justified acceleration is still needed. The frozen biased pose
contains penetration, so its converged recovery velocity is not a live resting
velocity prediction.
