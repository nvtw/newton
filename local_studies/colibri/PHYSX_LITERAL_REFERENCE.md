# Literal PhysX references for Colibri diagnosis

The user authorized experimental translation of the relevant PhysX pieces.
These are isolated diagnostic implementations. They do not replace production
PhoenX or establish that PhysX's formulation meets our physical requirements.

## Implemented pieces

- `physx_normal_reference.py`: FP32 hard-normal accumulated impulse update from
  `solverBlockTGS.cuh`. Ten analytical examples pass. In
  `check_physx_normal_source_parity.py`, 4096 random cases match an extract of
  the original C++ expressions byte for byte on CPU and on CUDA. CUDA is
  compared with those expressions compiled on CUDA: comparing against host
  arithmetic with fused operations disabled is not a GPU parity test.
- `physx_patch_reference.py`: CPU FP32 reference for persistent two-anchor patch
  friction, pooled normal load, source scalar trial and radial projection,
  accumulated-motion targets and actual break flags. It is not yet a live
  Colibri result. Separate impulse points can produce an angular-momentum
  discrepancy; the tests expose this rather than hiding it.
- `physx_d6_reference.py`: bounded prepared-row reference for source drive/hard
  row ordering, preprocessing and sequential updates. Independent one-row
  spring and impulse/work tests pass. This is not a full USD joint importer.

Translated files retain the PhysX BSD notices. The local source checkout is
`ed6e5ca2474c9c80ad4f4826591b88476779c6ef`; it is not proven identical to the
installed ovphysx binary. Compiled-expression checks record source hashes.

## Differences that must remain explicit

1. GPU TGS uses `p8=min(0.8, biasCoefficient)`, with the scene coefficient
   `min(0.9, 2/sqrt(positionIterations))`. At 30 position iterations this is
   approximately 0.36514837, not 0.8. Reproducing this source factor is a
   diagnostic comparison, not permission to change production PhoenX's SOR 1.
2. By default PhysX applies external forces once over the outer interval.
   `eENABLE_EXTERNAL_FORCES_EVERY_ITERATION_TGS` instead applies them before
   each position pass. PhoenX currently integrates gravity each microstep.
   The one-setting PhysX companion completed: enabling per-position-step forces
   reduced 10–60 s net translation from 0.93336 to 0.07741 µm, with maximum
   excursion 0.2550 µm. Runtime physical-property arrays remained identical.
   Gravity timing therefore does not explain PhoenX's larger creep. Artifact:
   `/tmp/colibri_physx_eachforce_comparison.json`.
3. PhysX accumulates linear and scaled angular motion relative to the start of
   an outer interval. Contact targets use frozen prepared Jacobians and that
   accumulated motion. World poses are committed at the end; velocity passes
   do not advance them. PhoenX's current geometry preparation reads updated
   world poses every microstep.
4. PhysX's source twist drive uses a sine-half-angle error and its finite
   iteration update. Its measured settled hinge angle differs from the nominal
   spring equilibrium. Low creep alone does not certify the drive law.

## Live comparison strategy

First test literal patch-friction behavior within the native conditioned
two-body path, clearly labeled a hybrid: it retains native joint response,
normal compliance and force integration. Then replace the remaining contact
and motion pieces separately. Record actual loaded kernel identities, matched
time windows, pose/drive residuals, physical impulse and work balances, and
synchronized step time. Preserve failed candidates. No change is accepted on
FPS alone, and a reference that violates momentum is not a production solution.

Latest validated native two-body baseline remains 3.34 µm/s over 10–60 s and
20.45 FPS, with correct nominal spring equilibrium. More contact sweeps and the
tangent joint-recovery candidate did not achieve stationarity.
