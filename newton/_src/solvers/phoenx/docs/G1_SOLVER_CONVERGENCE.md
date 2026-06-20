# PhoenX G1 Solver Convergence Notes

This note records the current G1 solver-fidelity findings that matter for
PhoenX RL. The goal is sim-to-real-useful training, so these settings prioritize
physically credible drive/contact behavior over the fastest possible sample rate.

## nanoG1 Reference

nanoG1 v3 uses the following production physics and drive setup:

- Control frame: 0.02 s.
- Physics step: 0.004 s, decimation 5.
- Solver: Newton iterations 2, line-search iterations 3.
- Actions: first 12 leg actions only, action scale 0.25 rad.
- Leg stiffness: `(100, 100, 100, 150, 40, 40)` per side.
- Leg Unitree damping: `(2, 2, 2, 4, 2, 2)` per side.
- Passive `dof_damping`: `(2, 2, 2, 2, 1, 0.2)` per side.
- Effective zero-target-velocity damping on controlled legs:
  `(4, 4, 4, 6, 3, 2.2)` per side.

The reset pose in PhoenX matches nanoG1 host constants at the semantic level:
base height 0.785 m, identity base orientation, and the same 29 default joint
positions. Newton stores free-joint quaternions as `x, y, z, w`; nanoG1/MuJoCo
stores them as `w, x, y, z`, so parity tools must reorder only for comparison.

## PhoenX Parity Fixes

Two fixes were required before solver tuning was meaningful:

1. The G1 RL environment initialized and observed the free-joint quaternion in
   nanoG1/MuJoCo order. In Newton layout, `[1, 0, 0, 0]` is a 180-degree X
   rotation, which put the legs above the pelvis and left the robot airborne.
   The environment now initializes identity as `[0, 0, 0, 1]` and observes
   gravity with the same layout.
2. The raw Menagerie G1 foot point contacts produce zero ground contacts in
   PhoenX at reset. The default G1 recipe now uses
   `contact_geometry="nanog1_foot_boxes"`, which disables those imported foot
   point colliders and adds the two MuJoCo/nanoG1 foot contact boxes with local
   position `(0.04, 0.0, -0.029)`, half-extents `(0.09, 0.03, 0.008)`, and
   friction `0.6`. The raw `contact_geometry="mjcf"` mode remains available
   as a diagnostic.

PhoenX also folds nanoG1 passive leg `dof_damping` into the first 12 leg drive
`joint_target_kd` values. For a position drive with zero target velocity, this
is the same damping force contribution, `-damping * qd`, for the controlled G1
legs.

## Open-Loop Parity

Benchmark command:

```bash
uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_open_loop_parity \
    --steps 20 --action-pattern leg_step --action-amplitude 0.2 --json-indent 0
```

Current results versus nanoG1 host physics:

| case | setting | fall step | final base z | base-z max error | joint-q traj RMS |
| --- | --- | ---: | ---: | ---: | ---: |
| zero action, 5 steps | `fast_5x2` | none | 0.764 m | 0.019 m | 0.0044 rad |
| zero action, 5 steps | `recipe_default` (`10x4`) | none | 0.777 m | 0.0066 m | 0.0048 rad |
| zero action, 5 steps | `phoenx_10x8` | none | 0.778 m | 0.0059 m | 0.0048 rad |
| leg step, 20 steps | `fast_5x2` | none | 0.751 m | 0.040 m | 0.0609 rad |
| leg step, 20 steps | `recipe_default` (`10x4`) | none | 0.756 m | 0.019 m | 0.0295 rad |
| leg step, 20 steps | `phoenx_10x8` | none | 0.757 m | 0.0176 m | 0.0266 rad |

Before the quaternion/contact fixes, the zero-action 5-step `10x4` base-z error
was about 0.049 m and the leg-step run fell at policy step 15. The current
remaining mismatch is a real solver/contact/drive-formulation difference, not a
reset-layout bug.

## Convergence Sweep

Benchmark command:

```bash
uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_drive_convergence \
    --steps 20 --world-count 4 --action-pattern leg_step --action-amplitude 0.2 \
    --reference-setting phoenx_20x8 --json-indent 0
```

The 20-step horizon is 0.4 s at the 50 Hz policy rate. The reference is
`phoenx_20x8` with 0.001 s physics dt, 8 position iterations, and 2 velocity
iterations.

| setting | physics dt | iterations | fall fraction | min base height | joint-q RMS vs ref | target RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rl_current` (`5x2`) | 0.004 | 2 + 1 | 0.0 | 0.752 m | 0.0965 rad | 0.0552 rad |
| `phoenx_5x4` | 0.004 | 4 + 1 | 0.0 | 0.750 m | 0.0615 rad | 0.0477 rad |
| `phoenx_10x4` | 0.002 | 4 + 1 | 0.0 | 0.758 m | 0.0351 rad | 0.0533 rad |
| `phoenx_10x8` | 0.002 | 8 + 2 | 0.0 | 0.759 m | 0.0289 rad | 0.0497 rad |
| `phoenx_20x8` | 0.001 | 8 + 2 | 0.0 | 0.766 m | 0.0000 rad | 0.0709 rad |

The `10x4` default is still the conservative setting: it materially improves
trajectory agreement versus `5x2` without making the recipe as expensive as the
`20x8` reference.

## Training Impact

With the grounded G1 setup (`contact_geometry="nanog1_foot_boxes"`,
`sim_substeps=10`, `solver_iterations=4`), a short graph-leapfrog training run
measured about 442k env samples/s, or about 4.42M physics steps/s, on the RTX
PRO 6000 Blackwell. Against nanoG1 rounded 1.28M env samples/s reference, that
is about 2.9x slower.

This is slower than the pre-fix fast setting, but it is the first measured setup
in which the full-coordinate PhoenX G1 starts grounded, keeps reset contacts,
and tracks nanoG1 open-loop behavior closely enough for RL tuning to be
meaningful.

## Current Decision

The default PhoenX G1 RL recipe keeps `sim_substeps=10`, `solver_iterations=4`,
and `contact_geometry="nanog1_foot_boxes"`. Speed remains important, but
further optimization should preserve the grounded reset, active foot contacts,
and open-loop parity checks above.

Next likely checks:

1. Re-run train-to-gate with the fixed reset/contact setup.
2. Compare PhoenX reward/reset traces against nanoG1 for identical policy
   outputs and noisy reset seeds.
3. Profile the grounded-contact pipeline before optimizing solver kernels; the
   old profiles were collected with the physically invalid airborne reset.
