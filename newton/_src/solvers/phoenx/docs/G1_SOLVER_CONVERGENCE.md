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

PhoenX also folds nanoG1 passive `dof_damping` into all 29 G1 position-drive
`joint_target_kd` values. For these drives the target velocity is zero, so this
reproduces the same damping force contribution, `-damping * qd`. The generic
PhoenX model adapter forwards `joint_friction` into the ADBS Coulomb-friction
row. `SolverPhoenX` defaults to hard PhoenX Coulomb friction; the G1 recipe
uses `joint_friction_model="mujoco"` so imported `solreffriction` and
`solimpfriction` soften the row in the same spirit as nanoG1 host physics. The
G1 recipe authors nanoG1 `dof_frictionloss = 0.1` on every actuated joint.

## Open-Loop Parity

Benchmark command:

```bash
uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_open_loop_parity \
    --steps 20 --action-pattern leg_step --action-amplitude 0.2 --json-indent 0
```

Current results versus nanoG1 host physics:

| case | friction mode | setting | fall step | final base z | base-z max error | joint-q traj RMS | joint-qd traj RMS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| zero action, 20 steps | none (`scale=0`) | `recipe_default` (`10x4`) | none | 0.760 m | 0.0205 m | 0.0262 rad | 2.02 rad/s |
| zero action, 20 steps | hard | `recipe_default` (`10x4`) | none | 0.763 m | 0.0182 m | 0.0187 rad | 6.20 rad/s |
| zero action, 20 steps | MuJoCo soft | `recipe_default` (`10x4`) | none | 0.763 m | 0.0182 m | 0.0258 rad | 4.52 rad/s |
| leg step, 20 steps | MuJoCo soft | `recipe_default` (`10x4`) | none | 0.758 m | 0.0168 m | 0.0232 rad | 4.48 rad/s |
| zero action, 20 steps | MuJoCo soft | `phoenx_10x8` | none | 0.762 m | 0.0187 m | 0.0237 rad | 3.10 rad/s |

Before the quaternion/contact fixes, the zero-action 5-step `10x4` base-z error
was about 0.049 m and the leg-step run fell at policy step 15. Hard joint
friction improves position parity but creates a large velocity mismatch. MuJoCo
soft friction is more faithful for the nanoG1 comparison and reduces that
velocity error, but it does not close the gap. Extra PhoenX iterations help,
which points to a remaining convergence/formulation gap rather than a pure RL
throughput issue.

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
`sim_substeps=10`, `solver_iterations=4`) and MuJoCo-soft nanoG1 joint friction
enabled, a 20-iteration graph-leapfrog train-save-load probe measured about
188k train env samples/s and about 174k total env samples/s on the RTX PRO 6000
Blackwell. Against the nanoG1 rounded 1.28M env samples/s reference, that short
probe is about 6.8x slower and did not pass the walking gate at 20 iterations.

The friction-enabled setup is more physically faithful, but it adds a scalar
axial row to every G1 joint and remains a throughput regression. Further work
should keep the friction path correct while reducing its per-joint cost and
should prioritize simulator-quality gaps, since the current short probe still
shows unstable velocity diagnostics rather than a merely slow PPO update.

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
