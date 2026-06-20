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

The reset pose in PhoenX matches nanoG1's host constants: base height 0.785 m,
identity base quaternion, and the same 29 default joint positions.

## PhoenX Parity Fix

The PhoenX G1 RL model was already using the nanoG1 v3 leg stiffness and Unitree
leg damping. However, the imported model had zero `joint_damping`, while nanoG1
adds passive `dof_damping` to the smooth forces. For a position drive with zero
target velocity, folding that passive term into PhoenX `joint_target_kd` is
algebraically the same damping force contribution, `-damping * qd`, for the G1
RL environment.

PhoenX now uses total leg drive damping `(4, 4, 4, 6, 3, 2.2)` on the first 12
controlled leg DOFs. Upper-body damping is unchanged because those values were
already represented by the imported actuator damping used while masked actions
hold the default pose.

## Convergence Sweep

Benchmark command:

```bash
uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_drive_convergence \
    --steps 20 --world-count 2 --action-pattern leg_step --action-amplitude 0.2 \
    --settings rl_current,phoenx_5x4,phoenx_10x4,phoenx_10x8 \
    --reference-setting phoenx_20x8 --json-indent 0
```

The 20-step horizon is 0.4 s at the 50 Hz policy rate. The reference is
`phoenx_20x8` with 0.001 s physics dt, 8 position iterations, and 2 velocity
iterations.

| setting | physics dt | iterations | fall fraction | base height | joint-q RMS vs ref | target RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rl_current` | 0.004 | 2 + 1 | 1.0 | 0.157 m | 0.154 rad | 0.137 rad |
| `phoenx_5x4` | 0.004 | 4 + 1 | 1.0 | 0.184 m | 0.111 rad | 0.104 rad |
| `phoenx_10x4` | 0.002 | 4 + 1 | 0.0 | 0.358 m | 0.020 rad | 0.020 rad |
| `phoenx_10x8` | 0.002 | 8 + 2 | 0.0 | 0.359 m | 0.019 rad | 0.015 rad |
| `phoenx_20x8` | 0.001 | 8 + 2 | 0.0 | 0.377 m | 0.000 rad | 0.012 rad |

A zero-action hold-pose sweep showed the same qualitative result: both 5-substep
settings fell below the 0.35 m gate height by 0.4 s, while 10-substep settings
stayed upright and remained close to the 20x8 reference.

## Training Impact

The fast graph-leapfrog PhoenX G1 setting, `5 substeps x 2 iterations`, reaches
about 1.09M env samples/s but failed the 75M-sample train-to-gate run. The final
checkpoint at 75.2M samples had `battery_perf=0.553`, `battery_falls=1584`, and
`pass_gate=false`.

The more credible `10 substeps x 4 iterations` setting reaches about 609k env
samples/s in the graph-leapfrog train benchmark. A 75.2M-sample single-final-gate
run trained in 145.0 s and finished in 174.6 s including save/reload/gate, but it
still failed the gate with `battery_perf=0.517`, `battery_falls=1584`, and
`pass_gate=false`.

## Current Decision

The default PhoenX G1 RL recipe now uses `sim_substeps=10` and
`solver_iterations=4`. This is slower, but it is the first measured setting in
this sweep that keeps the default G1 contact/drive setup upright over the early
standing and small leg-target tests.

The remaining failure to learn by 75M samples is therefore not solved by this
setting change alone. Next likely checks are:

1. Compare PhoenX contact/friction behavior against nanoG1's host stepper on the
   same open-loop controls.
2. Check whether the PhoenX reward and reset traces match nanoG1 for identical
   policy outputs and noisy reset seeds.
3. Revisit drive/contact solver formulation only with trajectory-level evidence,
   not by reducing iterations for speed.
