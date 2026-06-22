# PhoenX G1 RL Parity Notes

This note tracks the current quality-facing parity work between PhoenX G1
training and nanoG1. The goal is to train a useful full-coordinate G1 walking
policy with the pure-Warp RL stack, while keeping the training path reusable for
other environments.

## Reference

The nanoG1 checkout at `/home/twidmer/Documents/git/nanoG1` points to the
PufferLib G1 fork `kingjulio8238/PufferLib`, branch `g1`, pinned at commit
`e3825cea`. The exact task source was fetched for inspection from that commit:

- `ocean/g1gpu/g1_gpu.cu` for observation, action, reward, termination, command
  resampling, and reset semantics.
- `ocean/g1gpu/g1_mirror.h` for the left/right mirror map.
- `ocean/g1gpu/g1phys/g1_staged_kernels.cuh` for Unitree PD drive semantics.

## Covered By CUDA Graph Tests

`test_g1_rl_training.py` now covers these high-risk nanoG1/PufferLib contracts:

- Recipe values, deploy constants, action masking, observation layout, and
  default G1 model constants.
- Continuous Gaussian actor log-probability and hand-written PPO actor loss /
  mean and log-std gradients.
- Value clipping, V-trace shifted rollout layout, PufferLib sample-variance
  advantage normalization, and priority replay probability / importance weights.
- Muon optimizer update semantics, transposed MLP weight layout, PufferNet linear
  layer layout, MinGRU equations, recurrent reset behavior, save/load/resume,
  and graph-leapfrog smoke training.
- G1 v3 reward decomposition for a deterministic state, including actual
  actuator-force torque penalty, gait phase, contact/swing/hip/base-height
  shaping, action-rate penalty, dt scaling, done, and success metrics.
- Graph-captured G1 actuator-force gather using nanoG1's actuator-model formula,
  including force clamps and reset clearing so stale force cannot leak across
  episodes.

All of these tests run CUDA-only and use Warp CUDA graph capture.

## Current Difference List

- PhoenX now feeds the G1 torque penalty from the same actuator-model force
  signal nanoG1 writes to `g_af`: clamped `kp * (target - q) - kd * qd` for
  the 12 Unitree leg actuators and clamped model position-actuator force for
  the remaining waist/arm actuators. The signal is gathered before the final
  decimation substep solve, matching nanoG1's `k3_rne_act_solve -> ... ->
  k_epi` ordering.
- The local generic PufferLib checkout is branch `4.0` at `e90b58ed`, not the
  nanoG1 G1 fork. Parity work should use the nanoG1 recipe/deploy files plus
  the pinned fork source above.
- Short train-to-gate probes still fail the walking gate. The current evidence
  points to remaining simulator/reward/solver quality gaps and sample-efficiency
  issues, not just PPO math.
- The Python gate diagnostic now rotates root linear velocity with Newton's
  free-joint quaternion layout, `x, y, z, w`. The previous helper interpreted
  the same four numbers as `w, x, y, z`, corrupting the printed
  `mean_linear_velocity_error` diagnostics. The actual gate pass/fail metric was
  already computed on the device from `step_successes`, so this was a diagnostic
  bug, not a training-quality fix.

## Discrepancy Ledger

This table is the working order for quality bugs. New tuning or optimization
should land only after one row has a measured discrepancy and a regression test
or benchmark note.

| area | reference | PhoenX status | next action |
| --- | --- | --- | --- |
| PPO/V-trace replay math | PufferLib G1 fork at `e3825cea` | CUDA graph tests cover V-trace shifted rollout layout, priority weights, and whole-trajectory gather/scatter back to rollout buffers. | Keep as regression coverage; do not tune PPO until a new mismatch is proven. |
| Muon/network kernels | PufferLib source plus local finite-difference tests | Tests cover Muon update semantics, PufferNet linear layout, MinGRU equations, mirror maps, and manual PPO actor/value gradients. | Re-check precision/layout only if training probes show learner instability independent of physics. |
| Gate diagnostics | nanoG1-style command battery | Fixed Python velocity diagnostic quaternion order to Newton `xyzw`; graph-captured regression covers the case. | Re-run gates only when diagnostics are needed; do not treat this as quality progress. |
| Graph overlap / stale policy | Same recipe in eager and graph-leapfrog modes | A 60-iteration eager train-to-gate probe failed similarly to graph mode, so stream overlap/stale rollout policy is not the primary quality blocker. | Keep graph mode for throughput, but debug quality in the simpler eager path when possible. |
| G1 env/reward contract | `g1_gpu.cu`, `recipe.py`, deploy constants | Tests cover observation/action layout, actuator-force torque penalty, reward decomposition, gait/success terms, command constants, and graph recurrent reset behavior. | Add targeted tests for command resampling/reset timing and done/bootstrap semantics before changing rewards. |
| Drive/contact physics | nanoG1 host physics plus first-principles drive tests | No-contact PD/friction/armature response is close; grounded contact/friction coupling still differs, especially tangential support at `5x2`. | Prioritize first-principles grounded drive/contact tests and compare solver settings before changing training hyperparameters. |
| End-to-end training quality | nanoG1 reaches a working policy in roughly the README time scale | Current 75M-sample PhoenX runs train in a few minutes but fail the gate. | Run full probes only after a ledger row changes; record before/after quality and throughput. |

## Current Measurement

The latest full quality-facing probe used 4096 worlds, 64 rollout steps, the
default nanoG1-timed `5x2` recipe, graph-leapfrog execution, and one 75.2M
sample chunk. It completed training in 185.4 s and the whole train-save-reload
gate run in 201.2 s, about 405.7k train environment samples/s, but still failed
the gate with `battery_perf=0.325` and `battery_falls=372`. A 60-iteration eager
probe also failed at 15.7M samples with `battery_perf=0.321`, so the current
blocker is quality/sample efficiency, not only the two-stream graph schedule.

## Next Checks

1. Add or tighten command/reset/done-bootstrap tests against the pinned nanoG1
   source so the environment contract is exhausted before reward tuning.
2. Compare grounded drive/contact response against first-principles expectations
   and nanoG1 traces, especially tangential support and effective stiffness at
   the production `5x2` setting.
3. Run longer train-to-gate checkpoints only after a concrete discrepancy is
   fixed, then compare learning curves and gate diagnostics before profiling for
   throughput.
