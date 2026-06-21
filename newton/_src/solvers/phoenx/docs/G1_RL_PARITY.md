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
- Graph-captured G1 actuator-force gather from PhoenX ADBS drive impulses,
  including reset clearing so stale force cannot leak across episodes.

All of these tests run CUDA-only and use Warp CUDA graph capture.

## Current Difference List

- PhoenX now feeds the G1 torque penalty from the solver's actual ADBS drive
  impulse converted to force on the final physics substep. nanoG1 uses MuJoCo
  `g_af`; the remaining semantic question is whether final-substep force or a
  decimation aggregate best matches MuJoCo's post-step actuator-force buffer.
- The local generic PufferLib checkout is branch `4.0` at `e90b58ed`, not the
  nanoG1 G1 fork. Parity work should use the nanoG1 recipe/deploy files plus
  the pinned fork source above.
- Short train-to-gate probes still show unstable velocity diagnostics. That
  points to remaining simulator/reward/solver quality gaps, not just PPO math.

## Current Measurement

After the RL parity fixes, 20-iteration train-save-load-evaluate probes on the
RTX PRO 6000 Blackwell measured about 188k-226k train env samples/s and
174k-206k total env samples/s. Against the nanoG1 README/reference value of
about 1.276M env samples/s, PhoenX is about 5.6x-6.8x slower in these short
probes. The 20-iteration policy does not pass the walking gate yet, and the
latest diagnostic still shows unstable velocity outliers.

## Next Checks

1. Compare final-substep PhoenX actuator force against nanoG1/MuJoCo `g_af` on
   component-level drive scenarios and decide whether a decimation aggregate is
   needed.
2. Run longer train-to-gate checkpoints after each quality fix and compare
   learning curves, not just final throughput.
3. Keep profiling after correctness changes; optimize kernels only when the
   measured quality path is the one being profiled.
