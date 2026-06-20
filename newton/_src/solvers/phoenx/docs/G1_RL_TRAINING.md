# PhoenX G1 RL Training Notes

This is the current PhoenX path for reproducing nanoG1-style G1 training without
adding a PyTorch dependency to Newton's Warp-only RL stack.

## Entry Points

- `newton.rl.EnvG1PhoenX`: vectorized Unitree G1 environment backed by
  `SolverPhoenX`.
- `newton.rl.train_g1_ppo`: Warp-only PPO training loop using the reusable
  `collect_ppo_rollout` helper.
- `newton.rl.capture_env_steps`: reusable CUDA graph-capture helper for vectorized
  environments.
- `python -m newton.rl train-g1-ppo`: CLI wrapper for pure-Warp PPO
  training, checkpointing, resume, and the experimental
  `--execution-mode graph_leapfrog` rollout/update/copy graph schedule.
- `python -m newton.rl eval-g1-ppo`: Load and roll out a saved G1 PPO
  checkpoint.
- `python -m newton.rl gate-g1-ppo`: Run the nanoG1-style quality gate on a
  saved G1 PPO checkpoint.
- `python -m newton._src.solvers.phoenx.benchmarks.bench_g1_rl`: PhoenX G1
  env-step throughput benchmark with optional nanoG1 result ingestion.
- `python -m newton._src.solvers.phoenx.benchmarks.bench_g1_train`: Full
  collect-update PPO throughput benchmark.
- `python -m newton._src.solvers.phoenx.benchmarks.bench_g1_train_to_gate`:
  End-to-end train, save, reload, and quality-gate benchmark for estimating
  samples/time-to-walk.
- `python -m newton._src.solvers.phoenx.benchmarks.experimental.bench_g1_train_leapfrog`:
  experimental rollout/update overlap benchmark using separate CUDA graphs.

## nanoG1 Parity Choices

The environment follows nanoG1's G1 v3 control surface where practical:

## Recipe Surface

The default PhoenX G1 run is centralized in
`newton/_src/solvers/phoenx/rl_training/g1_recipe.py`. Tune dt/decimation,
solver iterations, reward weights, PPO replay/V-trace settings, BF16 manual MLP
kernels, and the left/right mirror regularizer there first; the train CLI and
benchmark defaults read from the same constants.

- MJCF asset: `unitree_g1/mjcf/g1_29dof.xml` with 36 coordinates, 35 DOFs,
  and 29 actions.
- Observation size: 98.
- Action size: 29.
- Control frame dt: 0.02 s.
- Physics dt: 0.002 s via `sim_substeps=10`. nanoG1 uses 0.004 s x 5,
  but the current PhoenX convergence sweep requires 10 substeps and 4
  position iterations for credible early G1 contact/drive behavior.
- Primitive-only collision by default (`parse_meshes=False`), preserving the
  same G1 coordinates/DOFs/actions while avoiding generic mesh/SDF contact
  overflow in high-world-count RL runs.
- Leading 12 leg actions are controlled by default; remaining upper-body actions
  are masked to zero.
- Reward weights mirror the nanoG1 recipe for velocity tracking, base penalties,
  torque proxy, action-rate penalty, alive reward, and termination penalty.
- Default G1 PPO uses trajectory-shaped prioritized minibatches with
  `minibatch_size=32768`, `replay_ratio=3.0`, and `priority_alpha=0.4`,
  matching nanoG1's learner schedule more closely than the older four-epoch
  full-buffer default.
- Default G1 PPO clips rewards to `[-1, 1]` before advantage/return
  computation, matching PufferLib's learner input scaling and keeping the value
  target bounded for early unstable G1 rollouts.
- Default G1 PPO clips actor and critic gradient norms at `0.3`, matching the
  nanoG1 recipe's max-gradient-norm setting through the reusable Warp Adam path.
- Default G1 PPO uses nanoG1's N1 left/right mirror regularization with
  `mirror_loss_coeff=0.25`, implemented through the reusable Warp PPO mirror-map
  hook and the validated G1 observation/action mirror map from the pinned fork.
- Default G1 PPO uses BF16 inputs with FP32 accumulation for manual CUDA MLP
  weight-gradient tile matmul, plus BF16 hidden-layer forward tile matmul for
  large PPO minibatches. This follows PufferLib's default precision direction
  while keeping FP32 master parameters and optimizer state. Pass
  `--manual-mlp-weight-grad-dtype float32` or
  `--manual-mlp-forward-dtype float32` to use exact FP32 manual kernels for
  those paths.

## End-to-End Checkpoint Workflow

A minimal user-facing pure-Warp lifecycle is:

```bash
uv run --extra dev -m newton.rl train-g1-ppo \
    --iterations 2 --rollout-steps 8 --world-count 4096 \
    --checkpoint-path /tmp/phoenx_g1_{iteration}.npz \
    --checkpoint-interval 1 --no-command-randomization

uv run --extra dev -m newton.rl eval-g1-ppo \
    --checkpoint /tmp/phoenx_g1_2.npz --steps 4 --world-count 4096

uv run --extra dev -m newton.rl gate-g1-ppo \
    --checkpoint /tmp/phoenx_g1_2.npz --no-fail-on-gate

uv run --extra dev -m newton.rl train-g1-ppo \
    --iterations 1 --rollout-steps 8 --world-count 4096 \
    --resume-checkpoint /tmp/phoenx_g1_2.npz \
    --checkpoint-path /tmp/phoenx_g1_{iteration}.npz \
    --checkpoint-interval 1 --no-command-randomization
```

The checkpoint stores actor, critic, optimizer state, PPO config, network shape,
and the absolute training iteration. Resuming from `/tmp/phoenx_g1_2.npz` writes
`/tmp/phoenx_g1_3.npz` and logs `iter=0002`.

Use `--execution-mode graph_leapfrog` on `train-g1-ppo` or `bench_g1_train`
to run the experimental separate-stream schedule: a frozen-policy rollout graph,
PPO update graph, and small policy-copy graph are replayed on separate CUDA
streams. Eager execution remains the default while this mode is validated for
longer training runs. Use `--mirror-loss-coeff 0.0` on `train-g1-ppo` or
`bench_g1_train` to disable
the default nanoG1-style mirror regularizer for throughput-only comparisons. Use
`--reward-clip 0.0` to disable PufferLib-style reward clipping,
`--vtrace-rho-clip 0.0 --vtrace-c-clip 0.0` to disable V-trace replay
correction, `--manual-mlp-weight-grad-dtype float32` to disable BF16 MLP
weight-gradient tile matmul, `--manual-mlp-forward-dtype float32` to disable
large-minibatch BF16 MLP forward tile matmul, or `--max-grad-norm 0.0` to
disable gradient clipping. The default training loop samples randomized G1
commands on the device and keeps progress monitoring compact: one 10-float
metric buffer is copied to pinned host memory per iteration. Use
`--no-readback-diagnostics` to skip that diagnostic copy in strict
capture/benchmark runs; history entries use zero placeholders for those
diagnostics in that mode.

The gate command mirrors nanoG1's frozen bar: a six-command deterministic
battery with noisy resets for falls/tracking performance, plus a separate
forward-walk diagnostic for action jerk, torso angular velocity, yaw rate, and
leg joint velocity. It exits nonzero when a checkpoint fails unless
`--no-fail-on-gate` is passed.

For the nanoG1-style time-to-walk metric, use `bench_g1_train_to_gate`. It trains
in chunks, saves a checkpoint, reloads that checkpoint, runs the quality gate,
and reports the first passing checkpoint if one is found:

```bash
uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_train_to_gate \
    --execution-mode graph_leapfrog --chunk-iterations 25

uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_train_to_gate \
    --max-iterations 1 --chunk-iterations 1 --world-count 2 --rollout-steps 1 \
    --hidden-layers 8 --minibatch-size 1 --replay-ratio 1.0 --train-epochs 1 \
    --battery-steps 1 --seeds-per-command 1 --diagnostic-steps 1 \
    --diagnostic-world-count 1 --no-command-randomization
```

## Current Benchmark Baseline

Measured on an NVIDIA RTX PRO 6000 Blackwell Workstation Edition with CUDA graph
replay and no learner in the measured loop:

```bash
uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_rl \
    --world-count 4096 --measure-replays 16 --warmup-steps 4
```

Older fast-setting result with `sim_substeps=5`, `solver_iterations=2`,
`velocity_iterations=1` relax pass enabled and with the environment no longer
nesting SolverPhoenX substeps inside its own policy-step decimation loop:

- 1,499,589 env steps/s at 4096 worlds.
- 7,497,944 physics steps/s at 4096 worlds.
- 5.0 s setup time at 4096 worlds.

A full train-loop benchmark with 4096 worlds, 64 rollout steps, the default
128x128x128 PPO networks, `minibatch_size=32768`, `replay_ratio=3.0`,
`priority_alpha=0.4`, V-trace replay correction, mirror regularization, and
BF16 manual MLP weight-gradient tile matmul plus large-minibatch BF16 hidden
forward tile matmul reached 635,494 environment samples/s with compact
diagnostics and 631,135 environment samples/s with `--no-readback-diagnostics`
after warmup on 2026-06-20. The corresponding no-readback physics rate was
3,155,676 steps/s. Short-run variation is larger than the compact diagnostic
copy cost, so the single diagnostic readback remains throughput-neutral at this
scale.

```bash
uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_train \
    --iterations 6 --warmup-iterations 2

uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_train \
    --iterations 6 --warmup-iterations 2 --execution-mode graph_leapfrog
```

nanoG1 reports about 1.28M environment samples/s while actually training, so
the current mirror-enabled pure-Warp PhoenX G1 training loop is about 2.01x
slower on this training-throughput metric. An experimental frozen-policy
leapfrog benchmark that launches a rollout graph, an update graph, and a small
policy-copy graph on separate streams reached 1,093,441 environment samples/s
with device-side command, action-noise, and minibatch seed counters. The same
separate-graph schedule is now exposed as the experimental
`train-g1-ppo --execution-mode graph_leapfrog` mode. The production
`bench_g1_train --execution-mode graph_leapfrog --no-readback-diagnostics`
path measured 1,093,535 environment samples/s after excluding the final
update-only graph drain interval from the steady-state mean, reducing the
throughput gap to about 1.17x versus nanoG1's reported 1.28M samples/s.

After the G1 solver-convergence sweep, the default recipe moved to
`sim_substeps=10` and `solver_iterations=4` because the faster 5-substep settings
fell below the 0.35 m gate height in short standing/drive tests. With the
safer 10x4 setting, this command measured 609,052 environment samples/s, about
2.10x slower than nanoG1's reported 1.28M samples/s:

```bash
uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_train \
    --execution-mode graph_leapfrog --no-readback-diagnostics \
    --sim-substeps 10 --solver-iterations 4
```

A 75.2M-sample train-save-reload-gate run with this setting completed
in 174.6 s including the final gate but still failed the gate
(`battery_perf=0.517`, `battery_falls=1584`). See `G1_SOLVER_CONVERGENCE.md` for
the fidelity study.

The old fast graph-leapfrog production path was remeasured with the default
auto fast-tail scheduler at 1,086,344 samples/s on 2026-06-20. Setting
`--prepare-refresh-stride 3` was neutral at 1,087,558 samples/s, while forcing
block-world schedulers was slower: 1,071,348 samples/s for `block_world_32`,
1,011,811 samples/s for `block_world_64`, and 931,857 samples/s for
`block_world_128`. The default scheduler therefore remains the best measured
choice for this G1 training workload.

```bash
uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.experimental.bench_g1_train_leapfrog \
    --iterations 4 --warmup-iterations 1 --graphs
```

The same short benchmark measured 966,730 samples/s for sequential graph replay
and 641,921 samples/s for the eager synchronous schedule, so the stream overlap
only pays off once the rollout, update, and copy phases are graph-replayed. An
Nsight Systems profile of the normal train-loop benchmark shows the largest
kernels are split between the Warp PPO learner and generic PhoenX rollout. Top
CUDA kernels by total GPU time in that profile were:

1. `dense_weight_grad_bf16_tiled`: 15.4%
2. `_make_fast_tail_prepare_plus_iterate`: 13.7%
3. `_per_world_greedy_coloring`: 11.6%
4. CUB radix sort onesweep (`long, int`): 5.3%
5. `dense_forward_bf16_tiled`: 4.9%
6. `eval_articulation_fk`: 4.6%
7. CUB radix sort onesweep (`int, int`): 3.9%
8. `dense_layer`: 2.8%
9. `_make_fast_tail_relax`: 2.4%
10. BF16 cast kernels: 1.7%

An Nsight Systems profile of the graph-leapfrog production path shifts the
dominant cost to PhoenX rollout kernels. Top CUDA kernels by total GPU time in
that profile were:

1. `_make_fast_tail_prepare_plus_iterate`: 19.9%
2. `_per_world_greedy_coloring`: 17.4%
3. CUB radix sort onesweep (`long, int`): 7.9%
4. `eval_articulation_fk`: 7.0%
5. CUB radix sort onesweep (`int, int`): 5.9%
6. `_make_fast_tail_relax`: 3.5%
7. `_contact_container_copy_current_to_prev`: 2.8%
8. `dense_layer`: 2.7%
9. `_apply_joint_forces`: 2.6%
10. `eval_articulation_ik`: 2.2%

nanoG1 reports about 8.5M production physics steps/s and 7.25M matched physics
steps/s in its README/benchmark notes, so this PhoenX path is currently about
1.1x slower than nanoG1 production and slightly faster than nanoG1 matched on
raw no-learner stepping. The benchmark can invoke the local nanoG1 checkout when
cloud execution is intended:

```bash
uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_rl --run-nanog1
```

That command shells out to:

```bash
cd /home/twidmer/Documents/git/nanoG1 && modal run bench/bench_nanog1.py --config production
```

## Known Gaps To nanoG1

- nanoG1 compiles a G1-specific CUDA engine through its pinned PufferLib fork;
  PhoenX currently runs through Newton's generic model, collision, broadphase,
  contact reduction, graph coloring, and solver paths.
- Full mesh collision is available with `parse_meshes=True`, but high-world-count
  RL runs should use primitive collision until the mesh/SDF reducer path is sized
  and optimized for thousands of humanoid worlds.
- nanoG1's Modal image installs `torch`, so using the same full PufferLib training
  stack is not currently a torch-free route even though the environment core is
  specialized CUDA.
- The Warp-only PPO loop is reusable and now supports trajectory minibatch
  replay, rollout-advantage priority sampling, V-trace replay correction,
  BF16 MLP weight-gradient tile matmul, and large-minibatch BF16 hidden forward
  tile matmul, but it does not yet include nanoG1's Muon optimizer path or
  PufferNet model stack.
- Environment stepping, command randomization, stochastic action sampling,
  priority minibatch sampling, Adam optimizer step state, and the manual PPO
  update pieces are CUDA-graph capturable with device-side counters. The
  default train loop still uses the eager collect-update schedule, while the
  separate rollout/update/copy graph schedule is available through
  `execution_mode="graph_leapfrog"`. Logging and checkpointing remain
  host-driven.
- Reset/domain randomization and command scheduling are still lighter than
  nanoG1, so sample efficiency is not yet directly comparable.

## Next Optimization Targets

1. Avoid generic replicated MJCF setup for high world counts; build or cache a
   compact fixed-topology G1 multi-world model path.
2. Remove avoidable broadphase/contact work for independent flat-ground G1 worlds.
3. Run `bench_g1_train_to_gate` to measure PhoenX samples-to-gate and identify
   whether the remaining gap is throughput, sample efficiency, or both.
4. Tighten the remaining host synchronization around metrics/checkpoint cadence.
5. Upgrade command scheduling and remaining domain randomization toward the
   nanoG1 recipe.
6. Add a PufferLib interop path behind an optional dependency boundary if we want
   exact nanoG1 trainer compatibility. Its Python package currently depends on
   PyTorch, while its compiled `_C` backend is closer to torch-free.
