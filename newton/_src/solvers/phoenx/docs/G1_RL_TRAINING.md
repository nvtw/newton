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
  training, checkpointing, and resume.
- `python -m newton.rl eval-g1-ppo`: Load and roll out a saved G1 PPO
  checkpoint.
- `python -m newton.rl gate-g1-ppo`: Run the nanoG1-style quality gate on a
  saved G1 PPO checkpoint.
- `python -m newton._src.solvers.phoenx.benchmarks.bench_g1_rl`: PhoenX G1
  env-step throughput benchmark with optional nanoG1 result ingestion.

## nanoG1 Parity Choices

The environment follows nanoG1's G1 v3 control surface where practical:

- MJCF asset: `unitree_g1/mjcf/g1_29dof.xml` with 36 coordinates, 35 DOFs,
  and 29 actions.
- Observation size: 98.
- Action size: 29.
- Control frame dt: 0.02 s.
- Physics dt: 0.004 s via `sim_substeps=5`.
- Primitive-only collision by default (`parse_meshes=False`), preserving the
  same G1 coordinates/DOFs/actions while avoiding generic mesh/SDF contact
  overflow in high-world-count RL runs.
- Leading 12 leg actions are controlled by default; remaining upper-body actions
  are masked to zero.
- Reward weights mirror the nanoG1 recipe for velocity tracking, base penalties,
  torque proxy, action-rate penalty, alive reward, and termination penalty.
- Default G1 PPO uses three train epochs, matching nanoG1's
  `train.replay_ratio=3.0` more closely than the older four-epoch default.
- Default G1 PPO clips rewards to `[-1, 1]` before advantage/return
  computation, matching PufferLib's learner input scaling and keeping the value
  target bounded for early unstable G1 rollouts.
- Default G1 PPO clips actor and critic gradient norms at `0.3`, matching the
  nanoG1 recipe's max-gradient-norm setting through the reusable Warp Adam path.
- Default G1 PPO uses nanoG1's N1 left/right mirror regularization with
  `mirror_loss_coeff=0.25`, implemented through the reusable Warp PPO mirror-map
  hook and the validated G1 observation/action mirror map from the pinned fork.

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

Use `--mirror-loss-coeff 0.0` on `train-g1-ppo` or `bench_g1_train` to disable
the default nanoG1-style mirror regularizer for throughput-only comparisons. Use
`--reward-clip 0.0` to disable PufferLib-style reward clipping, or
`--max-grad-norm 0.0` to disable gradient clipping.

The gate command mirrors nanoG1's frozen bar: a six-command deterministic
battery with noisy resets for falls/tracking performance, plus a separate
forward-walk diagnostic for action jerk, torso angular velocity, yaw rate, and
leg joint velocity. It exits nonzero when a checkpoint fails unless
`--no-fail-on-gate` is passed.

## Current Benchmark Baseline

Measured on an NVIDIA RTX PRO 6000 Blackwell Workstation Edition with CUDA graph
replay and no learner in the measured loop:

```bash
uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_rl \
    --world-count 4096 --measure-replays 16 --warmup-steps 4
```

Result from this checkpoint:

- 763,732 env steps/s at 4096 worlds.
- 3,818,658 physics steps/s at 4096 worlds.
- 9.8 s setup time at 4096 worlds.

A full train-loop benchmark with 4096 worlds, 64 rollout steps, the default
128x128x128 PPO networks, three train epochs, and the default mirror regularizer
reached 189,632 environment samples/s after the first warmup-heavy iteration.
Disabling the mirror regularizer reached 195,060 environment samples/s:

```bash
uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_train
```

nanoG1 reports about 1.28M environment samples/s while actually training, so
the current mirror-enabled pure-Warp PhoenX G1 training loop is about 6.7x
slower on this training-throughput metric. The throughput-only no-mirror path is
about 6.5x slower.

nanoG1 reports about 8.5M production physics steps/s and 7.25M matched physics
steps/s in its README/benchmark notes, so this PhoenX path is currently about
2.2x slower than nanoG1 production and about 1.9x slower than nanoG1 matched on
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
- The Warp-only PPO loop is reusable but intentionally simple: it does not yet
  include nanoG1's V-trace/prioritized-replay setup, Muon optimizer path, or
  PufferNet model stack.
- Environment stepping can be CUDA-graph replayed, but the full collect-policy-
  update loop is not captured end to end because actions and Warp Tape updates
  allocate intermediate arrays per rollout/update.
- Domain randomization and symmetry regularization still need to be added before
  sample efficiency can be compared meaningfully.

## Next Optimization Targets

1. Avoid generic replicated MJCF setup for high world counts; build or cache a
   compact fixed-topology G1 multi-world model path.
2. Remove avoidable broadphase/contact work for independent flat-ground G1 worlds.
3. Add nanoG1-style V-trace, prioritized replay, and command scheduling to Warp
   PPO.
4. Add a PufferLib interop path behind an optional dependency boundary if we want
   exact nanoG1 trainer compatibility. Its Python package currently depends on
   PyTorch, while its compiled `_C` backend is closer to torch-free.
