# PhoenX Performance Notes

Curated lessons-learned from past performance work on the PhoenX solver. Goal is to avoid re-trying ideas that have already been characterised, capture *why* a change won or lost, and surface knobs that scenes can opt into.

This is **not** a substitute for `git log` — it's a hand-maintained shortlist of the load-bearing decisions and the dead ends.

## Active wins

### Greedy graph coloring (single-world)
- Replaces round-based JP MIS for the global colouring on the single-world layout. Picks the smallest-free-colour for each MIS commit instead of "round = colour", landing 2-3x fewer colours on dense contact graphs (kapla, box stacks).
- Implementation: `partitioning_coloring_incremental_greedy_kernel` (`graph_coloring/graph_coloring_common.py`). Forbidden-colour bitmask is `int64`, capped at 64 colours. Overflow falls back to round-based JP within the same captured CUDA graph (`build_csr_greedy_with_jp_fallback`).
- Per-cid commits use `wp.atomic_sub(num_remaining, 0, 1)` so the `wp.capture_while` predicate is strictly monotonic.
- Toggle: `PHOENX_USE_GREEDY_COLORING` (defaults `True`). Disable to fall back to JP.

### Greedy CSR post-pass fusion
- The greedy coloring's commit step writes only the per-vertex `(colour, tid)` packed tag. Building the colour-major CSR (`element_ids_by_color`, `color_starts`) used to be a separate compact + scan launch each.
- Both passes now run in one kernel (`incremental_tile_compact_csr_and_advance_kernel`), saving a launch per build and ~1.5% of frame time on kapla.

### Greedy coloring int32 colour-tag mirror
- Added a parallel ``color_tags: wp.array[wp.int32]`` alongside the int64 ``partition_data_concat``. The greedy MIS+colour kernel reads ``color_tags`` on its hot path -- the per-thread early-exit check and the per-neighbour adjacency walk -- halving the per-read width.
- The kernel still mirror-writes ``partition_data_concat`` on commit; histogram / scatter post-passes keep reading the int64 view because the JP-fallback rewrites ``partition_data_concat`` only and leaves ``color_tags`` stale.
- Per-launch greedy kernel time on kapla: 54.7us → 51.8us (~5%); per-step coloring time 4.52ms → 4.32ms (~4%).

### Greedy coloring without compaction
- Original implementation maintained a compacted `remaining_ids` list across MIS rounds. The compact kernel was ~16% of frame time on kapla.
- Switched to a persistent grid-stride loop where each thread reads its own packed tag and skips already-coloured cids early. Net win was ~16% step time despite the kernel doing extra work in late rounds (most threads early-exit).
- **If you re-add compaction, measure the compact kernel cost first** — it has to beat ~16% to pay off.

### Tail-fuse kernel for small colours (single-world)
- Persistent-grid single-world iterate/prepare/relax kernels have an internal `count <= fuse_threshold` hand-off that clears `head_active`. A single-block tail kernel (`*_singleworld_fused_kernel`) then drains the small colours back-to-back with `__syncthreads` between, avoiding per-colour kernel-launch boundaries.
- Threshold: `FUSE_TAIL_MAX_COLOR_SIZE` in `solver_config.py`. Currently small (drains the trailing 0.7% of work).

### `_FUSED_INNER_SWEEPS = 2` (multi-world fast-tail)
- Wires up the per-cid register cache that `*_iterate_multi` already builds. At `1` the multi-sweep helper still loaded body state every sweep; at `2` each cid does two PGS sweeps from registers before writing back.
- Trade-off: cross-colour PGS feedback drops from `solver_iterations` rounds to `solver_iterations / FUSED_INNER_SWEEPS`. Tested `4` and it breaks `test_slam_ball_into_stack` — heavy ball into a tower needs the finer feedback.
- Multi-world g1_flat / h1_flat: **+15-22% env_fps** at 1024-16384 worlds.
- Constant lives in `solver_phoenx_kernels.py`. **Do not bump above 2 without re-running the impact-stack tests.**

### Revolute-only kernel specialisation (single + multi world)
- When every joint is `JointMode.REVOLUTE` (or there are no joints at all), the iterate kernels skip the per-cid `read_int(_OFF_JOINT_MODE)` and the four-way `joint_mode` branch in `actuated_double_ball_socket_iterate{,_multi}`. They call `revolute_iterate{,_multi}` / `revolute_prepare_for_iteration` directly.
- Detection: `PhoenXWorld._use_revolute_specialization`, set by scanning the `joint_mode` array once during `initialize_actuated_double_ball_socket_joints`. Default `True` for `num_joints == 0`.
- Single-world kernels are factory-generated with a `wp.static` `revolute_only` axis (`_make_singleworld_persistent_kernel` / `_make_singleworld_fused_kernel`). Multi-world fast-tail kernels likewise (`_make_fast_tail_*`).
- Wins are modest standalone (the contribution is mostly subsumed under `_FUSED_INNER_SWEEPS = 2` for multi-world); main value is keeping the iterate kernel binary smaller for the common all-revolute case.

### Per-substep `inverse_inertia_world` refresh
- After `_integrate_positions` rotates each dynamic body, `bodies.inverse_inertia_world` was stale for the next substep's solve. For anisotropic links this biased the angular impulse direction over the substep loop.
- Fix: `_phoenx_refresh_world_inertia_kernel` runs after every substep. Real correctness gain on G1/H1 hold-pose parity tests; modest cost.

### Adaptive threads-per-world (multi-world fast-tail)
- The fast-tail launch grid is fixed at `num_worlds * _STRAGGLER_BLOCK_DIM` (= warp), but the active lane count per world (`tpw`) is picked per step from the colour histogram. `_pick_threads_per_world_kernel` reads `_world_num_colors` and `_world_csr_offsets` and writes `_tpw_choice[0]`.
- Pinned to 32 below `8 * sm_count` worlds (picker overhead would never pay off). Captured-graph safe (no host sync).
- User opt-out: `threads_per_world={8,16,32}` in the solver constructor.

### Inertia + force-clear fusion
- Damping + rotated-inertia refresh + force/torque zeroing were three back-to-back per-body kernels with the same dim/gate. Fused into `_phoenx_update_inertia_and_clear_forces_kernel`. Saves ~3 launches per step.

## Tried and reverted

### Substep mega-kernel (one block per world, all substeps in one launch)
- Goal: collapse the entire `num_substeps` loop (forces, prepare, iterate, integrate, relax, inertia refresh, kinematic, damping, accumulate) into a single block-per-world kernel using the existing per-world body / constraint CSRs.
- Implemented and tested with `block_dim ∈ {32, 64, 128}`. Results were mixed: some configs +5-10%, others -5-10%. Net ~neutral.
- Why it didn't pay off:
  - Register pressure: a single mega-kernel inlines forces + prepare + iterate*N + integrate + relax*M + refresh + ...; occupancy drops vs. smaller specialised kernels.
  - At `block_dim=32` (one warp per world) `__syncthreads()` collapses to warp-sync, but the kernel is bandwidth-bound on body / constraint state and the launch-overhead saving is small relative to GPU time at the world counts we benchmark.
  - At `block_dim=128` idle lanes during small per-color cid lists waste occupancy.
- **Don't redo without a fundamentally different design** (e.g. splitting body work and constraint work into two mega-kernels, or moving body state into a packed struct).
- Reverted in commits `2566ef65 / 3216a44a / cb4dfcef`.

### `_FUSED_INNER_SWEEPS = 4`
- Doubles the per-cid register-cache benefit again (8 sweeps from one body load). Breaks `test_slam_ball_into_stack`: a heavy ball into a tower needs the finer cross-colour PGS feedback to dissipate the impulse without driving bodies through neighbours.
- Settled on `2`. Don't push past without re-running impact-stack scenes.

### `__ffsll` for the greedy first-free-colour scan
- Wrapped the CUDA ``__ffsll`` intrinsic in a ``wp.func_native`` (``_lowest_set_bit``) and used it in both single-world and per-world greedy kernels in place of the 64-iteration linear bit-scan.
- Per-launch greedy kernel time was unchanged (~52us on kapla single-world, multi-world bench unchanged). The original scan had an early-out on first set bit, and most masks have a low-set bit, so it rarely paid the full 64 iterations. The intrinsic is cleaner code but doesn't move the perf needle.
- Kept the change for clarity; PR-reviewable as a one-line refactor that removes the open ``__ffsll`` TODO.

### Single-world multi-sweep iterate
- Tried wiring `_FUSED_INNER_SWEEPS` into the single-world iterate path (call `*_iterate_multi(num_sweeps)` instead of `*_iterate`) and halving the outer `solver_iterations` loop.
- ~3% kapla regression (single-world contact-heavy scene). The body-load saving exists but the per-launch cost grows ~2x and you lose half the cross-colour feedback granularity.
- Multi-world wins because the entire substep runs in one launch, so launch-overhead saving stacks with body-load saving. Single-world has many head+tail launches per sweep, so the launch-overhead saving doesn't materialise.
- **Keep single-world on `num_sweeps = 1`.**

### Body wp.func extraction → per-block-per-world dispatcher
- Extracted every per-body kernel (`_phoenx_apply_forces_and_gravity_kernel` etc.) into `wp.func` helpers taking a body id, planning to grid-stride them inside a future mega-kernel.
- The mega-kernel itself didn't pay off (see above), so the extraction was reverted to keep the diff minimal. The funcs are cheap to re-add if a future fused design wants them.

## Knobs and where they live

| Constant / flag                     | File                                              | Effect                                           |
| ----------------------------------- | ------------------------------------------------- | ------------------------------------------------ |
| `_FUSED_INNER_SWEEPS`               | `solver_phoenx_kernels.py`                        | Multi-world per-cid sweep count (currently `2`)  |
| `_STRAGGLER_BLOCK_DIM`              | `solver_phoenx_kernels.py`                        | Multi-world fast-tail warp size (= 32)           |
| `_choose_fast_tail_worlds_per_block`| `solver_phoenx_kernels.py`                        | wpb tier (2/4/8) by `num_worlds`                 |
| `PHOENX_USE_GREEDY_COLORING`        | `solver_config.py`                                | Greedy vs round-based JP                         |
| `FUSE_TAIL_MAX_COLOR_SIZE`          | `solver_config.py`                                | Single-world fused-tail hand-off threshold       |
| `FUSE_TAIL_BLOCK_DIM`               | `solver_config.py`                                | Single-world fused-tail block size               |
| `NUM_INNER_WHILE_ITERATIONS`        | `solver_config.py`                                | Single-world capture-while host unroll factor    |
| `GREEDY_MAX_COLORS`                 | `graph_coloring/graph_coloring_common.py`         | Forbidden-mask bit width (= 64)                  |
| `_SINGLEWORLD_BLOCK_DIM`            | `solver_phoenx.py`                                | Persistent-grid block size (= 256)               |
| `_PER_WORLD_COLORING_BLOCK_DIM`     | `solver_phoenx_kernels.py`                        | Multi-world per-world coloring block size (= 64) |

## Open ideas (not yet attempted)

- **Drop the `partition_data_concat` int64 write entirely** — would require updating the JP-fallback to also write `color_tags`. Saves ~1 byte/8 bytes/commit and unifies the read path. Modest win since commits are only ~3K/round.
- **Per-instance configurable `_FUSED_INNER_SWEEPS`** — let scenes opt into `4` (or even `8`) when they don't have impact-driven stacks.
- **Pack body-hot fields into one 128B-aligned struct** — `velocity`, `angular_velocity`, `inv_mass`, `inv_inertia_world`, `body_com`, `orientation` are 5+ separate gathers per body per cid today. Likely a real win on contact-heavy scenes but it's a wide refactor.
- **Reduce greedy kernel launch count** — ~82 MIS rounds per step on kapla = ~82 launches × ~5µs overhead. A persistent kernel running all rounds with global atomics + sync flags could collapse that. Cross-block sync is the main hurdle.
- **Specialise `*_iterate_multi` for revolute-only** — saves the (already cheap) joint-mode branch in the multi-sweep helper. Marginal vs. specialising the kernel-level dispatch.

## Profiling tips

```bash
# CUDA kernel breakdown over 10 frames of kapla:
nsys profile -o my_report -t cuda --cuda-graph-trace=node --force-overwrite true \
  uv run python newton/_src/solvers/phoenx/examples/example_kapla_tower.py --num-frames=10
nsys stats --report cuda_gpu_kern_sum my_report.nsys-rep | head -25

# Multi-world bench (g1_flat / h1_flat at 3 world counts):
uv run --extra dev python -m newton._src.solvers.phoenx.benchmarks.run_benchmarks \
  --num-worlds 1024 4096 16384 --solvers phoenx -y

# Single-world / contacts: profile the singleworld iterate kernels
# (look for `_make_singleworld_persistent_kernel__locals__kernel_*` rows in nsys stats).

# Compare two `points.jsonl` runs (baseline vs after):
python3 -c "
import json
b = {(r['scenario'], r['num_worlds']): r for r in (json.loads(l) for l in open('/tmp/baseline_points.jsonl') if l.strip()) if r['solver']=='phoenx'}
a = {(r['scenario'], r['num_worlds']): r for r in (json.loads(l) for l in open('newton/_src/solvers/phoenx/benchmarks/results/points.jsonl') if l.strip()) if r['solver']=='phoenx'}
for k in sorted(a):
    if k in b:
        d = (a[k]['env_fps']-b[k]['env_fps'])/b[k]['env_fps']*100
        print(f'{k[0]:<10} {k[1]:>6} {b[k][\"env_fps\"]:>14,.0f} {a[k][\"env_fps\"]:>14,.0f} {d:>+6.1f}%')
"
```

When investigating a regression: kapla and h1_flat at 4096-16384 worlds give the cleanest signal. Run one or two scenes for ~5-10s; ignore the first ~2s of any nsys capture (kernel compilation warmup).
