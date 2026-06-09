# PhoenX CUDA Scheduler Microbench

This directory contains standalone CUDA experiments for PhoenX scheduling ideas.
The benchmarks model colored per-world PGS work without Warp codegen or CUDA
graph capture, so scheduler overhead is easier to isolate.

Build:

```bash
make -C newton/_src/solvers/phoenx/benchmarks/cuda_scheduler
```

Run:

```bash
newton/_src/solvers/phoenx/benchmarks/cuda_scheduler/scheduler_microbench \
  --scenes h1,g1,dr_legs,tower --worlds 32 --epochs 4
```

The benchmark compares:

- `fast_tail`: one production-like lane group per world, looping colors in order.
- `tile`: one cooperative block owns `--worlds-per-block` worlds and shares the
  block's threads across that tile's rows for each color.
- `chunk_chain`: persistent worker blocks that process row chunks and publish the
  next color for a world when the current color's chunks finish.

Useful knobs:

- `--worlds`: use small values for under-occupied single/few-world behavior and
  large values for RL-style full-GPU occupancy.
- `--work-iters`: synthetic row payload. Larger values make scheduling overhead
  less dominant.
- `--imbalance`: deterministic per-world/color row-count variation in percent;
  useful for modeling tail effects.
- `--worlds-per-block`: tile size for the cooperative block scheduler.
- `--run-chain 0`: skip the experimental persistent chunk-chain scheduler.

Recent signal on RTX PRO 6000 Blackwell: for 2048-world h1/g1/dr_legs/tower
synthetic graphs with 25% imbalance, the production-like `fast_tail` model was
faster than both `tile` and `chunk_chain`. The single-world tower-like case was
different: `tile` beat `fast_tail` in the synthetic benchmark, and the real
`bench_singleworld_tail_fuse` tower run showed the existing fused-tail path
improving solve time by roughly 1.19x over no tail fusion. This points toward a
hybrid policy: keep fast-tail for full-GPU RL fleets, and invest scheduler work
in cooperative/fused paths for under-occupied large worlds. The single-world
tower-like case can make `chunk_chain` fail under bounded spins; the benchmark
reports that as `nan` instead of aborting so sweeps can continue.

The scene row counts are synthetic PhoenX-shaped distributions. They are meant
for scheduler research, not solver correctness or quality evaluation.
