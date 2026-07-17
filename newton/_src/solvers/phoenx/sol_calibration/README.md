# PhoenX speed-of-light calibration

These local CUDA microbenchmarks establish hardware ceilings for interpreting
PhoenX profiles. They are not part of CI or the ASV benchmark suite.

Run the complete suite:

```bash
uv run -m newton._src.solvers.phoenx.sol_calibration
```

The memory suite includes the `bfloat16` scalar type used by PhoenX RL as well as
`float`, `vec2`, and `vec4`. The Tensor Core suite mirrors RL's 128 x 16 x 32
tile shape with BF16 inputs and FP32 accumulation.

The default data array is 256 MiB, so sequential and random-access measurements
exceed typical GPU caches. Peak allocation is roughly three times `--array-mib`.
Use a power-of-two size for the random-access permutation. A fast compilation
and execution check is available with `--quick`.

Each benchmark can also be run separately:

```bash
uv run -m newton._src.solvers.phoenx.sol_calibration.bench_memory
uv run -m newton._src.solvers.phoenx.sol_calibration.bench_random_access
uv run -m newton._src.solvers.phoenx.sol_calibration.bench_compute
uv run -m newton._src.solvers.phoenx.sol_calibration.bench_tensor
```

For stable peak numbers, stop other GPU work and lock the GPU clocks. The tables
show both the best-trial throughput and median-trial throughput. Sequential copy
counts source reads plus destination writes. Indexed gather additionally counts
the streamed 32-bit index array. These are logical byte rates rather than memory
controller counters. Known GPUs also show published memory, FP32, and dense
BF16 Tensor Core ceilings with best-trial utilization. Use the corresponding
`--memory-peak-gbps`, `--fp32-peak-gflops`, or
`--bf16-tensor-peak-gflops` option to override these ceilings.
