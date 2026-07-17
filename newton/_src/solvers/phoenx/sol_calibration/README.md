# PhoenX speed-of-light calibration

These local CUDA microbenchmarks establish hardware ceilings for interpreting
PhoenX profiles. They are not part of CI or the ASV benchmark suite.

Run the complete suite:

```bash
uv run -m newton._src.solvers.phoenx.sol_calibration
```

The default data array is 256 MiB, so sequential and random-access measurements
exceed typical GPU caches. Peak allocation is roughly three times `--array-mib`.
Use a power-of-two size for the random-access permutation. A fast compilation
and execution check is available with `--quick`.

Each benchmark can also be run separately:

```bash
uv run -m newton._src.solvers.phoenx.sol_calibration.bench_memory
uv run -m newton._src.solvers.phoenx.sol_calibration.bench_random_access
uv run -m newton._src.solvers.phoenx.sol_calibration.bench_compute
```

For stable peak numbers, stop other GPU work and lock the GPU clocks. The tables
show both the best-trial throughput and median-trial throughput. Sequential copy
counts source reads plus destination writes. Indexed gather additionally counts
the streamed 32-bit index array. These are logical byte rates rather than memory
controller counters.
