# PhoenX Warp Native Extensions

PhoenX uses `wp.func_native` only where Warp does not expose the CUDA primitive
or where a very small bounds-check-free helper has been measured on hot paths.
Keep new native snippets tiny, GPU-scoped, and documented here.

## Synchronization Intrinsics

- `solver_phoenx_kernels._sync_threads`: wraps `__syncthreads()` for generated
  fast-tail kernels whose block-local phases cannot be expressed with a Warp
  builtin.
- `solver_phoenx_kernels._sync_warp` and `_sync_warp_mask`: wrap
  `__syncwarp()` for one-warp-per-world fast-tail kernels. The block size is
  kept at an integer number of warps so the synchronization scope is explicit.

## Warp Vote and Shuffle Intrinsics

- `benchmarks/experimental/bench_lock_scheduled_pgs._warp_shfl_i32`,
  `_warp_shfl_u32`, `_warp_ballot`, and `_warp_popc`: experimental wrappers
  used by the one-warp-per-world no-coloring PGS feasibility benchmark. They
  let lanes propose constraint rows and greedily accept a body-disjoint
  micro-wave without global atomics.
- `benchmarks/experimental/bench_color_grid_actual_solve` defines the same
  shuffle/ballot wrappers for an actual-solve no-color scheduler experiment,
  including a refill/tile-stack variant. These are benchmarking entry points
  only; current real-kernel measurements do not justify production use.

## Bit Operations

- `graph_coloring_common._lowest_set_bit`: wraps CUDA `__ffsll()` for greedy
  coloring's 64-bit free-color mask. The intrinsic did not materially change
  performance, but keeps the first-set-bit operation direct and readable.

## Raw Array Accessors

- `helpers.array_access.read1d_i32`, `read2d_f32`, and `write2d_f32`: raw
  pointer accessors for contiguous hot arrays when the caller has already
  proven indices in range. They avoid Warp's bounds-check wrapper and add
  `__restrict__` to help nvcc alias analysis. Do not use them for mutable PGS
  state unless the same kernel never writes the same storage through another
  path.

## Timing

- `timer.read_global_timer_ns`: reads CUDA `%globaltimer` for optional
  per-constraint profiling. This is GPU-only instrumentation and should stay
  off the default hot path.
