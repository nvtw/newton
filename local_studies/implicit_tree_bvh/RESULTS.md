# Implicit BVH vs `wp.Bvh` / `wp.Mesh`

RTX PRO 6000 Blackwell, Warp 1.18.0 (NVRTC 13.4). Code: `newton/_src/geometry/implicit_bvh.py`
(`newton.geometry.ImplicitBvh`) and `implicit_bvh_mesh.py` (`ImplicitBvhMesh`).
Reproduce: `uv run python local_studies/implicit_tree_bvh/bench_bvh.py`, then `scoreboard.py`.
GPU work is timed as CUDA-graph replays (best of several trials) with the GPU otherwise idle;
`wp.Bvh` / `wp.Mesh` constructors are not capturable and are timed eagerly. Queries are 262,144
threads; times are ms per launch. Every query's results are cross-checked between all structures
(and against brute force for n <= 10k). Scenes: uniform random boxes, and a shuffled heightfield
mesh (`terrain`). Warp structures: `wp.Bvh` (LBVH, SAH, cuBQL) and `wp.Mesh` (LBVH, SAH).

## Design

* Complete binary heap with `leaf_base` leaves over two levels (no child pointers, no padding to a
  power of two). Build = Hilbert keys + radix sort + one split launch per level + one refit launch.
* Leaves follow a Hilbert curve; every subtree is split where its keys change in the highest bit
  (a curve cell boundary, as in an LBVH), clamped to the subtree's leaf capacity. A few spare leaf
  slots (default: 1.75 below 2^17 items, 1.15 from 2^20) leave room for such splits.
* Isotropic key scaling (one scale for all axes; per-axis scaling wrecks flat scenes).
* Refit is one launch: 32-leaf warps reduce their subtree's five lowest levels with shuffles, the
  rest climbs with an arrival mark in the parent record (atomic exchange; no counter array).
* Node loads are two 16-byte `__ldg` loads; build scratch lives in the node buffer.
* Queries: an AABB iterator, closest box along a ray / to a point, and fused front-to-back traversals (closest ray hit, every ray hit,
  nearest item, all overlapping pairs) that call a user function inline; stackless with a one-bit
  per level trail. `ImplicitBvhMesh` adds ready-made ray and closest-point queries on meshes
  with `wp.mesh_query_*` conventions.

## Scoreboard (best implicit vs best Warp; >1 = implicit faster or smaller)

| category | 1k | 10k | 100k | 1M |
|---|---|---|---|---|
| build (vs `Bvh.rebuild("lbvh")`) | 1.05-1.17x | 1.25-1.28x | 1.13-1.20x | 1.82-1.96x |
| refit | 1.22-1.29x | 1.33-1.58x | 1.17-1.20x | 1.09-1.14x |
| AABB query small / medium | 2.0-2.4x | 1.3-2.1x | 1.3-1.9x | 1.4-1.7x |
| ray, every hit (long / short) | 1.93 / 1.85x | 1.69 / 1.67x | 1.37 / 1.39x | 1.19 / 1.15x |
| ray, closest box | 2.15x | 2.44x | 3.40x | 4.70x |
| ray, closest triangle (vs `wp.Mesh` SAH) | 0.97x | 1.24x | 1.32x | 1.24x |
| closest point on triangles (vs `wp.Mesh` SAH) | 0.75x | 1.03x | 1.25x | 1.23x |
| self overlap pairs | 1.09-1.23x | 1.41-1.47x | 2.7-3.8x | 4.3-6.2x |
| memory | 0.65-0.71x | 0.65-0.71x | 0.65-0.71x | 0.97-1.06x |

Raw numbers: `results.json` / `bench_bvh.log`. Broad phase (`bench_broadphase.py`): identical pair
sets, 3-4% faster than before the tree was factored out.

## Profiling (Nsight Compute, `profile_implicit_bvh_ncu.sh` + `ncu_driver.py`)

* No query is memory bound (DRAM <= 6%, L1 hit 50-99%); all run ~5-10 of 32 lanes per warp.
* Small trees are issue bound: at 1k triangles the closest-point kernel issued 1.9x the
  instructions of `wp.Mesh` (7x the `MOV`s, 2x the branches) at equal IPC. Large trees are
  latency bound, where `wp.Mesh` stalls twice as long on loads.
* Ruled out by A/B: leaf size (`wp.Mesh` with 1 triangle per leaf is as fast as with 4), SAH split
  placement along the curve, a full SAH order with SAH splits, stack traversal, a smaller node
  footprint (less slack). An unconstrained SAH tree needs only ~20% fewer visits at 1k.
* What helped (same toolchain, old -> new): jumping from a finished subtree to the deepest pending
  sibling in one step (trail bits + `clz`) instead of climbing per level, the same O(1) step for the
  AABB/pair successor, testing two leaf children in place, strict pruning that lets the inverted
  bounds of empty slots reject themselves, and a single-exit triangle distance. Speedups:
  closest point 1.34-1.49x, rays 1.02-1.36x, pairs 1.18-1.41x, AABB 1.02-1.25x.
* Toolchain: the same generated CUDA runs 1.4-1.6x slower on rays and closest points with Warp
  1.18's NVRTC 13.4 than with Warp 1.17's NVRTC 12.9 (old code: closest point 0.40 -> 0.59 ms at
  100k). Compile flags, module options and the relevant native headers are identical; loading the
  PTX that Warp 1.17 compiled from the same source into Warp 1.18 restores the old speed (box rays
  0.541 -> 0.340 ms at 10k), so the cause is NVRTC 13.4 code generation. The scoreboard above is on
  1.18.
* Node loads: kernels read nodes through typed `func_native` helpers (two 16-byte `__ldg` loads
  from the `wp.array`, no raw pointer in `ImplicitBvhData`). A plain `wp.array` element read of a
  vec4/struct is split into 4-byte loads (Warp's vector types are 4-byte aligned) plus a
  negative-index wrap; the broad phase used such reads and got 7-13% faster with the helper.
  Plain (non-`.nc`) loads win dense box rays by up to 15% on NVRTC 13.4 but lose pairs and
  triangle rays by 6-13%. The NVRTC 13.4 gap is unchanged by any load variant (box rays
  1.14-1.41x), so it is in the traversal code generation, not the loads.

## Verdict

The implicit tree is the better default for GPU use: it builds 1.1-2x faster than the fastest
`wp.Bvh` (and >1000x faster than SAH/cuBQL, which build on the CPU), refits faster, answers box
queries 1.3-2.4x faster, closest-box rays 2-5x faster, and triangle rays and closest-point queries
as fast or faster from 10k items up, with no tuning. It loses only on very small meshes (~1k
triangles; closest point 0.75x), where a CPU-built SAH tree is tighter and query cost is
instruction bound, and spends ~1.4x the memory of `wp.Bvh` below ~100k items for its spare slots
(parity at 1M).
