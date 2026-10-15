# Implicit AABB tree vs `wp.Bvh` / `wp.Mesh`

RTX PRO 6000 Blackwell, Warp 1.17.0. Code: `newton/_src/geometry/implicit_aabb_tree.py`
(`ImplicitAABBTree`) and `implicit_triangle_tree.py` (`ImplicitTriangleTree`).
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
  slots (`default_slack`: 1.75 below 2^17 items, 1.15 from 2^20) leave room for such splits.
* Isotropic key scaling (one scale for all axes; per-axis scaling wrecks flat scenes).
* Refit is one launch: 32-leaf warps reduce their subtree's five lowest levels with shuffles, the
  rest climbs with an arrival mark in the parent record (atomic exchange; no counter array).
* Node loads are two 16-byte `__ldg` loads; build scratch lives in the node buffer.
* Queries: an AABB iterator, and fused front-to-back traversals (closest ray hit, every ray hit,
  nearest item, all overlapping pairs) that call a user function inline; stackless with a one-bit
  per level trail. `ImplicitTriangleTree` adds ready-made ray and closest-point queries on meshes
  with `wp.mesh_query_*` conventions.

## Scoreboard (best implicit vs best Warp; >1 = implicit faster or smaller)

| category | 1k | 10k | 100k | 1M |
|---|---|---|---|---|
| build (vs `Bvh.rebuild("lbvh")`) | 1.06-1.17x | 1.24-1.29x | 1.14-1.20x | 1.83-1.97x |
| refit | 1.40-1.58x | 1.50-1.75x | 1.20-1.29x | 1.04-1.22x |
| AABB query small / medium | 4.3-5.7x | 3.9-4.7x | 4.4-4.8x | 4.3-5.1x |
| ray, every hit (long / short) | 1.67 / 1.84x | 1.80 / 1.91x | 1.48 / 1.67x | 1.36 / 1.48x |
| ray, closest box | 2.22x | 2.94x | 3.94x | 5.67x |
| ray, closest triangle (vs `wp.Mesh` SAH) | 0.89x | 1.03x | 1.08x | 1.02x |
| closest point on triangles (vs `wp.Mesh` SAH) | 0.67x | 1.06x | 1.33x | 1.33x |
| self overlap pairs | 0.92x | 0.99-1.10x | 2.2-2.7x | 3.3-4.4x |
| memory | 0.65-0.71x | 0.65-0.71x | 0.65-0.71x | 0.97-1.06x |

Raw numbers: `results.json` / `bench_bvh.log`. Broad phase (`bench_broadphase.py`): identical pair
sets, 3-4% faster than before the tree was factored out.

## Verdict

The implicit tree is the better default for GPU use: it builds 1.1-2x faster than the fastest
`wp.Bvh` (and >1000x faster than SAH/cuBQL, which build on the CPU), refits faster, answers box
queries 4-6x faster and rays and closest-point queries as fast or faster from 10k items up, with
no tuning. It loses only on very small meshes (~1k triangles), where a CPU-built SAH tree is
tighter, and spends ~1.4x the memory of `wp.Bvh` below ~100k items for its spare slots (parity at
1M).
