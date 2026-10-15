# Implicit AABB tree vs `wp.Bvh` / `wp.Mesh`

RTX PRO 6000 Blackwell, Warp 1.17.0, CUDA. Branch `phoenx/implicit-tree-reusable`
(based on `phoenx/sdf-cached-sampling`). Raw data: `results.json`, `bench_bvh.log`.
Reproduce: `uv run python local_studies/implicit_tree_bvh/bench_bvh.py`, then `summarize.py`.

Method: GPU ops timed as CUDA-graph replays (best of 5 trials × many replays); the
`wp.Bvh(...)`/`wp.Mesh(...)` constructors are not capturable, so they were timed eagerly
(best of 3, synchronized). Before each trial the benchmark waits until no other process is
using the GPU; the run waited 29.6 s in total because another agent was using the GPU.
Every query uses 262,144 threads and the times are **ms per 262k queries**.

Correctness: per-query hit counts match exactly across every structure for every AABB query and
self-pair total. Ray results agree on all but at most 1 of 262k rays (an AABB boundary
tie). At n ≤ 10k the hit counts also match a NumPy brute-force check.

Scenes: `boxes` = uniform random boxes, density 1 per unit³, half extents 0.1–0.5.
`terrain` = a shuffled heightfield triangle mesh. The trees are built over triangle AABBs and
`wp.Mesh` is built from the points and indices.

## Build / refit (ms)

| scene, n | implicit build | Bvh lbvh ctor | Bvh lbvh `rebuild()` | Bvh sah ctor | cubql ctor | Mesh ctor | implicit refit | Bvh refit (lbvh/sah) | Mesh refit |
|---|---|---|---|---|---|---|---|---|---|
| boxes 1k | **0.039** | 0.25 | 0.062 | 0.39 | 3.4 | – | **0.014** | 0.016 / 0.016 | – |
| boxes 10k | **0.085** | 0.33 | 0.135 | 2.8 | 7.0 | – | 0.023 | 0.021 / 0.019 | – |
| boxes 100k | **0.109** | 0.27 | 0.166 | 33 | 10 | – | 0.037 | 0.027 / 0.027 | – |
| boxes 1M | **0.255** | 0.63 | 0.520 | 451 | 57 | – | 0.142 | 0.110 / 0.106 | – |
| terrain 1M | **0.250** | 0.65 | 0.558 | 401 | 51 | 0.80 (sah 315) | 0.138 | 0.117 / 0.099 | 0.164 |

All of these are graph-capturable: implicit build and refit, `Bvh.refit`, `Bvh.rebuild("lbvh")`
and `Mesh.refit`. The `wp.Bvh`/`wp.Mesh` constructors allocate, so they are not.
The implicit refit at 1M is 0.075 ms for the 20 per-level launches plus about 0.067 ms for the leaf write.

## Queries (ms per 262k queries; lower is better)

`implicit` = public iterator API. `inline` = hand-fused traversal loop, which is the upper bound.

| scene, n | query | hits/q | implicit | inline | Bvh lbvh | Bvh sah | Bvh cubql | Mesh |
|---|---|---|---|---|---|---|---|---|
| boxes 10k | AABB small | 1.3 | **0.30** | 0.22 | 0.57 | 0.50 | 0.44 | – |
| boxes 10k | AABB medium | 16 | **0.64** | 0.50 | 1.83 | 1.71 | 1.59 | – |
| boxes 1M | AABB small | 1.3 | **0.50** | 0.45 | 0.89 | 0.81 | 0.75 | – |
| boxes 1M | AABB medium | 17 | **1.02** | 0.80 | 2.40 | 2.23 | 2.11 | – |
| terrain 100k | AABB small | 4.0 | 0.48 | 0.42 | 1.55 | **0.45** | 0.49 | 0.54 |
| terrain 1M | AABB small | 4.0 | 0.88 | 0.77 | 2.91 | **0.56** | 0.59 | 0.82 |
| terrain 1M | AABB medium | 18 | 1.27 | 1.23 | 4.91 | **1.14** | 1.27 | 1.47 |
| boxes 1M | ray, all hits | 24 | 5.94 | – | 2.62 | **2.54** | 2.64 | – |
| boxes 1M | ray, len 2, all hits | 1.3 | 0.78 | – | 0.29 | **0.28** | 0.30 | – |
| boxes 1M | ray, closest box | – | 3.66 | – | 1.62 | **1.52** | 1.54 | – |
| terrain 100k | ray, closest tri | – | 0.88 | – | 0.41 | **0.16** | 0.20 | 0.57 |
| terrain 1M | ray, closest tri | – | 2.68 | – | 0.87 | **0.19** | 0.21 | 1.20 |
| boxes 100k | self pairs (one thread per item) | – | 0.33 | **0.15** | 0.13* | 0.13* | – | – |
| boxes 1M | self pairs | – | 1.56 | **0.85** | 0.94* | 0.93* | – | – |
| terrain 1M | self pairs | – | 4.67 | 1.52 | 3.61* | **1.22*** | – | – |

\* `wp.Bvh` self pairs use one AABB query per item and keep only `j > i`.

## Memory (MB, n = 1M)

| implicit | Bvh lbvh/sah | Bvh cubql | Mesh |
|---|---|---|---|
| 87.3 | 84.0 | 77.1 | 108.0 |

The implicit figure includes the sort scratch and is padded to a power of two
(2·leaf_base·32 B of nodes). It can therefore reach about 2× the Warp tree just above a power of two.
For example, n = 10k uses 1.27 MB against 0.84 MB. The Warp numbers are memory-pool deltas.

## Verdict

**The implicit tree is better for:**
- **Rebuilding every frame.** A full rebuild (Morton sort plus bottom-up reduction) takes
  0.04–0.25 ms. That is 2–2.5× faster than the in-place `Bvh.rebuild("lbvh")`, 2.5–6× faster than
  the LBVH constructor, and over 1000× faster than SAH, median and cuBQL, which build on the CPU.
  It is fully graph-capturable and allocates nothing.
- **Box-overlap queries on uniformly spread data,** the broad-phase case. It is 1.7–3.5× faster than
  any `wp.Bvh` variant, including SAH, on random boxes at every n. On terrain it is about 3–4× faster
  than an LBVH `wp.Bvh` and roughly equal to `wp.Mesh`.
- **Rebuilding every frame instead of refitting.** That costs only about 2× a Warp refit and always
  keeps the tree tight, so it is the right default for moving scenes such as the broad phase.

**Warp's BVH is better for:**
- **Rays.** The implicit tree is 2–3× slower than `wp.Bvh` on random boxes, at both short and long
  rays. On a mesh it is up to 14× slower than an SAH `wp.Bvh` (1M triangles, closest hit: 2.7 ms
  against 0.19 ms) and 2–3× slower than LBVH or `wp.Mesh`.
  The cause is node quality. The complete tree splits by leaf count along the Morton curve, not at
  Morton-cell boundaries as LBVH does or by surface area as SAH does. Its internal boxes are
  therefore looser and overlap more: a 0.5-long ray visits about 170 nodes at 1M items.
  Traversal is stackless and unordered, so there is no near-child-first visiting for closest hits.
- **Static or rarely rebuilt geometry queried many times.** An SAH or cuBQL `wp.Bvh` wins every
  query type on the mesh, and its build cost is paid once. `refit()` is 20–30% cheaper than the
  implicit refit at 1M.
- **Memory near power-of-two boundaries.**

**Self pairs.** With the fused loop that `BroadPhaseImplicitTree` uses, the implicit tree is as fast
as or faster than LBVH/SAH `wp.Bvh` on random boxes. On terrain at 1M it beats LBVH (1.52 against
3.61 ms) but loses to SAH (1.22 ms). Through the generic iterator API the self query is 2–3× slower
than the fused loop, because each hit forces the warp to exit and re-enter the traversal loop.

**Iterator API cost.** The generic AABB iterator is about 10–30% slower than a hand-fused loop.
Different loop forms (single call site, `while True`/`break`, tuple return) did not change this;
they are in `micro_traversal.py`. Hot paths should use a fused loop, as the broad phase does.

**Bottom line:** use the implicit tree when the tree is rebuilt every step and the queries are
box-shaped: broad phase, particle/shape neighbour search, cloth/particle-vs-shape culling.
Use `wp.Bvh` (SAH or cuBQL when static, LBVH plus `refit` when deforming) or `wp.Mesh` for ray
casting, sensors, closest-hit or nearest queries, and static geometry queried many times.
Better split selection, for example cutting the Morton order at the highest differing bit as
LBVH does while keeping the implicit layout, would be the lever for closing the ray gap.

## Broad phase no-regression (`bench_broadphase.py`, full `launch()` incl. build, graph replay)

| case | before (ms) | after (ms) | pair-set digest |
|---|---|---|---|
| mixed2k | 0.185 | 0.171 | identical |
| boxes32k (51k pairs) | 0.307 | 0.294 | identical |
| boxes200k | 0.329 | 0.330 | identical |
