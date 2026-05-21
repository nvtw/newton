# PhoenX constraint clustering

Reference notes for the cluster-aware PGS path in PhoenX. Adapts
Algorithm 4 of Ton-That, Kry & Andrews 2022 (*Parallel Block
Neo-Hookean XPBD using Graph Clustering*) to a fully-parallel,
graph-capture-safe, deterministic GPU implementation.

## Hard limits

| Name                          | Value         | Where                                                 | Why                                                       |
| ----------------------------- | ------------- | ----------------------------------------------------- | --------------------------------------------------------- |
| `MAX_CLUSTER_SIZE`            | 4 constraints | `clustering/cluster_builder.py`                       | Member ids pack into a single `vec4i`; `-1` = unused slot |
| `MAX_BODIES_PER_CLUSTER`      | 8 bodies      | `clustering/cluster_builder.py`                       | Cluster body union must fit in `vec8i` (= 1 EID struct)   |
| `MAX_BODIES` (per constraint) | 8             | `graph_coloring/graph_coloring_common.py`             | `ElementInteractionData.bodies: vec8i` (re-widened 6→8)   |

## Algorithm (Ks=4 → 3 → 2 → 1 cascade)

1. **Adjacency build**: standard vertex-to-element CSR
   (`adjacency.ElementVertexAdjacency`, shared with graph coloring).
2. **For Ks_target in 4, 3, 2, 1**:
   a. **propose**: each uncommitted element becomes a seed iff it's
      strictly priority-max among uncommitted 1-hop neighbours (MIS
      over `packed_priorities`).
   b. **grow**: each fresh seed atomic-min-claims every 1-hop
      uncommitted neighbour. No body-cap check here, no early
      termination. Final `pending_id_of[n]` = min seed id that
      contested n.
   c. **validate**: each fresh seed finds the 3 smallest-id confirmed
      members (those with `pending_id_of[n] == seed`), then admits
      them in id-sorted order subject to the 8-body cap. Non-admitted
      claims released back to `_EMPTY`. Commit iff confirmed ≥
      Ks_target. Ks_target=1 commits singletons unconditionally.
3. **Compact**: scan over the seed bitmap → dense `[0, num_clusters)`
   cluster ids → per-element `element_to_cluster` map and a `vec4i`
   per cluster with members sorted ascending (`-1` padded).

## Determinism

The output is bit-identical across runs given the same input + same
priority seed. Key invariants:

- `propose` is a pure function of `packed_priorities` (unique
  permutation, host-seeded).
- `grow`'s race outcomes are order-independent: `atomic_min` on seed
  id always picks the lower-id seed regardless of thread schedule.
- `validate` picks the 3 smallest-id confirmed members (insertion sort
  in 3 register slots) — independent of which order they were claimed
  in grow.
- `vec4i` member emission bubble-sorts via `atomic_min` so slot
  ordering is ascending.

No per-thread state depends on the sequence of atomic outcomes; all
decisions are made against the post-settled `pending_id_of` array.

## Graph-capture safety

Every kernel launch uses a fixed dim (`max_num_interactions` or
`max_num_nodes`); active-count gating via `tid >= num_elements[0]`.
The outer Ks_target schedule is a static host loop (4 iterations).
The inner round loop uses `wp.capture_while` with a hard
`_MAX_OUTER_ITERS = 32` cap on iterations and `_INNER_UNROLL = 4`
unrolled `propose→grow→validate→commit→reset` rounds per body
invocation.

## Files

```
phoenx/
├── adjacency.py                            # vertex→adjacent-element CSR (shared with graph coloring)
├── clustering/
│   ├── cluster_builder.py                  # ConstraintClusterBuilder (the algorithm above)
│   ├── supernodal_elements.py              # SupernodalElements (emit per-cluster ElementInteractionData)
│   └── clustering_pipeline.py              # ClusteringPipeline composition
├── solver_phoenx_kernels.py                # cluster_aware static axis on persistent + fused PGS kernels;
│                                           #   _make_singleworld_dispatch_func extracts the per-cid tree
└── solver_phoenx.py                        # enable_clustering flag + _cluster_aware_active gate
```

## Solver integration

- `PhoenXWorld(enable_clustering=True)` (or `SolverPhoenX(..., enable_clustering=True)`).
- Gate: `_cluster_aware_active = enable_clustering AND step_layout == "single_world"`.
- When active: main partitioner colours the supernodal graph (CSR
  slots hold cluster ids, not constraint ids). PGS iterate kernels
  read `cluster_members[cluster_id]` (vec4i) and run the standard
  per-cid dispatch tree once per non-`-1` member.
- When not active: per-constraint partitioning + dispatch, byte-for-
  byte identical to pre-clustering behaviour. A 1-element placeholder
  `cluster_members` is passed but never read (static `wp.static(cluster_aware)`).

## Mass splitting

Composes for free. The mass-splitting overflow column's `parallel_id`
is computed per CSR slot, so all members of a cluster share the same
`parallel_id` (Gauss-Seidel within the cluster). Different clusters
in the overflow column own different `parallel_id`s (Jacobi across
clusters).

`_rebuild_mass_splitting_graph` reads from `supernodal_elements` and
gates on `num_clusters` when cluster-aware. The body union naturally
deduplicates per-cluster `(body, partition_key)` emits, allocating
exactly one copy-state slot per (cluster body, partition_key).

## Constraint types

Type-agnostic: clusters mix any combination of contact, soft-tet,
cloth-triangle, cloth-bending, joint (revolute / ADBS). The cluster
builder operates on `ElementInteractionData.bodies` alone. The
PGS dispatch tree is the same per-cid function used in the per-
constraint path (extracted into `_make_singleworld_dispatch_func`).

## Known gaps

- **Multi-world layout**: `step_layout="multi_world"` not yet wired
  for clustering. Per-world coloring uses different kernels.
- **`SolverPhoenX` model adapter**: the Newton-model wrapper doesn't
  auto-populate soft-tetrahedron constraints from `add_soft_grid`.
  `example_soft_body_drop` builds `PhoenXWorld` directly, so
  `--enable-clustering` works there. Soft-tet auto-population from
  Newton models is orthogonal model-adapter work.

## Enabling and benchmarking

```bash
# Example with clustering on
uv run python -m newton._src.solvers.phoenx.examples.example_soft_body_drop \
    --headless --enable-clustering

# Dump a constraint-graph snapshot at frame N
PHOENX_DUMP_COLORING_GRAPH=20 uv run python -m \
    newton._src.solvers.phoenx.examples.example_soft_body_drop \
    --headless --num-frames 25

# Compare baseline vs clustered coloring + per-stage timings
uv run python -m newton._src.solvers.phoenx.benchmarks.bench_clustering \
    --snapshots soft_body_drop_graph.npz --repeats 5
```

## Tests

CUDA only, graph-capture path. Per repo memory, run via:

```bash
uv run --extra dev -m newton.tests -k test_cluster_builder
uv run --extra dev -m newton.tests -k test_supernodal_elements
uv run --extra dev -k test_clustering_pipeline
```

Coverage: cluster algorithm unit tests (`test_cluster_builder` — 11
cases), supernodal emit (`test_supernodal_elements` — 8 cases),
end-to-end captured-graph physics validation
(`test_clustering_pipeline` — 7 cases incl. determinism, baseline
equivalence, and mass-splitting composition).
