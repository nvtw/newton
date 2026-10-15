# Plan: cached + importance-sampled SDF-SDF collision ("SDF-DCD") in Newton

Source: Giles & Andrews, *Real-Time Collision Handling for Signed Distance Fields via
Caching and Importance Sampling*, Pacific Graphics 2026 (short paper),
reference code `~/Documents/git/sdf-dcd-demo/source/{collide,optimize,shapes,voxel,solver}.cpp`.

Goal: a second, selectable implementation of the rigid SDF contact path
(`shape_pairs_mesh_mesh` today), chosen per `CollisionPipeline`, production quality,
compared against the current edge-sampling path on PhoenX scenes. Hydroelastic is untouched.

---

## 1. The method (what we must reproduce)

Per overlapping shape pair (A, B), every collide call (Alg. 1):

1. **Domain** `V = AABB_A ∩ AABB_B`, both grown by half the contact margin (Sec. 3.1).
2. **Objective** (Eq. 2–3), minimised over `x ∈ V`:
   `g(x) = ½(φA + φB + sqrt((φA − φB)² + ε)) − α Σ_{m∈M} ‖x − x_m‖`.
   The soft-max is a C¹ upper bound of `max(φA, φB)`, so it is lowest deep inside the overlap. The repulsion term spreads points over the manifold.
   Defaults: ε = 0.1, α = 0.01, τ = 0.01, with models normalised to unit size.
   **These must become scale-relative in Newton** (ε has units of length², α is dimensionless, τ is a length).
3. **Seeds**, two sources:
   - **Cache.** Up to `n_cache` (=4) points from the previous step, advected with the average point velocity of A and B (Eq. 6), or only the moving body's velocity if the other is static.
   - **Stochastic.** `n_rand` (=1) samples plus enough to top the cache back up to `n_cache`. Each is drawn from the precomputed high-curvature seeds of A or B that lie inside V, with a uniform-in-V fallback (Sec. 3.5.2).
4. **Optimiser:** Nelder–Mead, derivative-free (the paper's best choice for non-convex shapes).
   - Initial simplex is axis-aligned with size 0.1·|V|, clamped to V.
   - For cached points the simplex is shrunk to the advection distance, or 5τ if that is larger (demo, not in the paper). This stops cached points wandering on flat faces.
   - Stop when the simplex extent is below τ, the value spread is below 0.05τ, or after 50 iterations.
5. **Contact data** (Eq. 5):
   - At x*, take φ and ∇φ of both shapes.
   - Depth and normal come from the deeper field. The normal is ∇φA, or −∇φB if B is deeper, so it always points A→B.
   - Demo extras that matter for stability:
     - *Normal coherence:* a cached point keeps whichever of nA / −nB is closer to its previous normal. This avoids flips on thin spikes.
     - *Separation:* use a Newton-step closest-point gap `gap = ∇φA·φA/|∇φA|² − ∇φB·φB/|∇φB|²` when x* is outside either body. Outside both, the normal is `gap/|gap|`, unless that would reverse a cached normal.
     - `active = separation < margin`, so speculative skin contacts are emitted too.
6. **Dedup** points within τ, cached points first (Sec. 3.7).
7. **Rank** by `g` against the final manifold.
   - New samples pay a `replacementThreshold` (0.01) penalty (demo). This keeps flat contacts from swapping points every frame, which would destroy the warm start.
   - Keep the best `n_cache` as the next cache.
8. **Curvature seeds** (Eq. 7), built offline:
   - Take surface samples where `|Δφ| > κ` (κ = 4/L, δ = 0.001·L), Laplacian by central differences of ∇φ.
   - Thin with a Poisson disk of spacing 0.08·L.

Paper results that matter here:
- About 4 contacts per pair (vs 184 max/pair in our reducer). Collision time scales linearly with `n_cache + n_rand`.
- The cache is the biggest win in every ablation (no cache: 3–4× more iterations, and SGD fails outright).
- Curvature sampling mainly cuts transient penetration and rest-state KE.
- n_cache = 4 is "stable". n_cache = 16 approaches the dense-sampling oracle for penetration.

## 2. How it maps onto Newton (decisions)

| Concern | Today (edge path) | New path | Reuse |
|---|---|---|---|
| Pair routing | `narrow_phase.py:701-718` → `shape_pairs_mesh_mesh` | Same routing. A new `wp.static` switch sends pairs where **both** shapes have a texture SDF to `shape_pairs_sdf_dcd`. BVH-only and heightfield pairs stay on the edge kernel. | routing kernel, pair buffers |
| Hydroelastic | routed earlier at `:632-639` | untouched (routing happens before the new branch) | — |
| SDF evaluation | `texture_sample_sdf_hw*`, `grad_only` | Optimiser loop: `texture_sample_sdf_hw` (1 fetch). Final contact: `texture_sample_sdf_grad` (exact trilinear value + analytic gradient). | `sdf_texture.py` samplers |
| Frames / scale | `sdf_contact.py:123,157,1228` | Same helpers. World→SDF-local per shape, scale via `safe_sdf_scale_inverse` / `scale_sdf_result_to_world`. | as-is |
| Contacts/pair | ≤184, via GlobalContactReducer | ≤ `n_cache + n_rand` (default 5). **No reducer**: direct `writer_func`. | `write_contact` (collide.py:290) |
| Temporal state | none in narrow phase; ContactMatcher downstream | Per-pair cache (Sec. 4). | ContactMatcher's persistent-array pattern (contact_match.py:638-689) |
| Curvature seeds | — | Phase 1: reuse **existing** `mesh_edge_centers` (sharp collision edges from `_build_collision_edges`, plus box edges). These are exactly the high-curvature features, so no new data is needed. Phase 3: optional Eq. 7 seeds. | builder.py:13990-14075 |
| Matching / warm start | sort key `(edge<<2)\|mode` | `sort_sub_key = cache slot id`, which stays stable while a point survives. Cached points advect rigidly, so midpoints move far less than the 0.5 mm `pos_threshold`. | ContactMatcher unchanged |

### Deviations from the paper (deliberate, each to be validated)
- **Cache advection by pose, not velocity.**
  - Store each cached point in both bodies' local frames: `pA_loc`, `pB_loc`.
  - Advect as `x = ½(X_A·pA_loc + X_B·pB_loc)`. If one shape is static (body −1 or kinematic), use the other's frame only.
  - This is Eq. 6 to first order, but exact under rigid motion, needs no `body_qd` in the collision pipeline, and is independent of dt and of how many collide calls PhoenX makes per frame (1–4).
- **GPU parallel layout: lane-per-candidate, with frozen repulsion.**
  - In the demo, each `addPoint` repels against the points found so far, which is sequential.
  - On GPU, one tile of `T = n_cache + n_rand` lanes (padded to 8) processes one pair, and each lane runs one Nelder–Mead.
  - The repulsion set M is the *advected seed positions* of all lanes, fixed before optimisation. That makes it deterministic and needs no inter-lane sync inside the NM loop.
  - Dedup and ranking then run cooperatively in the tile (T² ≤ 64 distance checks).
  - A test-only, sequential, one-thread-per-pair kernel (a faithful port of `Manifold::update`) is kept as a reference for parity tests.
- **Units.** All tolerances are relative to the pair's characteristic length `L = min(diag(AABB_A), diag(AABB_B))`:
  - τ = `tol_rel·L`, floored at `voxel_radius`;
  - ε = `(eps_rel·L)²`;
  - the replacement threshold is relative to L.
  - Absolute overrides are allowed in the config.
- **RNG.** `wp.rand_init(hash(pair_key, frame_counter, lane))` with a device-side frame counter. This is deterministic and works under graph capture.

## 3. Public API (global switch)

```python
class CollisionPipeline:
    class SDFContactConfig:            # nested, like HydroelasticSDF.Config
        method: Literal["edge", "cached_sampling"] = "edge"
        cache_size: int = 4             # n_cache  (compile-time cap MAX_CACHE = 16)
        random_samples: int = 1         # n_rand
        curvature_sampling: bool = True
        skip_convex_samples: bool = True   # demo heuristic: convex pairs with full active cache skip n_rand
        softmax_epsilon_rel: float = 0.1   # ε = (rel·L)²
        repulsion_weight: float = 0.01     # α
        tolerance_rel: float = 0.01        # τ = rel·L, >= voxel_radius
        replacement_threshold_rel: float = 0.01
        max_iterations: int = 50

    def __init__(..., sdf_contact_config: SDFContactConfig | None = None): ...
```

- `"edge"` stays the default. It is today's path, bit-identical.
- This is the global switch: every non-hydroelastic SDF pair in the pipeline uses the chosen method.
- Per-shape selection is deliberately left out (YAGNI).
- `SolverPhoenX`'s auto-created pipeline (solver.py:804-851) gets the same config passthrough.
- Examples get a `--sdf-contact-method {edge,cached_sampling}` flag through a shared helper. Candidates: `newton/examples/phoenx/_rigid_solver.py`, or a small `_sdf_method.py` next to it.
- Naming needs a check against `CODING_GUIDELINES.rst` (no acronyms like "dcd" in the public API). Add a changelog fragment `+sdf-cached-sampling-contacts.added.md`.

## 4. Implementation layout (new files, minimal edits to existing ones)

```
newton/_src/geometry/sdf_cached_sampling.py      # NEW: all kernels and funcs below
    softmax_objective(...)             @wp.func   φA, φB, hw texture fetches, repulsion over ≤8 anchors
    nelder_mead_3d(...)                @wp.func   fixed-size simplex in registers, clamp to V
    contact_from_point(...)            @wp.func   Eq. 5 + normal coherence + Newton gap → (pA, pB, n, sep)
    draw_seed(...)                     @wp.func   rejection-sample mesh_edge_centers of A/B inside V (≤16 tries) → uniform V fallback
    cached_sampling_contacts_kernel    tile kernel, one pair per tile of 8 lanes
    sequential_reference_kernel        test-only faithful port of Manifold::update
newton/_src/geometry/sdf_pair_cache.py           # NEW: SDFPairCache (persistent, graph-capture safe)
```

**SDFPairCache** reuses ContactMatcher's pattern: sorted keys with binary search, no hashtable.
1. `key = (shape_a << 32) | shape_b` for each DCD pair; `wp.utils.radix_sort_pairs` sorts them (the pair count is device-side; launch with capacity).
2. Each pair binary-searches `prev_sorted_keys` to get `prev_slot` (or −1).
3. The narrow-phase kernel reads `prev_cache[prev_slot]` and writes `next_cache[sorted_index]`.
4. A copy kernel moves `next` → `prev` and `count`. This uses persistent arrays and no Python-side swaps, so it is graph-capture safe.
5. Per-pair record (SoA):
   - `pA_loc[MAX_CACHE]`, `pB_loc[MAX_CACHE]`, `normal_loc_A[MAX_CACHE]` (stored in A's frame), `slot_id[MAX_CACHE]`, `active_mask`, `count`;
   - optional `prev_g` for the replacement test.
6. Pairs that disappear (no longer overlapping in broad phase) drop out naturally.
7. `reset(world_mask)` hooks into `CollisionPipeline.reset_contact_matching`.

**Edits to existing files (small):**
- `narrow_phase.py`:
  - one `wp.static(method)` branch at the `:701-718` routing, plus a new pair buffer;
  - launch the new kernel next to the mesh-mesh launch (`:3391-3486`); the edge kernel still runs for leftover BVH/heightfield pairs.
- `collide.py`: config plumbing; size the pair cache from `max_mesh_mesh_pairs`; a `rigid_contact_max` estimate of `pairs × (n_cache + n_rand)`.
- `contact_data.py` / `write_contact`: no change. Set `center = ½(pA + pB)`, `distance = signed separation`, and `sort_sub_key = slot_id << 2`.

### Kernel shape (performance design)
- One warp holds 4 pairs × 8 lanes.
- Each lane keeps its 4-vertex simplex and 4 values in registers (about 20 floats).
- Each objective evaluation costs 2 hw texture fetches (one per SDF), plus coordinate transforms precomputed per pair into `X_A_inv`, `X_B_inv` and scales, which are shared in tile memory.
- Expected work per pair (from paper Tables 3–4): about 400–2500 objective evaluations per scene step. Compare that with today's Brent over all culled edges in both directions plus the global hashtable reducer and export.
- Expected win: the reducer and export kernels disappear entirely for these pairs. Fewer contacts also means less ingest and coloring work in PhoenX.
- Divergence: lanes finish NM at different iterations. Mitigations:
  - cap iterations;
  - cached lanes converge in a few iterations, which is the common case at rest;
  - if a profile shows a long tail, drop to `max_iterations = 30` for cached lanes.
- Bench rules apply: nsys only, CUDA graph capture, solo GPU.

## 5. Phases and gates

**Phase 0: validate assumptions (½ day)**
- [ ] Coarse texture holds true interior distances (the build queries the source at subgrid corners), not narrow-band-clamped values.
  - Check the bunny SDF and the nut/bolt SDF (`narrow_band_range = ±1 mm`).
  - If interior values are clamped, the deep-overlap landscape goes flat. Note that the edge method has the same depth limit.
- [ ] Hardware-filtered sampling (8-bit weights) is accurate enough for NM with τ ≈ voxel. Otherwise use software `texture_sample_sdf` inside the loop and measure the cost.
- [ ] `mesh_edge_centers` density per shape (bunny, gears, bike parts, nut). Too few seeds means a uniform fallback; too many means rejection-sampling inefficiency. A Morton-sorted per-shape seed grid is the fallback plan.

**Phase 1: correctness MVP (texture SDF ↔ texture SDF, mesh/convex/box-with-SDF)**
- [ ] `sdf_cached_sampling.py` with the sequential reference kernel first. It is the simplest faithful port and the oracle for later work.
- [ ] `SDFPairCache`, routing switch, config, and writer integration.
- [ ] Unit tests (CUDA only, graph-captured, under 1 min each, in `newton/tests/test_sdf_cached_sampling.py`):
  - two overlapping spheres (mesh SDF): x* sits at the lens centre, depth matches analytic within one voxel, normal is along the centre line;
  - box on box (mesh cubes with SDF) resting for 200 frames: ≥3 non-collinear active contacts, contact midpoints stationary under 0.1 mm per frame, match rate 100% after warm-up;
  - cache survives pair reordering (shuffle shape order between frames);
  - determinism: two runs are bitwise identical;
  - hydroelastic pair plus `method="cached_sampling"`: hydro contacts are bitwise identical to `method="edge"`;
  - routing: BVH-only mesh pair still takes the edge path;
  - `method="edge"` leaves results bit-identical to main (regression guard);
  - each test must be shown to fail without its fix, per AGENTS.md.

**Phase 2: GPU tile kernel + PhoenX A/B**
- [ ] Lane-per-candidate tile kernel. Parity vs the sequential reference on test scenes; statistical, not bitwise, because repulsion is frozen.
- [ ] `--sdf-contact-method` flag in PhoenX examples; `pipeline_factory` switch in `tune_phoenx_scenes.py`.
- [ ] Comparison harness: `local_studies/sdf_dcd/compare.py`, built on `tune_phoenx.run_trial` and `_contact_separation`.
  - Scenes: rabbit pile, nut/bolt (a 4×4 grid is enough), analog clock, tourbillon, bike transmission, caterpillar.
  - New micro scenes from the paper:
    - **SDF box pyramid**: mesh cubes with SDFs, comparable to paper Fig. 5. This is the stacking gate.
    - **Gear stacking**: Fig. 5 right.
    - **Bunny/cow drop**: 100 randomised orientations, max penetration over time (Fig. 6).
    - **Pile at rest**: kinetic energy over time (Fig. 7).
  - Metrics:
    - **Performance:** narrow-phase kernel time (nsys), end-to-end FPS (graph-captured, solo GPU), contacts per pair, NM iterations and SDF queries per step (debug counters behind a `wp.static` flag).
    - **Stability/accuracy:** max and p95 penetration, rest-state KE and drift (position over 10 s at rest), warm-start match rate, joint error and drive speed tracking for clocks and bike (existing `extra_metrics`), plus contact count variance frame to frame.
  - Chaotic piles run several `--jitter-seed` repeats per method; the quality noise lesson from the ragdoll pile applies.

**Gate to continue**
- No stacking or penetration regression on the box pyramid, rabbit pile and clocks across ≥3 seeds.
- Bike and caterpillar keep their joints.
- A clear narrow-phase speedup on at least the mesh-heavy scenes.

If it fails, record why in this folder and stop. Likely failure modes:
- **nut/bolt helical threads:** many disjoint contact regions, too few points. Try `cache_size = 8–16`.
- **Gear teeth:** chatter.

**Phase 3: production polish (only if Phase 2 passes)**
- [ ] Analytic primitive vs SDF. The paper supports analytic SDFs. Mesh vs sphere/capsule/box currently goes through triangle GJK. Evaluate `sdf_box` / `sdf_capsule` etc. (`kernels.py`) inside the objective via `wp.static` dispatch on shape type, which avoids baking box textures.
- [ ] Eq. 7 curvature seeds for shapes without sharp edges (smooth organic meshes where `_build_collision_edges` keeps few edges):
  - Laplacian from the texture with δ = 1 voxel, not 0.001·L (the trilinear gradient is piecewise constant at sub-voxel steps);
  - Poisson thinning on host at build time;
  - stored as `shape_sdf_seed_range` plus a `sdf_seed_points` array;
  - cached with the SDF (disk cache key).
- [ ] Speculative contacts: grow V by `speculative_contact_gap_max`; emit when `sep < gap_sum + margins + speculative`; match `test_speculative_contacts.py` semantics.
- [ ] Heightfield vs SDF (`sample_sdf_heightfield` as φ_B) if needed.
- [ ] Docs (`docs/` collision page section, config reference), changelog, example README entry if a new example lands.
- [ ] `/code-review-newton` pass; `uvx pre-commit run -a`.

## 6. Risks / open questions
1. **Coverage on large flat contacts with only 4–5 points.** The paper shows boxes are fine (116 contacts for 29 boxes). PhoenX needs ≥3 non-collinear points per face, and the replacement threshold plus repulsion α must spread them to the corners. α is relative to L; we may need α tuned per pair size.
2. **Interaction with contact matching.**
   - Sticky mode overwrites matched anchors with previous geometry. That is fine, because cached points already advect.
   - New samples replacing cached ones change the slot → new sub-key → cold start for that contact. This should be rare thanks to the replacement threshold.
3. **Many-region contacts** (threads, chain-sprocket, gear meshes) are where the edge method's dense sampling is strongest. Pairs should probably report how many cached points were active, so a future adaptive `n_cache` (paper "future work") has a signal.
4. **Deep penetration beyond the narrow band** is limited by coarse grid accuracy. It is equally limited on the edge path, so it is not a regression.
5. **Determinism under graph capture** needs device-side frame counters, sorted keys and no host reads. All are covered by the design above.
