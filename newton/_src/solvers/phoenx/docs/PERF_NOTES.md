# PhoenX Performance Notes

Curated lessons-learned from past performance work on the PhoenX solver. Goal is to avoid re-trying ideas that have already been characterised, capture *why* a change won or lost, and surface knobs that scenes can opt into.

This is **not** a substitute for `git log` — it's a hand-maintained shortlist of the load-bearing decisions and the dead ends.

## Active wins

### Topology-selected packed reduced-contact gather (2026-07-03)
- NCU showed the gather at 67% memory SOL, 128 registers, 33% occupancy, and roughly 57% sector-utilization headroom. The old launch dedicated one 32-lane warp to each articulation even when only a few contact points were live.
- A specialized 8-lane mapping packs four independent articulations per warp. It preserves every contact, all 32 points per page, arbitrary column counts, and the existing multi-page graph loop. Contact math remains identical.
- The packed path is selected once at construction only when articulation_count is at least 8 times the device SM count. Smaller fleets keep the original kernel byte-for-byte; there is no user tuning knob.
- A final 300-replay 8192-world bracket improves 1.417M to 1.512M steps/s (+6.7%); an earlier cooler bracket measured 1.625M to 1.717M (+5.6%). Steady graph-leapfrog training improves 795.6k to 803.7-805.6k samples/s (+1.0-1.3%).
- The complete 40-test reduced CUDA-graph suite passes. The arbitrary-contact-count test explicitly forces the packed kernel through more than two 32-point pages, zero-contact replay, and contact restoration in the same captured graph. Anymal/H1/G1 512-world screens stay on the unchanged serial path.

### Reduced factor/contact multi-stream overlap (2026-07-03) - REJECTED
- Full begin_substep overlap is incorrect: ABA advance publishes link velocities read by contact warm-start tangent construction. A race-free prototype overlapped only kinematics/factorization with ingest and joined before advance.
- It captured and replayed correctly inside the leapfrog trainer, but lost twice: 778.7k vs 800.9k samples/s, then 774.7k vs 796.1k in reversed order (about -2.7%). Concurrent factorization contends with the already-overlapped learner more than it hides setup. The implementation was removed completely.

### Reduced contact-row builder: coalesced zero fill + recomputed path Jacobian (2026-07-02)
- Nsight Compute (privileged run, user-assisted) on the G1 row builder:
  Memory 66% vs Compute 25% SOL, **5.1/32 bytes utilized per global-load
  sector, 5.8/32 per store sector**, 61% of warp stalls on L1TEX
  scoreboard, occupancy register-limited to 62.5% theoretical (64 regs),
  4.36 waves. The kernel is a *strided-access* problem, not a latency wall:
  every `[packed_row, dof]` access puts adjacent row-threads 144 B apart.
- Retained (bit-identical, all 40 reduced tests pass): block-cooperative
  coalesced Jacobian zero fill (all 96 block threads stride the flat slab)
  and effective-mass accumulation recomputing `dot(joint_s, source_wrench)`
  on the source path instead of reloading the padded Jacobian row per DOF
  (off-path terms are exact zeros and are skipped). Builder median
  410.0 -> 399.3 us; physics-only profile runs 1.19 -> 1.22-1.24M env
  steps/s (profiler-run variance ~2%).
- **Not retained: always routing responses through joint_work + tile
  transpose.** Builder median dropped to 312.5 us (-24%) and physics-only
  improved, but steady leapfrog *training* regressed 2.7% (consistent,
  bracketed) - the extra 8192x6-tile transpose per substep contends with
  the overlapped learner stream. Lesson: **validate physics-only wins
  against `bench_g1_train --execution-mode graph_leapfrog`**.
- **Stream-balance measurement (nsys, steady leapfrog window):** GPU
  union-busy is 96%; the rollout stream (~0.45 s/iter: physics + env +
  policy forward) is the long pole over the update stream (~0.14 s/iter).
  Training IS rollout-bound - physics wins land on the critical path, but
  a few-percent physics gain is only ~1-2% end-to-end, inside the ~±1.5%
  bench noise. Judge small candidates by *rollout-stream busy time* from
  an nsys trace, not raw samples/s. Also: the row builder averages
  ~205 us in real training vs ~400 us in the standing-pose physics
  profile (fewer active contact rows mid-episode), so standing-pose
  kernel shares overstate the contact path roughly 2x; the training-trace
  physics ranking is advance 8.6% > factor 6.2% > row build 5.9% >
  contact solve 4.9% > publish 4.5% of total GPU.

### Tiered contact sort under graph capture (2026-07-02) - REVERTED, brittle
- Attempted: `wp.capture_if` chain in `ContactSorter._sort_and_permute`
  sorting the smallest quarter/half/full capacity tier holding the live
  count. Result was bit-identical (verified by state hashes) and cut the
  onesweep total 4.30 -> 3.64 ms per 60 substeps (-16%, only ~0.5% physics
  because the 64-bit key forces 8 radix passes regardless of size).
- **Reverted (commit 3f51b909): warp's CUB radix sort allocates temp
  storage per stream, and conditional graph bodies must not allocate.**
  Constructor pre-warm fixes the default stream, but the leapfrog trainer
  captures on other streams and dies with "Conditional body graph contains
  an unsupported operation (memory allocation)". Don't re-try until warp
  offers externally-provided temp storage for `radix_sort_pairs` (or
  pre-instantiated child graphs per tier are practical); the ~0.5% is not
  worth the stream-configuration fragility.
- Also judged not worth it: a pass-count cut via compact sort keys - it
  conflicts with the top-aligned `make_contact_sort_key` bit layout and
  the matcher's low-32 tiebreak (shared-code blast radius for ~1%).

### Block contact gather: skip builder-overwritten effective masses (2026-07-02)
- `reduced_contact_prepare` computed three per-direction inverse-mass
  traversals per contact, but for block-owned contacts the packed row
  builder derives exact effective masses from the generalized response
  rows and overwrites the `eff_*` slots before any solve reads them. A
  `compute_effective_mass` flag now skips that work on the gather path
  (fallback + deferred callers still compute it).
- Gather kernel median 110.8 -> 84.6 us (-24%) at 8192 G1 worlds;
  bit-identical state hashes; all 40 reduced tests pass.

### Second ncu digest: factor / gather / solve (2026-07-02)
- **factor**: 168 regs -> 25% theoretical occupancy (register-limited),
  memory 58% SOL, ~50% sector-utilization headroom on loads/stores.
- **gather** (`_gather_reduced_contact_blocks_kernel`): 128 regs, 33%
  occupancy, memory 67% SOL, **~57% estimated coalescing headroom** - the
  next builder-style strided-access target.
- **solve tile**: memory 57.5% SOL, ~28% coalescing estimate.
- Depth-synchronized kernels (advance/factor/publish/kinematics) each run
  the GPU well under 35% utilized while serialized in the captured graph;
  the structural lever is overlapping them with independent branches
  (multi-stream capture: factor/advance are independent of collision
  gather until the contact solve).

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

### Multi-world global color order
- Multi-world PGS keeps single-world iteration semantics: each solver iteration visits all colors once before any color repeats.
- There is no production knob for local multi-sweeps; they were faster, but changed the fixed point for coupled contact/joint scenes such as G1 standing contact-drive.

### Revolute-only kernel specialisation (single + multi world)
- When every joint is `JointMode.REVOLUTE` (or there are no joints at all), the iterate kernels skip the per-cid `read_int(_OFF_JOINT_MODE)` and the four-way `joint_mode` branch in `actuated_double_ball_socket_iterate{,_multi}`. They call `revolute_iterate{,_multi}` / `revolute_prepare_for_iteration` directly.
- Detection: `PhoenXWorld._use_revolute_specialization`, set by scanning the `joint_mode` array once during `initialize_actuated_double_ball_socket_joints`. Default `True` for `num_joints == 0`.
- Single-world kernels are factory-generated with a `wp.static` `revolute_only` axis (`_make_singleworld_persistent_kernel` / `_make_singleworld_fused_kernel`). Multi-world fast-tail kernels likewise (`_make_fast_tail_*`).
- Wins are modest standalone; main value is keeping the iterate kernel binary smaller for the common all-revolute case.

### Cable joint: combined pd_coefficients + Nyquist clamp + full log-map
- **Combined PD softness (Jitter2 / Box2D-v3 implicit-Euler).** Cable bend/twist rows route through ``pd_coefficients`` (same helper BEAM uses). The earlier branch used the spring/damping split (``pd_coefficients_split``) under the assumption that splitting would converge faster at high damping; in practice the split's relax-pass damping was an unsoftened ``lam = -damp_mass * Jv`` that drove ``Jv -> 0`` rather than to the implicit-Euler steady state ``Jv = J v_init / (1 + dt*c*M_inv)``, so multiple PGS applications overshot the analytical answer. Long cables with the split formulation also under-propagated through the chain because damping ran only in the relax pass (``velocity_iterations`` sweeps, default 1) instead of every iterate sweep (``solver_iterations``, default 4-8). Combined ``pd_coefficients`` has positive ``gamma`` softness so PGS converges to the implicit-Euler answer in ~2 iterations and ``gamma * acc`` brakes against overshoot; damping naturally lives in the iterate loop and gets the full ``solver_iterations`` propagation budget. Cable's high-damping settle test now lands at ~14% residual (analytical ``exp(-2) = 13.5%``) -- the split's "10%" was overdamping past the analytical answer.
- **Nyquist stiffness clamp** in ``pd_coefficients``. Caps ``k`` at ``1 / (M_inv * dt^2)`` so user-supplied gains beyond the substep's resolution degrade gracefully instead of producing impulse spikes.
- **Full quaternion log-map** in ``_cable_prepare_at`` and the cable branch of ``actuated_double_ball_socket_world_error_at`` (replaces the small-angle ``kappa = 2 * q.xyz`` which underestimates by ~6% at 80 deg).
- **Cross-substep warm-start cleared** for cable bend/twist. The previous fold applied the prior substep's accumulated impulses along the *new* substep's basis directions, which is wrong whenever the parent body rotates between substeps. Within-substep warm-start (acc accumulation across the iterate sweeps) is unchanged.

### ``velocity_iterations`` may be 0
- Pre-combined-formulation, the cable damping split made the relax pass load-bearing and ``velocity_iterations >= 1`` was enforced. With the combined ``pd_coefficients`` formulation damping moves back into the iterate loop, so the validator now allows ``>= 0``. ``TestVelocityIterationsValidator`` exercises the new minimum (a single box settles correctly with the relax pass disabled).

### Per-substep `inverse_inertia_world` refresh
- After `_integrate_positions` rotates each dynamic body, `bodies.inverse_inertia_world` was stale for the next substep's solve. For anisotropic links this biased the angular impulse direction over the substep loop.
- Fix: `_phoenx_refresh_world_inertia_kernel` runs after every substep. Real correctness gain on G1/H1 hold-pose parity tests; modest cost.

### Adaptive threads-per-world (multi-world fast-tail)
- The fast-tail launch grid is fixed at `num_worlds * _STRAGGLER_BLOCK_DIM` (= warp), but the active lane count per world (`tpw`) is picked per step from the colour histogram. `_pick_threads_per_world_kernel` reads `_world_num_colors` and `_world_csr_offsets` and writes `_tpw_choice[0]`.
- Host-side `threads_per_world="auto"` pins obvious graph-capture-stable topologies before capture. On RTX PRO 6000, 4096-world H1-like fleets (`~23` joint rows/world, `~320` contact capacity/world) are faster at `tpw=8`, while 2048-world H1 and 4096-world G1/DR-Legs remain faster at `tpw=16`; `_choose_initial_threads_per_world` encodes that split.
- Pinned to 32 below `8 * sm_count` worlds (picker overhead would never pay off). Captured-graph safe (no host sync).
- User opt-out: `threads_per_world={8,16,32}` in the solver constructor.
- Greedy coloring always builds per-family offsets. Large rigid mixed fleets
  (512+ worlds with joints and contacts) consume those joint/contact subranges
  directly in fast-tail kernels, avoiding a per-cid joint/contact branch in the
  hottest solve loops. Narrow G1/H1/DR-Legs scheduling sweeps on RTX PRO 6000
  showed 5-18% lower frame time for the affected fast-tail path.


### Block-world scheduler for dense mixed robot fleets
- 2026-06-23: Anymal PPO profiling showed multi-world fast-tail
  `prepare_plus_iterate` dominated kernel time. A scheduler sweep on the real
  Anymal RL environment, plus H1/G1/DR-Legs benchmark scenes, found that one
  32-thread CUDA block per world preserves the same colouring and PGS row work
  while improving lane utilisation on short mixed joint/contact colour loops.
- Stable auto-vs-`block_world_32` graph replay sweep (`substeps=4`,
  `solver_iterations=8`, `prepare_refresh_stride=auto`, 32 captured frames,
  3 trials, RTX PRO 6000): H1 512/1024 improved `1.20x`/`1.25x`, G1
  `1.11x`/`1.12x`, DR-Legs `1.12x`/`1.19x`, Anymal `1.13x`/`1.12x`.
- Short Anymal PPO nsys before/after: prepare+iterate changed from
  `_make_fast_tail_prepare_plus_iterate` at `733.4 ms` total / `286.5 us`
  average to `_make_block_world_prepare_plus_iterate` at `376.0 ms` total /
  `146.9 us` average over the same 2,560 launches. Total CUDA kernel time in
  that short profile dropped roughly `29%` (`~1.29 s` -> `~0.91 s`).
- Production `multi_world_scheduler="auto"` selects `block_world_32` for
  supported mixed fleets with many worlds. The selector was originally gated on
  *substantial contact capacity* (64-512 contacts/world), which wrongly excluded
  **joint-heavy contact-light fleets** like dr_legs (38 joints + 30
  contacts/world) -- they stayed on fast-tail and lost ~18%.
- 2026-06-25: generalised the selector (`_choose_multi_world_scheduler`) -- once
  a fleet is large and non-sparse, *any* joint-bearing topology picks
  `block_world_32` regardless of contact density. Diagnosis via ncu
  (`analysis_tools/ncu_profile_iterate.sh`): dr_legs@4096 fast-tail was
  latency-bound, only ~12/32 active threads/warp at 25% occupancy -- one CTA
  per world restores lane utilisation. **dr_legs@4096: +17.9% env_fps**
  (2.18M -> 2.57M, drift-controlled interleave). h1/g1/Anymal already met the
  old contact-dense gate (320-510 contacts/world) so their selection and
  performance are unchanged; contact-only (big_box/tower) and sparse fleets
  keep their existing choices. Guarded by `test_multi_world_scheduler_helper`.

### Inertia + force-clear fusion
- Damping + rotated-inertia refresh + force/torque zeroing were three back-to-back per-body kernels with the same dim/gate. Fused into `_phoenx_update_inertia_and_clear_forces_kernel`. Saves ~3 launches per step.

### Packed symmetric `inverse_inertia_world` (vec6)
- 2026-06-25: `inverse_inertia_world` (= R·I⁻¹·Rᵀ) is always symmetric, so the full `mat33f` carries 3 redundant entries. Stored as `inertia_sym6` (6 floats, `body.py`) and reconstructed via `mat33_from_sym6` at read sites / packed via `sym6_from_mat33` at writes; host I/O uses `inertia_sym6_{pack,unpack}_np`. It is the **largest single body field** (36 B → 24 B).
- **Why it's a broad win:** the multi-world fast-tail iterate/prepare is memory-bound and loads inertia **per-cid** (two bodies per joint/contact row ≈ the bulk of per-row traffic), so the 33% field shrink cuts real bandwidth. Single-world (kapla) loads inertia once per *contact column* (amortised over the column's contacts) so it sees no change.
- **Measured (RTX PRO 6000, env_fps median of 3, ~1.5% within-session variance):** dr_legs @4096 **1.83M → 2.23M (+22%)**; h1_flat @4096 **2.25M → 2.55M (+13%)**; kapla single-world **66.83 → 66.84 fps (neutral)**. 12 PhoenX regression tests pass.
- Reconstruction is a few register moves (cheap on a bandwidth-bound kernel). Off-diagonals come from the upper triangle, which also drops the FP asymmetry a general `R*I*Rᵀ` store could carry. `inverse_inertia` (body frame) stays `mat33f` — it's only touched in the once-per-substep refresh, not the per-cid hot path.
- **Likely applies to other narrow per-body fields too** if a future scene shows them dominating per-cid traffic; inertia was the obvious first target as the widest field.

### Contact prepare: defer tangent effective masses past the sticky-break
- 2026-06-25: `_make_contact_prepare_for_iteration_at` (rigid path) computed `eff_n`/`eff_t1`/`eff_t2` up front, but the sticky-friction-break block then re-projects fresh anchors and recomputes all three — so the tangent masses were computed twice whenever a contact's anchor broke. Only `eff_n` is consumed before that block (the Baumgarte `load_boost`/bias). Now only `eff_n` is computed up front; `eff_t1`/`eff_t2` are computed once after the anchor decision, from the final `r1`/`r2`.
- **Bit-identical** (no-break: same inputs; break: same fresh-anchor inputs as the old recompute). Removes one `effective_mass_scalar` pair per broken contact.
- Kapla steady-state prepare kernel: **252.6 → 248.5 ms / 16.70 → 16.44 us (-1.6%)** (drift-robust nsys, 100 frames). Small at steady state (few breaks) but free, and helps the transient settle / make-and-break phase of every rigid contact scene (single + multi world share this code).

### Unified local-block pipeline prototype
- `benchmarks/experimental/bench_unified_block_pipeline.py` extracts real PhoenX coloured graphs and maps rigid contacts / ADBS joint modes into a shared local-block operation set: contact3, point3, angular3, tangent4, scalar-linear, scalar-angular. It compares compact typed math (`split`), shape-grouped compact math, fully uniform 4-row sidecar descriptors, and a graph-capture-safe hybrid dispatcher.
- Real 2048-world RL scenes, 20 substeps, `prepare_refresh_stride=auto`, `tpw=16`: hybrid is consistently best on the local-block proxy (`h1`: 0.0616 ms vs split 0.0738, +19.8%; `g1`: 0.1015 vs 0.1413, +39.2%; `dr_legs`: 0.1005 vs 0.1121, +11.6%).
- Real 4096-world guardrails split by production lane count: H1 at `tpw=8` stays on compact split policy inside the hybrid dispatcher and wins over split (+18.6% in the proxy); G1/DR-Legs at `tpw=16` prefer shape-grouped compact math (+8-10%).
- Conclusion: do not promote a full sidecar4 path as-is; it loses on all measured robot schedules. The promising production direction is a unified dispatcher / scheduling shape with per-colour policy (`split`, `grouped`, or sidecar descriptor) selected from topology or a setup-time tournament.

### Soft-tet contact interaction-element: drop 4th tet vertex from coloring adjacency
- 2026-05-12: ``_constraints_to_elements_kernel`` (``solver_phoenx_kernels.py``) emits only the first 2 of the 3 ``side*_nodes_extra`` particles for ``SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON`` sides — i.e. 3 nodes (``b1 + e0a + e0b``) per soft-tet contact side instead of 4. The 4th tet vertex is opposite the contact face, so its barycentric weight is zero on a true face contact (and small on edge/vertex contacts); excluding it from the coloring adjacency lets the greedy MIS commit contacts on the same tet (sharing only the dropped apex) into the same colour.
- **Single-world soft_body_drop steady contact: −19.5% total GPU/frame** (12.98 → 10.45 ms, median over 3 nsys runs; per-run spread 0.2-0.5% baseline, 1-4% optE — clear signal). Per-launch breakdown: greedy coloring −12.7%, fused PGS iterate −22.2%, mass-splitting broadcast −16.1%, persistent kernel unchanged.
- Why the fused-iterate per-launch dropped 22%: fewer (body, partition_key) pairs in the mass-splitting interaction graph means smaller per-node ``section_end`` spans, so ``get_state_index``'s linear/binary search shrinks across every position read and write inside the iterate. The coloring win comes from a sparser adjacency graph (fewer neighbours to walk per element).
- **Iterate is unchanged.** Contact impulses are still applied to all 4 tet vertices with full barycentric weights — only the coloring's adjacency walk and the mass-splitting interaction-graph emit_pair skip the 4th vertex. Concurrent writes to the dropped vertex from contacts in the same colour fall through to direct particle storage (write-race) — the test suite (20 phoenx tests, including ``test_soft_body_mass_splitting_determinism`` and ``test_cloth_mass_splitting_determinism``) confirms the practical impact is negligible: for face contacts the racing impulse is zero (bary_d = 0) and for edge/vertex contacts it's small.
- **The 4th vertex is dropped UNCONDITIONALLY**, not picked by smallest-weight. Picking the smallest weight per cid is doable but invasive (needs per-contact bary lookup at ingest); the unconditional drop already lands the headline win.
- **Pair with Opt-F (next entry) for MAX_BODIES = 6**: post-Opt-E the densest constraint type uses 6 slots (tet-tet contact 3+3), so MAX_BODIES can shrink 8 → 6.

### Tighten MAX_BODIES 8 → 6 (follow-up to soft-tet 4th-vertex drop)
- 2026-05-12: after the soft-tet adjacency drop above, the largest interaction-element occupancy is 6 (soft-tet-vs-soft-tet contact 3+3; cloth-cloth 3+3 was already 6). Shrunk ``MAX_BODIES`` from 8 to 6, ``vec8i`` → ``vec6i`` in ``ElementInteractionData``, dropped the trailing ``s6, s7`` slots from the contact-ingest compact loop, and trimmed ``element_interaction_data_make`` to 6 args.
- **Perf: neutral on soft_body_drop** (+1.43% vs Opt-E alone, within the 4-5% per-run noise). The coloring kernel's adjacency walk already early-exits at the first -1, so the loop-bound shrink doesn't translate to runtime savings on this scene's typical element densities.
- **Memory: real 25% saving on ``copy_state`` capacity** (sized as ``constraint_capacity * MAX_BODIES``). On contact-heavy scenes (kapla, large cloth) this is the difference between fitting comfortably and pressuring GPU memory.
- Kept for code hygiene: the constant now matches actual usage, the compact loop is shorter, and the ``vec6i`` is one cache-line word smaller. Tests updated: ``test_graph_coloring{,_overflow}`` had hardcoded ``itemsize=32`` (the old ``vec8i`` byte width); now derived from ``MAX_BODIES``.

### Re-widen MAX_BODIES 6 → 8
- 2026-05-21: ``ElementInteractionData`` re-widened to ``vec8i`` to keep headroom for wider constraints. Existing kernels are unaffected at runtime — the adjacency walk still early-exits on the first -1, so the trailing -1 slots cost zero. The five ``element_interaction_data_make`` callsites in ``_constraints_to_elements_kernel`` were padded with two extra ``-1`` args.
- **Memory: ``copy_state`` per-row footprint goes back to 8 ints** (undoes the 25% saving from the 2026-05-12 shrink).

### Warm-start coloring cache stir (drift fix on tall stacks)
- 2026-05-19: graph-coloring warm-start was reusing the same per-(body-pair) colour across frames, which made the PGS solve converge to a biased fixed point under the locked coloring. On the Kapla tower (10620 bricks, ~67k contact columns, single-world) the bias compounded over a few hundred frames into a full tower collapse: max brick drift 3.33 m at 1000 frames, mean 0.09 m.
- Cold-start coloring (fixed seed + per-step ``contact_count``-shifted priorities) varies the colouring slightly each frame because the high bits of the JP priority shift as bodies settle. The variation averages the bias out (max drift 0.15 m, same scene) at a ~10 % step-time cost from the extra MIS work.
- Fix: two complementary mechanisms, both capture-safe and graph-replay safe:
  - ``warm_start_invalidate_period=N`` (default 4): every Nth ``build_csr`` zeroes ``cache_num_entries`` so the seed kernel finds an empty cache and greedy MIS rebuilds from scratch.
  - ``warm_start_rotate_skip_color=True`` (default True): each step picks one cached colour round-robin (via a step counter) and the seed kernel skips entries with that colour; MIS re-derives ~1/num_colors of the assignments per step.
- Combined cost (Kapla): ~3 % step time. Combined drift (Kapla 1000 frames): max 0.175 m vs 3.33 m unmitigated. Tested in the ``example_kapla_tower2`` drift probe.
- What doesn't work:
  - **Symmetric Gauss-Seidel** (alternate forward/reverse colour sweep). Marginal -- only edge colours swap position; middle colours stay put. Drift 3.33 -> 3.00 m.
  - **Cyclic shift of colour sweep order**. Made things worse (4.18 m). PGS doesn't converge when iteration order changes each round.
  - **Reducing ``MAX_GREEDY_OUTER_ITERS`` globally**. K=7 enough for Kapla cold-start but cloth needs more. Not safe as a default.
  - **Per-thread ``num_remaining`` early-exit in the MIS kernel**. ~20 % slowdown -- 38k threads broadcast-reading the same atomic-hot cache line invalidates ``atomic_sub`` writes elsewhere.

### ``wp.capture_while`` on the greedy MIS outer loop (default 2026-05-19)
- The legacy fixed-loop did 16 outer × 8 inner = 128 MIS launches per build, relying on per-thread ``color_tags[tid] != 0`` to make post-convergence iters cheap no-ops. The post-convergence no-ops still cost ~1.67 us each (driver / kernel-launch overhead), totalling ~213 us/frame.
- Switching to ``wp.capture_while(num_remaining, body)`` exits as soon as ``num_remaining`` hits 0:
  - **Warm-start fast path: 12 800 -> 880 partitioner launches per 100 frames** (18x fewer); coloring kernel time 0.21 ms -> 0.04 ms/frame.
  - Cold-start: 128 -> ~80 launches (~36 % reduction); time savings smaller (~76 us/frame) because the dropped launches were no-ops to begin with.
- The capture_while watcher adds ~210 ``set_conditional_if_handle_kernel`` launches per 100 frames -- noise-level.
- Enabled by default via ``PhoenXWorld(capture_while_greedy_coloring=True)``. The legacy fixed-loop path is preserved on the flag.

### Speculative coloring (Çatalyürek-style, opt-in)
- Implemented at ``speculative_pick_kernel`` / ``speculative_validate_kernel`` / ``speculative_commit_kernel`` (``graph_coloring_common.py``); deterministic via the same fixed priority permutation as JP-MIS.
- 3-phase per round: pick smallest free colour, validate vs uncoloured-neighbour-with-higher-priority-same-tentative, commit. Race-free because the commit lives in a separate launch so phase 2 has a stable ``color_tags`` snapshot.
- **Halves the round count** vs JP-MIS on dense graphs because MIS only commits "local maxima" while speculative commits at multiple colours per round. Kapla: ~32 rounds (96 launches) vs ~80 (80 launches).
- **Wall-clock comparable, not faster.** Per-round work is roughly 3x JP-MIS (3 kernels with neighbour scans on both ``color_tags`` and ``tentative_color``), so the fewer-rounds win cancels out. Cold-start Kapla 100-frame nsys: 1.37 ms speculative vs 1.65 ms MIS+capture_while -- 17 % faster on raw kernel time but within noise on end-to-end FPS.
- Default OFF. Useful as a building block for: dense graphs that exceed ``MAX_GREEDY_OUTER_ITERS`` on MIS, or future tuning (shared-mem ``tentative_color`` caching, warp-level forbidden-mask reductions) that closes the per-round cost gap.

## Designed, not yet implemented

### Multi-stream capture overlap for the reduced pipeline (design, 2026-07-03)
- **Motivation (counter-backed):** advance/factor/publish/kinematics are
  depth-synchronized tree kernels that each leave the GPU well under 35%
  utilized while serialized in the captured graph (~30% of physics time).
  The collision/ingest chain is memory-heavy and independent of them until
  the contact row build.
- **Verified structure** (`solver_phoenx.py step()` + `solver.py`): per
  solver.step the sequence is import -> ingest_and_warmstart ->
  build_schedule -> coloring -> [substep loop: begin_substep (kinematics +
  factor + advance) -> integrate forces -> contact gather/rows/solve ->
  integrate positions -> relax -> publish]. For the G1 recipe substeps=1,
  so `begin_substep` (~410 us: kinematics 54 + factor 213 + advance 141)
  can fork against ingest+schedule+coloring (~200 us) right after import,
  with a join before the first contact gather (gather reads
  `reduced.body_q_com` from kinematics; row build reads `joint_d_inv`/
  `joint_u`/`joint_s` from factor).
- **Sketch:** persistent side stream + wp.Event fork/join inside step();
  skip `begin_substep` for substep 0 in the loop. Works eager and under
  capture (events create parallel graph branches); leapfrog already proves
  multi-stream capture in this codebase. Same kernels, disjoint arrays,
  fixed intra-stream order => bit-identical.
- **Open hazards to audit before coding:**
  1. `_advance_reduced_articulations_warp_kernel` writes `public_body_qd`
     and body velocities - confirm ingest/warm-start/coloring never read
     `bodies.velocity` (classic warm-start and contact prepare do, but they
     run after the join in-substep; check `_ingest_and_warmstart_contacts`
     specifically).
  2. `_kinematic_prepare_step` writes kinematic movers' velocities and
     currently runs between ingest and the substep loop - it must stay
     ordered against BOTH streams or be proven disjoint.
  3. Gate the fork on: reduced articulation present, CUDA, no picking, no
     sleeping, `reuse_partition=False`, and substep 0 only.
  4. Multi-substep recipes (substeps>1) only overlap the first substep
     unless collide/ingest also move inside the loop - fine for the
     3x(1-substep) G1 recipe, neutral elsewhere.
- Expected win if overlap is clean: O(100-150 us) of the ~810 us G1
  solver.step hidden, i.e. up to ~10-15% physics; discount for SM
  contention. Validate with bit-exact state hashes + the leapfrog trainer
  (per the tiered-sort lesson, always test the non-default-stream path).

## Experimental-only code

### Warp-local no-coloring PGS scheduler
- `benchmarks/experimental/bench_lock_scheduled_pgs.py --lock-mode warp` tests
  one warp per world. Each lane proposes an unfinished row, the warp accepts a
  body-disjoint micro-wave using shuffle/ballot intrinsics and a 32-bit local
  body mask, then solves accepted rows immediately. There are no global body
  locks and no graph coloring.
- Synthetic scalar PGS results on RTX PRO 6000 Blackwell, four iterations,
  graph-captured timing:
  - `2048` worlds, `32` bodies/world, `64` rows/world, mixed graph:
    colored `0.335 ms`, warp-local `0.145 ms` (`2.3x`).
  - `4096` worlds, `30` bodies/world, `43` rows/world, G1-sized mixed graph:
    colored `0.268 ms`, warp-local `0.141 ms` (`1.9x`).
- Global runtime-claiming variants are still bad: queued global locks were
  about `7.8 ms` on the same 2048-world mixed graph. If a no-coloring solver is
  pursued, keep it warp-local or block-local.
- Follow-up actual-solve benchmark (`bench_color_grid_actual_solve.py`) did
  **not** transfer the synthetic win to the real current G1/H1/DR-Legs kernels.
  G1 4096 solve-only: colored fast-tail `0.764 ms`, basic warp-local
  `0.950 ms` (`0.80x`). A refill/tile-stack variant that keeps lanes scanning
  for a compatible row was worse: `2.392 ms` (`0.35x` vs a `0.824 ms` baseline
  in that run). H1 512 was `0.44x`; DR-Legs 512 was `0.83x`.
- Caveats: the synthetic benchmark is a scalar row solve and changes GS row
  order. The actual-solve prototype keeps a 64-body fast-path mask with a clean
  refusal for wider worlds; production would need a wider/chunked representation
  plus convergence tests. Do not promote the no-color warp-local scheduler on
  current evidence.

### Actual-solve color-grid scheduler prototypes
- `benchmarks/experimental/bench_color_grid_actual_solve.py` keeps the flat/global-colour, block-per-world, grouped-subfamily, autotune/adaptive, and software-barrier mega-kernel scheduler prototypes in one place. They are useful research scaffolding, not production evidence.
- Current state: some prototype timing paths have been observed to hang (`flat_grouped` and `block_world` on small H1 smoke runs), while the production `world._solve_main` path with the same scene setup completes. The benchmark now defaults to `--mode baseline`, which only times the production solve path; all prototype modes require `--allow-unsafe-prototypes`.
- Use the production benchmark suite and the unified-block proxy benchmark for decisions that affect defaults. Re-enable actual-solve prototype modes only for isolated scheduler debugging with short timeouts.

## Tried and reverted

### prepare_refresh_stride=3 for the G1 recipe (2026-07-02) - REJECTED, physics-changing
- +3.1% physics-only throughput (1.226M vs 1.189M env steps/s). Full
  train-to-gate at stride 3 *passes its own internal gate* (battery_perf
  0.920, zero falls at 125.8M samples) - but that gate inherits the
  stride-3 physics. **The same checkpoint collapses under true stride-1
  physics: 17k falls / battery_perf 0.073 at BOTH gate seeds.** The policy
  learned to exploit stride-3 contact staleness (worst in the final phase,
  where stride 3 > substeps 2 leaves anchors stale across policy steps).
  This is the canonical "faster unsuccessful training" trap and empirically
  vindicates the auto heuristic choosing stride 1 below 8 substeps.
- **Methodology lesson:** `bench_g1_train_to_gate` evaluates under the same
  physics flags it trains with; always re-gate candidates with the
  standalone `gate-g1-ppo` CLI (recipe-default physics) before believing a
  pass. The bench-vs-CLI discrepancy is a built-in physics-overfit detector.

### Advance/publish tile-width sweep at 8192 articulations (2026-07-02)
- The reduced-pipeline ncu report shows the ABA advance at 0.34 waves/SM
  (512 blocks) with an "81% speedup" small-grid recommendation. Widening
  `advance_tile_width` does NOT deliver it: tile 8/16/32 give advance
  141/134/154 us and end-to-end physics 1.23/1.19/1.16M env steps/s -
  **tile 8 stays best**. The tree's per-depth joint width (~2-6 for G1)
  bounds true parallelism; wider tiles only add idle lanes and lose the
  warp-shared joint-data locality. Same conclusion as the factor sub-warp
  tiling attempt. Don't revisit tile widths; the remaining headroom for the
  depth-synchronized kernels is overlapping them with independent graph
  branches (multi-stream capture), not launch-shape tuning.

### Reduced G1 contact-row builder micro-optimizations (2026-07-02)
- Fresh physics-only nsys (8192 worlds, 3 substeps, 2 iters, 1 relax, stride 1):
  `_build_packed_generalized_contact_rows_kernel` is the #1 reduced-pipeline
  kernel at **18.7% / ~410 us per launch**; contact pipeline (build + gather +
  tile solve + finalize + transpose) totals ~38%. Three hypotheses falsified:
  - **Scalar 1-DOF fast path** (skip the unrolled 6x6 `joint_d_inv` loops for
    revolute joints, bit-identical): **neutral** (408 vs 410 us median). The
    kernel is not ALU/instruction-bound.
  - **Per-thread ancestor-delta stack** (replace the `aba_body_response`
    global scratch round trip with a depth-indexed local stack, bit-identical;
    ~280 MB/launch of scratch traffic removed): **-25% regression** (512 us).
    The global round trip was already L2-absorbed (per-articulation working
    set ~69 KB, re-read right after write); the 384 B/thread local stack frame
    only added local-memory traffic. `ptxas` shows both variants at 64
    registers, zero spills - occupancy is not register-limited either.
  - **block_dim sweep** (32/48/64/96/192) on the builder: flat within +-4%.
- Row-count scaling is strongly sub-linear (12-row unit-wrench basis build:
  349 us vs 462 us for 24 rows in the isolated bench), so the cost is
  dominated by per-thread serial full-tree traversal latency, not row count.
- **Unit-wrench basis compression re-measured at the current 36-wide packed
  layout** (`bench_g1_response_basis_aba`, width fixed from the stale 48):
  compressed prepare+solve 504 us vs direct 566 us = only **1.12x**, with
  FP-tolerance (not bit-exact) impulses. The old "~19%" claim predates the
  four-DOF-aligned width and the scratch-clear/transpose-skip wins. Not worth
  productionizing at 1.12x given the body-diversity dispatch + gate
  revalidation it needs.
- **Factor kernel sub-warp tiling also regressed** (same session):
  `_factor_reduced_warp_kernel` (11.1%, 168 registers, ~12 warps/SM occupancy)
  was given the advance kernel's `tile_width=8` articulation packing
  (4 G1 trees per warp, 4x lane utilization, single wave instead of 3.6):
  **+9% slower** (232 vs 213 us median). Extra concurrency does not help these
  depth-synchronized reduced kernels either - consistent with a dependent
  memory-latency wall across the whole reduced pipeline, not an
  occupancy/utilization shortfall.
- **Next diagnostic requires privileged counters** (`ERR_NVGPUCTRPERM` blocks
  ncu for non-admin): `benchmarks/profile_g1_contact_rows_ncu.sh` captures a
  full-set report of one builder launch. The stall/occupancy breakdown decides
  between a depth-parallel cooperative builder vs accepting the latency wall.

### Raising block_world occupancy by cutting registers (latency-bound iterate)
- 2026-06-25: ncu on `block_world` `prepare_plus_iterate` (dr_legs@4096) = latency-bound, **occupancy 25% limited by registers** (binding cap: 12 blocks/SM vs 24 from SM/barriers), 38% Long-Scoreboard stalls, ~7/32 active threads/warp. So the lever *looked* like "more resident warps to hide load latency".
- **`__launch_bounds__` register cap (`launch_bounds=(block_dim, minBlocks)`): −35%.** Forcing 16 or 24 blocks/SM made the compiler spill the kernel's ~170 live registers to local memory; the spill traffic/latency dwarfed the occupancy gain (both 16 and 24 targets regressed identically → a spill cliff, not a tuning curve). The kernel's registers are load-bearing.
- **Organic register reduction (skew→cross): neutral.** The revolute iterate materialised four `skew(r)` mat33 (held ~36 register-words across the sweep loop); replaced every `skew(r)@v`→`cross(r,v)` / `transpose(skew(r))@v`→`cross(v,r)` in `_revolute_iterate_at_multi` (bit-equivalent, 11 multi_world tests pass). dr_legs@4096 unchanged (2.69M vs 2.68M, identical gpu mem) — the compiler already lowers skew efficiently and/or ~24 of ~170 registers doesn't cross an occupancy block-limit threshold. Reverted.
- **Conclusion:** after the block_world scheduler win (+18%, the achievable optimization), the iterate is at a register/latency wall. Incremental math-sharing register cuts don't move occupancy; a register cap spills. The only structural register cut left is **un-fusing prepare from iterate**, but the fusion exists precisely to avoid the extra derived-data memory round-trip — adding it back on a *latency*-bound kernel is likely a wash or worse. Don't chase this without a fundamentally different solver formulation (e.g. coloring that fills warps better than ~7/32, which is the real remaining inefficiency).

### Further inertia compression: vec6 → uvec4 (21-bit packed)
- 2026-06-25: after the lossless vec6 win, tried compressing `inverse_inertia_world` further to a `uvec4` (16 B) — the C# `DataCompression.CompressMat3Sym` scheme: each of the 6 symmetric entries → 21-bit float (radix flip + keep sign+8-bit-exp+top-12-mantissa; preserves full exponent range, unlike fp16, which matters for dr_legs' ~6 g bodies whose *inverse* inertia is huge), six packed 3-into-2-uint32.
- Validated: device decode bit-exact vs a numpy reference, roundtrip rel err max 0.024% / mean 0.009% across 10⁻⁷..10⁷; 12 regression tests pass.
- **Perf: neutral-to-slightly-negative.** Back-to-back vs vec6 (median of 3, 4096 worlds): dr_legs 2.231M → 2.227M (neutral), h1_flat 2.567M → 2.532M (−1.4%). Reverted.
- **Why:** vec6 already cut inertia 36→24 B, after which inertia is no longer the dominant per-cid load, so shaving another 8 B buys almost no bandwidth — while the bit-twiddle decode (6× float-flip + unpack vs vec6's free float rearrange) adds ALU. Net wash, plus 0.024% precision lost for nothing. **General lesson: compression pays only on the *largest uncompressed* per-cid field and only while the decode stays cheap; once the big field is lossless-packed, further lossy packing of it (or of smaller fields) tends to net neutral.** Likely the same verdict for octahedral-encoding the per-contact normals/tangents (12 B → 8 B, lossy + decode normalize) and for quaternion compression — measure before implementing; expect marginal.

### Contact-major (AoS) `cc.derived` layout
- 2026-06-25: `cc.derived` is SoA `(CC_DERIVED_DWORDS, n)` with k inner ("for coalesced loads"). Probing the kapla colour CSR showed the 8 colored partitions (56% of contacts) have **scattered cids** within a colour (median |Δcid|=11, only ~15% within 4), so adjacent warp lanes touch non-adjacent contacts and those per-contact derived reads do *not* coalesce across the warp — each thread pays ~1 cache line per field (~15 lines/contact). The 32k overflow colour (45%) has sequential cids and *does* coalesce. Hypothesis: a contact-major `(n, CC_DERIVED_DWORDS)` AoS layout packs a contact's ~15 fields into ~1 cache line, cutting lines/contact on the scattered 56%.
- Implemented cleanly via two `_derived_read`/`_derived_write` wrappers (kept the `(field, k)` accessor call signature, so zero index-swap risk). **Bit-identical** (kapla regression test passes).
- **Perf: neutral-to-slightly-worse.** Bracketed A/B (AoS, SoA, AoS to cancel thermal drift): SoA 66.83 fps vs AoS 66.33 / 66.26 — SoA faster by ~0.8% despite running between the two AoS runs. The derived working set (~16 dwords × ~71k contacts ≈ 4.5 MB) is **L2-resident**, so the SoA strided reads hit L2 cheaply (high bandwidth, fine sectoring); AoS gives back the coalescing on the 45% overflow colour and reads the full 16-field line even though ~15 are used. The original k-inner SoA is the right call.
- **Could not measure the transaction-count delta directly:** GPU performance counters are blocked on the dev box (`ERR_NVGPUCTRPERM`) so ncu SpeedOfLight / nsys gpu-metrics are unavailable without admin. Roofline analysis puts `contact_iterate_at` at ~1.5 flop/byte (vs ~70 ridge) → firmly **memory-bound**; the lever is bytes/transactions, not flops, but the L2-resident working set means layout tweaks don't move it. Reverted.

### Cooperative grid-sync iterate megakernel (single-world)
- 2026-06-25: collapsed the per-colour single-world PGS iterate launches into ONE cooperative kernel that walks all colours internally with a grid-wide `wp.kernel_sync()` barrier between colours (Gauss-Seidel ordering), grid-striding each colour across a co-resident grid. Built on the local Warp `dev/tw/cooperative_launch_experiment` branch (`wp.kernel_sync()` + `wp.launch(cooperative=True)`).
- **Both feasibility gates pass:** cooperative launch *does* capture into a CUDA graph and replay correctly under standard `wp.ScopedCapture` (the branch only blocks cooperative under *APIC* capture). Occupancy is ample — a register-heavy iterate co-resides ≥1504 blocks (8/SM) on the RTX PRO 6000, far more than the ~126 the 32k-contact overflow colour needs.
- **Bit-identical:** `max|Δ| = 0` on brick positions over 60 frames vs the per-colour path (non-overflow colours are an independent set, so thread→cid assignment is irrelevant; overflow `parallel_id` is the slot index, independent of grid size).
- **Perf: neutral.** Kapla 65.2-65.8 fps with the megakernel vs 65.5-65.7 baseline, flat across grid sizes {188, 376, 752, 1504} (median of 3, interleaved). Collapsing ~90 per-colour launches/substep into one buys nothing because the single-world PGS solve is **work-bound, not launch-bound** — the per-colour launch/spin-up overhead in the captured-graph persistent-grid design is already negligible, and the grid-wide barrier across many blocks costs about what the launch saved. Corroborates the earlier flat `NUM_INNER_WHILE_ITERATIONS` sweep.
- **Don't redo for the solve.** Inlining the mass-splitting average/broadcast wouldn't help either — its 5.2% is averaging *work*, not launch overhead. A grid-sync megakernel could still pay off on a scene that is genuinely launch-bound (many tiny colours / near-no-op launches), but kapla and the like are not. Reverted (kept stock-warp compatible).

### Substep mega-kernel (one block per world, all substeps in one launch)
- Goal: collapse the entire `num_substeps` loop (forces, prepare, iterate, integrate, relax, inertia refresh, kinematic, damping, accumulate) into a single block-per-world kernel using the existing per-world body / constraint CSRs.
- Implemented and tested with `block_dim ∈ {32, 64, 128}`. Results were mixed: some configs +5-10%, others -5-10%. Net ~neutral.
- Why it didn't pay off:
  - Register pressure: a single mega-kernel inlines forces + prepare + iterate*N + integrate + relax*M + refresh + ...; occupancy drops vs. smaller specialised kernels.
  - At `block_dim=32` (one warp per world) `__syncthreads()` collapses to warp-sync, but the kernel is bandwidth-bound on body / constraint state and the launch-overhead saving is small relative to GPU time at the world counts we benchmark.
  - At `block_dim=128` idle lanes during small per-color cid lists waste occupancy.
- **Don't redo without a fundamentally different design** (e.g. splitting body work and constraint work into two mega-kernels, or moving body state into a packed struct).
- Reverted in commits `2566ef65 / 3216a44a / cb4dfcef`.

### `_FUSED_INNER_SWEEPS > 1`
- Local multi-sweeps repeat a color before later colors see the new impulses. This is not equivalent to global PGS order on coupled contact/joint graphs.
- `_FUSED_INNER_SWEEPS = 2` was +15-22% env_fps on multi-world G1/H1, but made G1 contact-drive multi_world diverge from single_world for the same `solver_iterations`.
- `_FUSED_INNER_SWEEPS = 4` also broke `test_slam_ball_into_stack`; heavy impacts need fine cross-color feedback to dissipate without driving bodies through neighbours.

### Body-hot AoS pack (``BodyHot`` struct + per-substep sync)
- Added a packed ``BodyHot`` ``wp.struct`` holding the four iterate-time read-only fields (``inverse_mass``, ``body_com``, ``orientation``, ``inverse_inertia_world``) and a parallel ``bodies.hot`` AoS array. ``_sync_body_hot_kernel`` snapshotted the SoA fields into the AoS mirror once per substep before the constraint solve. Iterate / prepare / relax read the four fields with one wide gather per body instead of four separate SoA gathers.
- **Net loss.** Single-world kapla iterate per-launch was unchanged; multi-world g1_flat / h1_flat regressed 1-6% across 1024-16384 worlds.
- The SoA layout already serves cache-line-shared loads across warp lanes touching different (but index-adjacent) bodies; the AoS struct kills that sharing because each thread's 68 B struct lives in a different cache line. The compiler also already keeps the SoA fields in registers across the inner loop, so the read-coalescing the AoS pack was supposed to provide didn't materialise.
- Reverted in this session before commit. **Don't re-try the same shape unless you also redesign the access pattern** (e.g. groupings of cids that share bodies, or a thread-block-cooperative load).

### Body-hot AoS pack v2 (``BodyIterateHot``, post-locality-sort)
- 2026-05-11: tried again, now that the CSR body-locality sort puts cids sharing a body in adjacent positions within a colour. Hypothesis: locality sort makes warp lanes touch adjacent bodies, so AoS would win where SoA wider-gather no longer helps. 5-field struct (added ``position`` to the 4 the first attempt used); single ``_phoenx_pack_iterate_hot_kernel`` repopulates from SoA twice per substep (entry + after integrate_positions).
- **Single-world kapla: +7-14% FPS** across all 4 ms/iter configs, drift flat or improved.
- **Multi-world regressed again**: g1_flat 4096 **-12.0%** (1.85M → 1.62M env_fps), h1_flat 4096 **-5.5%** (2.49M → 2.35M env_fps). Same root cause as v1 — multi-world warp lanes serve *different worlds*, not adjacent bodies in one world, so the SoA cache-line sharing the AoS pack kills was load-bearing in a way the locality sort doesn't fix.
- Reverted in commits ``1420a63b`` (the pack) and ``db589d9d`` (revert).
- **Suspect the headline +7-14% is partly Step-1's contribution**: see v3 below; baselining matters.

### Body-hot AoS pack v3 (single-world gated, post-locality-sort)
- 2026-05-11 (same session as v2): retried with ``use_aos: bool`` threaded through the ``_make_contact_{prepare_for_iteration,iterate}_at`` factories via ``wp.static``, plus a parallel factory for the entry-point wrappers so no hand-written wp.func wrappers were duplicated. Single-world kernel factories called the ``*_aos`` variants; multi-world fast-tail kept SoA. Pack kernel guarded by ``step_layout == "single_world"``.
- **Result: neutral.** Kapla single-world +0.3-1.0% over the pre-AoS baseline (53.12 → 53.58, 34.72 → 34.54, 55.91 → 56.22, 37.27 → 37.39). Multi-world unchanged within noise (g1_flat 1.82M, h1_flat 2.47M).
- The v2 "+7-14% Kapla" headline was measured against the **pre-Step-1** baseline JSON; Step 1's CSR body-locality sort had already captured the bulk of the win. AoS alone contributes only ~+1% once locality sort is in place.
- **Don't re-try without re-checking against a *current* baseline.** The pack kernel + 4 extra factory variants doubled the compiled binary size for these funcs while contributing under a percent. Reverted (not committed).

### `__ffsll` for the greedy first-free-colour scan
- Wrapped the CUDA ``__ffsll`` intrinsic in a ``wp.func_native`` (``_lowest_set_bit``) and used it in both single-world and per-world greedy kernels in place of the 64-iteration linear bit-scan.
- Per-launch greedy kernel time was unchanged (~52us on kapla single-world, multi-world bench unchanged). The original scan had an early-out on first set bit, and most masks have a low-set bit, so it rarely paid the full 64 iterations. The intrinsic is cleaner code but doesn't move the perf needle.
- Kept the change for clarity; PR-reviewable as a one-line refactor that removes the open ``__ffsll`` TODO.

### Single-world multi-sweep iterate
- Tried wiring local multi-sweeps into the single-world iterate path (call `*_iterate_multi(num_sweeps)` instead of `*_iterate`) and halving the outer `solver_iterations` loop.
- ~3% kapla regression (single-world contact-heavy scene). The body-load saving exists but the per-launch cost grows ~2x and you lose half the cross-colour feedback granularity.
- Multi-world wins because the entire substep runs in one launch, so launch-overhead saving stacks with body-load saving. Single-world has many head+tail launches per sweep, so the launch-overhead saving doesn't materialise.
- **Keep single-world on `num_sweeps = 1`.**

### Body wp.func extraction → per-block-per-world dispatcher
- Extracted every per-body kernel (`_phoenx_apply_forces_and_gravity_kernel` etc.) into `wp.func` helpers taking a body id, planning to grid-stride them inside a future mega-kernel.
- The mega-kernel itself didn't pay off (see above), so the extraction was reverted to keep the diff minimal. The funcs are cheap to re-add if a future fused design wants them.

### Hoist `set_access_mode_unified` out of soft-tet / cloth-tri iterate
- 2026-05-12: tried removing the per-iterate `set_access_mode_unified` calls (4 per soft-tet, 3 per cloth-tri) on the theory that ``*_prepare_for_iteration`` already flips every vertex to POSITION_LEVEL once per substep entry.
- **Breaks momentum conservation.** ``test_soft_body_mass_splitting_momentum.test_normal_impulse_balances_weight`` failed with 24% rel-err on time-averaged contact normal-impulse (2.31 vs M*g*dt = 1.86 N·s, threshold 1%). Root cause: a body shared between a soft-tet shear constraint (POSITION_LEVEL) and a contact constraint (VELOCITY_LEVEL) gets its access mode flipped by *both* within the same PGS sweep. Without the iterate's defensive re-flip, a subsequent soft-tet iterate reads stale state after a contact iterate (different colour, same body) has flipped the mode to VELOCITY_LEVEL.
- **The "re-flip every iterate" pattern is correctness-load-bearing, not paranoia.** The C# FemTetPBD reference encodes this for the same reason. Don't re-hoist.

### Algebraic simplification: ``g1 = -(g2 + g3 + g4)`` (soft-tet shear gradient)
- 2026-05-12: replaced the explicit Jitter2 form ``g1_* = -2 * (kz*cs + ld*cg + li*cm) * my`` (where cs/cg/cm are column sums of inv_rest) with the algebraically exact ``g1 = -(g2 + g3 + g4)``. Saves ~6 FMAs per iterate.
- **24% momentum-balance drift on ``test_normal_impulse_balances_weight``.** The transformation is exact in real arithmetic but in FP32 with FMA fusion the rounding pattern differs from the original, and the per-iterate rounding error compounds across 200 settle frames × 5 substeps × 5 iters of XPBD position updates into a >1% drift on the time-averaged contact impulse.
- **Don't re-try this kind of "algebraically exact" microoptimisation inside the XPBD inner loop.** Single-precision iterative solvers are sensitive to rounding order; the original Jitter2 form is empirically more numerically stable.

### Polar-decomp iteration cap 15 → 4 (soft-tet shear iterate)
- `_extract_rotation_3d` in `constraint_soft_tetrahedron.py` caps the Mueller quaternion-axis polar decomposition at 15 iters. Tried lowering to 4 on the theory that warm-started tets converge in 2-3 iters anyway.
- **Neutral.** Per-launch avg on the fused PGS iterate kernel: 282.8 μs (15 iters) → 298.2 μs (4 iters) on soft_body_drop steady contact — within the 4-5% measurement noise floor (an unchanged kernel showed +4.2% between two baseline runs). The convergence break-out at `if w_mag < _EXTRACT_ROT_EPS: break` already short-circuits in practice; the 15-cap only catches pathological non-convergent cases.
- **Don't re-try.** Warm-start + break-out already captures the win.

### Lean greedy coloring (mass-splitting-aware fixed-iter MIS)
- 2026-05-12 Premise: with mass splitting on, reaching minimum colour count doesn't matter because the overflow bucket is consumed by copy-state slots. So we should be able to skip `wp.capture_while` entirely and run a fixed `K × 2 = 24` JP-luby launches (the C# experimentalsim pattern), force-spilling any leftovers into the overflow bucket via a small kernel.
- **Looked like a -13% win** on `example_soft_body_drop` steady-contact: 13.18 → 11.49 ms/frame (5000-frame avg of two runs), -64% on the coloring kernel and -29% on the fused PGS iterate.
- **Then failed `test_cloth_mass_splitting.test_mass_splitting_on_regression`:** the rigid cube fell through the pinned cloth (cube_z=0.27 instead of expected >1.8). Root cause discovered by binary-search through variants:
  - With 24-64 fixed greedy launches the dense cloth contact graph doesn't converge — many constraints stay uncoloured. Without spill, those constraints get `interaction_id_to_partition = -1` and are silently SKIPPED by the iterate (explains apparent perf gain — work was being dropped, not done faster).
  - With spill, the uncoloured constraints land in the overflow bucket. Mass-splitting averages impulses across overflow batches (`_average_and_broadcast_kernel`), and when the bucket holds many normal cloth-cube contacts the averaging dilutes the cube-supporting normal forces below the threshold needed to catch the cube.
  - Empirically, the cloth scene needs >64 inner greedy launches per build to converge; soft-body drop steady state needs ~64. A fixed cap can't simultaneously be under-baseline for soft-body and over-cloth-convergence.
- **Don't re-try a fixed-iter cap on top of the existing kernel.** A stall-detection variant (zero `num_remaining` to force-exit `wp.capture_while` only when commits have stalled for N rounds) is the right pattern in principle — only genuinely-stuck elements would spill, and mass splitting handles small overflow correctly. But validating perf wins at sub-5% requires either GPU clock locking or many-run averaging; under our current noise floor the win wouldn't be confidently distinguishable from noise. Not worth a re-try without a tighter measurement harness.

### 3-vertex (surface-triangle) soft-tet contact endpoint — investigation only
- 2026-05-12 user-proposed: drop the 4th tet vertex from the **contact endpoint storage and iterate**, not just the coloring. Renormalise the 3 surface-triangle bary weights to sum to 1.
- **Feasibility: yes.** Narrow-phase already emits `bary: wp.vec3f` with the 4th tet weight derived as `1 - sum(bary)` (see `_side_world_contact_point` for soft-tet kind in `constraint_contact_cloth.py`). Identifying the face is `argmin` over the four implicit weights.
- **Win on top of the coloring-only variant (see "Soft-tet contact interaction-element..." above) is likely under noise.** The coloring + interaction-graph win already lands the headline; cutting the iterate's 4 → 3 position reads gains only the iterate-bandwidth fraction (~5% of frame). Not worth the cost: narrow-phase output, contact endpoint storage (vec4i → vec3i), contact ingest, and the cloth-aware contact iterate would all need parallel modifications.
- **If pursued later**, the trigger is: contact iterate becomes the dominant share of the fused PGS kernel, OR `MAX_BODIES` is shrunk to 6 (which requires this change first).
- **Cost is high.** Changes narrow-phase output, contact endpoint storage (vec4i → vec3i), contact ingest, contact iterate (both rigid and cloth-aware paths), and every reader of `side*_nodes_extra`.
- **Don't pursue** unless a future scene shows soft-tet contact iterate as the per-kernel dominant cost. On the soft-body drop scene, the iterate is dominated by tet shear, not tet contacts.

## Knobs and where they live

| Constant / flag                     | File                                              | Effect                                           |
| ----------------------------------- | ------------------------------------------------- | ------------------------------------------------ |
| `_STRAGGLER_BLOCK_DIM`              | `solver_phoenx_kernels.py`                        | Multi-world fast-tail warp size (= 32)           |
| `_choose_fast_tail_worlds_per_block`| `solver_phoenx_kernels.py` / `solver_phoenx.py`   | wpb tier plus robot-fleet cap in `_fast_tail_worlds_per_block` |
| `PHOENX_USE_GREEDY_COLORING`        | `solver_config.py`                                | Greedy vs round-based JP                         |
| `FUSE_TAIL_MAX_COLOR_SIZE`          | `solver_config.py`                                | Single-world fused-tail hand-off threshold       |
| `FUSE_TAIL_BLOCK_DIM`               | `solver_config.py`                                | Single-world fused-tail block size               |
| `NUM_INNER_WHILE_ITERATIONS`        | `solver_config.py`                                | Single-world capture-while host unroll factor    |
| `GREEDY_MAX_COLORS`                 | `graph_coloring/graph_coloring_common.py`         | Forbidden-mask bit width (= 64)                  |
| `_SINGLEWORLD_BLOCK_DIM`            | `solver_phoenx.py`                                | Persistent-grid block size (= 256)               |
| `_PER_WORLD_COLORING_BLOCK_DIM`     | `solver_phoenx_kernels.py`                        | Multi-world per-world coloring block size (= 64) |

## Correctness gotchas (example / integration)

### CUDA-graph capture + odd-count state-swap pins the simulation in a tiny cube

Symptom: in a captured-graph example loop, every body appears trapped in a ~1 cm box. Joints articulate (they're driven by ``Control`` arrays the user can update from host code each frame), but the chain as a whole drifts a few mm and snaps back. Without graph capture (``ex.graph = None``) the same code progresses normally. Easy to misdiagnose as a constraint, drive, or contact issue inside PhoenX -- it isn't. The bug is in the example's own per-frame state-swap pattern.

The pattern that breaks (one outer ``solver.step`` per frame, one swap):

```python
def simulate(self):
    self.model.collide(self.state_0, self.contacts)
    self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.frame_dt)
    self.state_0, self.state_1 = self.state_1, self.state_0   # ← single swap
```

Why it fails under capture:

1. The captured kernel chain binds the ``body_q`` / ``body_qd`` ``wp.array`` buffers it reads at capture time -- call them ``sA`` (read) and ``sB`` (write).
2. The Python-level ``state_0, state_1 = state_1, state_0`` rebinds attributes; it is **not** captured.
3. After capture, ``self.state_0`` aliases ``sB`` (the just-stepped buffer).
4. Every subsequent ``wp.capture_launch`` replays the same sequence: read ``sA``, write ``sB``. ``sA`` is **never written** between replays, so each frame re-integrates the same starting pose into ``sB``. ``sB`` oscillates around ``step(initial)`` with float-noise amplitude -- the "tiny cube".

Why most existing examples are fine: ``example_robot_h1`` and ``example_robot_anymal_d`` use ``sim_substeps=4`` with the swap inside the inner loop. Four swaps end with ``self.state_0`` rebound to its original buffer, so the captured chain reads ``sA``, writes ``sB``, reads ``sB``, writes ``sA``, ... and each replay correctly carries state forward. They get this for free because the count is even. Any example that does **one** outer step per frame (or any odd count) hits the bug.

Two safe patterns:

```python
# (a) Even outer-substep count: swap inside the loop, count must be even.
for _ in range(self.sim_substeps):  # sim_substeps even
    self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
    self.state_0, self.state_1 = self.state_1, self.state_0
```

```python
# (b) Single step + explicit copy-back. No swap, no even-count requirement.
self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.frame_dt)
wp.copy(self.state_0.body_q, self.state_1.body_q)
wp.copy(self.state_0.body_qd, self.state_1.body_qd)
```

(b) is the right pattern when the example wants PhoenX's internal substepping to do all the work and only one ``solver.step`` per frame. Reference: ``newton/examples/robot/example_robot_dr_legs_phoenx.py``, where (b) was used after this bug was diagnosed.

If you ever see a graph-captured PhoenX scene where bodies appear to be on rails / trapped in a small region: drop ``ex.graph = None``, re-run, and compare. Identical motion -> there is a real solver issue. Wildly different motion -> it's almost certainly the swap pattern, not PhoenX.

## Open ideas (not yet attempted)

- **Drop the `partition_data_concat` int64 write entirely** — would require updating the JP-fallback to also write `color_tags`. Saves ~1 byte/8 bytes/commit and unifies the read path. Modest win since commits are only ~3K/round.
- **Drive / limit PD spring/damping split** — same XPBD-style split that cable now does, applied to ``_axial_drive_limit_iterate`` (revolute drive PD, prismatic drive PD, PD limit rows). Blocked on column layout: prismatic mode_extras is fully consumed by anchor-3 state, so a ``damp_mass_drive`` / ``damp_mass_limit`` slot needs new dwords on the constraint struct. Worth it whenever users start hitting "high damping kills convergence" on drive PD too.
- **Reduce greedy kernel launch count** — ~82 MIS rounds per step on kapla = ~82 launches × ~5µs overhead. A persistent kernel running all rounds with global atomics + sync flags could collapse that. Cross-block sync is the main hurdle.

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
