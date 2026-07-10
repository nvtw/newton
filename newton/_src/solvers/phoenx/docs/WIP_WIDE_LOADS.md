# WIP: Lossless wide loads for per-cid body fields (PHOENX_BODY_WIDE_LOADS)

Status 2026-07-10: implemented + bit-identity-gated + PTX-verified. Perf
brackets NOT yet run (machine shutdown cut the session before timing).

## What was implemented

Env flag `PHOENX_BODY_WIDE_LOADS` (default OFF = untouched baseline):
`1`/`all` enables everything; comma list selects fields: `vw`, `quat`, `sym6`
(debug: `vwlayout` = interleaved storage without wide kernel access).

Fields widened (all lossless — same bytes, fewer load requests):

- `vw` — velocity + angular_velocity interleaved per body into one 24 B slot
  (2N-element vec3 buffer). `bodies.velocity` / `bodies.angular_velocity`
  become **strided vec3 views** (stride 24 B) into it, so all ~230 existing
  reader/writer sites are unchanged. Hot path loads/stores the pair as three
  8-byte words (`ld/st.global.v2.f32`) via `wp.func_native`. Only field that
  needed a layout change (two separate 12 B vec3 arrays can't be 8-byte
  aligned per element).
- `quat` — `orientation` read as one 16-byte load (`ld.global.v4.f32`) via a
  u64 alias view. Zero storage change.
- `sym6` — `inverse_inertia_world` (24 B) read as three 8-byte words via a
  u64 alias view. Zero storage change.

### Files

- `body.py` — flag parsing; native snippets `_load_vw_wide` / `_store_vw_wide`
  / `_load_quat_wide` / `_load_sym6_wide`; `wp.func` accessors `body_load_vw`,
  `body_store_vw`, `body_load_orientation`, `body_load_inv_inertia_sym6`
  (all `wp.static`-gated → flag-OFF compiles to the identical scalar code);
  allocation helpers `body_alloc_velocity_storage` + `body_attach_wide_aliases`;
  new `BodyContainer` fields `velocity_pair_u64`, `orientation_u64`,
  `inverse_inertia_world_u64` (dummies when off/empty).
- `world_builder.py` — second construction site routed through the same
  helpers (both sites consistent).
- `constraints/constraint_joint.py` — `_ms_load_body_pair` /
  `_ms_store_body_pair` fast paths (covers all D6/revolute multi iterates);
  `_d6_prepare_rows_at` + `_d6_angular_limits_block` orientation loads.
- `constraints/constraint_contact.py` — `contact_iterate_at_multi` (fast-tail
  multi-sweep contact path): vw pair load/store, quat, sym6.
- `constraints/constraint_contact_cloth.py` — rigid lean prepare
  (`_make_contact_prepare_for_iteration_at`: quat + sym6 + warm-start vw
  scatter), `contact_cached_warmstart_lean`, `_make_contact_iterate_at`
  lean loads + fast writeback.

Cold paths (reporting/wrench/init, ms slot-mixed branches, cloth barycentric
side helpers) intentionally left on scalar reads.

## Validated so far

1. **Bit-identity gate PASSED** for `vw`, `quat`, `sym6`, `all` vs baseline:
   deterministic scenes (test_determinism 4-layer contact pyramid, 60 frames
   x 10 substeps, multi_world layout + test_basic_joints revolute/prismatic/
   ball scene, 600 substeps); position/orientation/velocity/angular_velocity
   byte-identical (`/tmp/widetest/gate_dump.py` pattern — recreate: build both
   scenes, step, np.savez body state, compare uint32 views across processes
   with the env var set).
   NOTE: full-frame A/B on the bench scenarios (dr_legs etc.) is NOT byte-
   stable even baseline-vs-baseline — the per-world greedy coloring emits
   different colorings run-to-run (`_world_num_colors` / `_elements` differ;
   collide output itself IS deterministic). Don't chase byte-identity there;
   use the deterministic test scenes.
2. **PTX evidence** (real kernels, separated caches): baseline fast-tail
   prepare+iterate has ZERO vector loads; flag=all has 48x `ld.global.v2.f32`,
   6x `ld.global.v4.f32`, 24x `st.global.v2.f32` (relax kernel: 24/2/12).
3. Strided-view mechanics (fill_/zero_/numpy/clone/kernel access in structs)
   verified in isolation.

## Remaining (blocked by shutdown)

- Brackets (graph-captured, 300 replays, median of 3, per PERF_NOTES vec6
  methodology): dr_legs @4096 + h1 @4096 env_fps (block_world_32 path) via
  `benchmarks/bench_multi_world_scheduler.py` or the `/tmp` harness pattern;
  kapla single-world control (`bench_phoenx_kapla.py`). Measure `vw` alone
  first (stop if neutral), then `all`.
- Focused tests: `test_multi_world_scheduler_helper`, plus 2-3 fast-tail /
  block-world contact tests (`test_determinism`, `test_stacking`,
  `test_basic_joints` are the ones the gate already exercised indirectly).
- If `vw` wins: consider wide stores in `_integrate_velocities_kernel` (per-
  body coalesced, likely neutral) and enabling by default + removing flag.
- A regression test (repo `tests/test_body_wide_loads.py`) for primitive
  bit-identity + interleaved-layout equivalence still needs to be written.

## Container-rewrite opportunities noted (not implemented)

- Coordinator allowed layout rewrites: a 32 B padded (vec4+vec4) velocity
  block would give two 16-byte loads instead of three 8-byte, but adds
  +8 B/body traffic on a bandwidth-bound kernel — the PERF_NOTES uvec4 lesson
  (extra bytes ≈ net loss) argues against; the 24 B interleave keeps bytes
  identical. Measure before believing either.
- Alignment-first regrouping of the remaining hot iterate fields
  (`inverse_mass` 4 B + `body_com` 12 B = natural 16 B block → one v4 load;
  `position` 12 B could pad to 16 or pair with `inverse_mass`) — same
  strided-view trick applies; prior Body-hot AoS failures (PERF_NOTES) do NOT
  apply since SoA per-field cache-line sharing is preserved.
- Same technique is applicable to the contact/joint containers
  (`ContactContainer` per-contact vec3 normal/tangent/anchors are scalar
  loads today; `read_vec3` on the column container is 4 B strided).
