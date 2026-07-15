# PhoenX Performance Notes

Curated lessons-learned from past performance work on the PhoenX solver. Goal is to avoid re-trying ideas that have already been characterised, capture *why* a change won or lost, and surface knobs that scenes can opt into.

This is **not** a substitute for `git log` — it's a hand-maintained shortlist of the load-bearing decisions and the dead ends.

## Training-throughput predictive model (2026-07-11)

**training gain % ≈ (isolated bandwidth win %) × (phase's share of training GPU %).**
G1 graph_leapfrog training is rollout-bound with physics on the critical path
(rollout ~0.45 s/iter >> update ~0.14 s/iter, 96% union-busy), so a solver-phase
win scales down by that phase's GPU share. Verified: fp16x2 rows = +9.4% on the
contact-row phase (~11% of training GPU: build 5.9% + solve 4.9%) → predicted ~+1%,
**measured +1.06% G1 training (clean, 3x ABAB: fp16x2 958.2-959.7k vs fp32
947.6-949.7k, no overlap).** This is the first confirmed training-throughput win
and the shipped default.

Consequences for prioritization:
- Only target phases that are BOTH large AND bandwidth-bound. Training GPU
  ranking (trace): advance 8.6% > factor 6.2% > row build 5.9% > contact solve
  4.9% > publish 4.5%.
- Constraint-iterate body/joint fields and the ABA advance/factor inertia loads
  are latency-hidden / L2-resident → width and coalescing wins there land
  sub-noise (rejected, this session). The contact ROW stream (rows x nv, large
  at 8192 worlds) is the one genuinely bandwidth-bound big phase → fp16 (shipped)
  and row-count reduction (patch rows) land.
- Composition is the path: each win is ~1%, so stack row-stream reducers
  (fp16x2 shipped, patch rows next: ~25% fewer rows x 11% phase → ~+1.5-2%).

## Measured architecture laws (2026-07-10)

Three contact-solve architectures were compared head-to-head on identical
8192-world G1 snapshots (validated outputs, graph-captured reversed brackets):

- **Global row streaming (production) wins at training scale.** On-chip
  warp-per-world (rows in smem) loses 13% ex-gather at 8192 worlds (residency
  ~3 worlds/SM at ~40 KB/world) though it wins 1.65x at 512 worlds; a
  matrix-free solve (no response rows, one O(depth) tree pass per sweep) is
  0.36x — each tree pass costs as much as a full dense sweep because one warp
  walks a ~30-joint latency chain while production streams rows at full
  occupancy. Law: *at high occupancy, streaming rows x nv is cheap; per-sweep
  factorized-solve latency is expensive.* Oracles retained at
  `benchmarks/experimental/bench_warp_world_contact_oracle.py` and
  `bench_matrix_free_reduced_contacts.py`.
- **Warp emits only scalar global loads for all vector types** (PTX-verified:
  zero `ld.global.v2/v4` in any hot kernel; a vec6 access = 6 strided 4-byte
  requests). Wide loads require aliased u64/uint4 views + a minimal
  `wp.func_native` unpack (pattern in `reduced_contact_block.py`
  `_load_spatial_wide`, proven bit-identical with `ld.global.v2.f32`). Where
  it pays: gather-style per-cid accesses (classic fast-tail/block-world body
  fields — follow-up in flight). Where it does not: the reduced row arrays at
  8192 worlds are already sector-efficient (neighbor lanes share sectors);
  wide-loading `body_response` measured +2.4% at 512 worlds, noise at 8192.
- **Load-request count, not just bytes, is the lever on the contact phase**:
  naked fp16 rows (same request count as fp32) gained nothing (0.98x); packed
  fp16x2 (full 4-byte words) gained +9.4%; fp16x4/x8 add nothing because
  pack>=2 already reaches one warp request per row load (x4 measured +8.3%,
  slightly worse). fp16x2 is the packing optimum for this layout.

## Active wins

### FP16x2 packed contact rows (2026-07-10) - retention in progress
- `PHOENX_CONTACT_ROWS_FP16=2`: packed_jacobian/packed_response stored as
  fp16 pairs in uint32 words (func_native `_unpack_h2`), all accumulators,
  lambdas, velocities, effective masses FP32. Isolated contact phase +9.4%
  at 8192-world G1 (2600 -> 2376 us); nsys inside real training confirms
  build -11% / solve -4%. Deviation vs fp32 <= 3e-4 rel (rms 7.6e-6),
  bit-stable replays, 56/58 reduced battery passes including long-horizon
  momentum/energy and Featherstone parity, far-translation invariance passes
  (rows are COM-relative).
- Seed-lottery evidence: all five compared seeds match fp32 iteration-75
  screens within 0.02, but 0/4 full-horizon draws passed the final 0.90 gate
  (fp32's own rate is ~1/6-1/3; not statistically distinguishable). Retained
  by explicit user decision on the physics evidence; expected-cost baseline
  stays fp32-derived until successes accumulate under the new default.
- Landing chores: fix `test_generalized_contact_rows_match_aba` (launches the
  module-level fp32 kernel directly against fp16 arrays) and add a tolerance/
  mode-guard for its exactness assertion; add an fp16-mode regression test;
  flip default to mode 2 with fp32 opt-out.
- Methodology lesson: any physics-affecting change (even 3e-4) re-rolls the
  seed lottery. Iteration-75 screens are robust cross-physics comparators;
  full-gate races are expensive and low-powered at n<=4 — budget them only
  for candidates whose prize justifies it.

### Register-resident generalized contact delta (2026-07-03)
- A user-assisted one-launch Nsight Compute report measured the reduced
  generalized contact solve at 52% long-scoreboard stalls, only 0.62 eligible
  warps per scheduler, 44.5% achieved occupancy, 50.7% L1 hit rate, and 69.8%
  L2 hit rate. The kernel is latency-bound; another full shared row cache would
  reduce occupancy and had already regressed 14%.
- The accumulated generalized-velocity delta now remains in a distributed
  register tile across warmstart and PGS rows instead of round-tripping through
  shared memory. Equations, contact order, tile reductions, and deterministic
  solve order are unchanged; only tile storage changes.
- Reversed 300-replay contact-rich G1 brackets improve 1.626M to
  1.636-1.640M physics steps/s (+0.6-0.9%). Reversed graph-leapfrog brackets
  improve 1.139M to 1.150-1.154M samples/s (+1.0-1.3%).
- The complete 40-test reduced CUDA-graph battery passes (39 together plus the
  recurring driver graph-instantiation case alone), covering 4/8/36/64-DOF
  widths, arbitrary contact pages, loops, ABA/Featherstone parity, momentum,
  energy, determinism, self-contact, cloth, and live mass updates. The
  graph-leapfrog train-to-gate smoke also passes.

### Mask post-reset G1 observation work (2026-07-03)
- Auto-reset already produces an exact per-world articulation mask. The second
  observation pass now uses that mask, so reset worlds still recompute the
  identical observation, reward, termination, success, and timer state while
  non-reset worlds keep the values produced by the first pass. Periodic command
  resampling deliberately retains a full observation pass.
- A matched ten-replay Nsight Systems bracket reduces total G1 observation
  kernel time from 5.498 to 3.927 ms (-28.6%). The controlled short
  graph-leapfrog benchmark improves from 1.15705M to 1.16033M samples/s
  (+0.28%); this is retained as a small exact rollout-path win.
- A CUDA-graph regression proves selected outputs are bit-identical to a full
  pass and every unselected output remains bit-exact. Observation-contract,
  command-resampling, reset/FK, contact-warmstart, and train-to-gate graph tests
  pass. A stochastic no-update threshold failure reproduces on the committed
  baseline (0.883 done fraction), so it is not attributed to this change.

### Statically omit provably empty GJK/MPR stages (2026-07-03)
- CollisionPipeline now proves from the complete collision topology whether any
  pair can reach the generic convex stage. Explicit broad phases inspect every
  configured pair; NXN/SAP conservatively inspect every geometry-type
  combination. External shapes and every unproven case retain GJK/MPR.
  Eligible graphs omit the node at construction, so there is no runtime
  conditional, host synchronization, tuning knob, or contact approximation.
- Fresh steady-training profiling had attributed 4.3% of kernel time to GJK/MPR
  even though canonical G1 produces only analytic plane-cylinder and plane-box
  pairs. A same-session short-training bracket improves about 880k to 891k
  samples/s (+1.2%). Physics-only is neutral (1.610M original versus 1.609M
  specialized), as is H1 at 4096 worlds (3.229M versus 3.226M); the measured
  training gain comes from removing contention on the overlapped rollout path.
- The specialized and forced-original graphs produce bit-identical contact
  arrays. A real box-box scene proves the generic fallback remains active.
  Long deterministic/sticky collision tests pass. The 40 reduced tests pass
  (39 together plus the recurring driver graph-instantiation case alone), and
  contact-rich Anymal/H1/G1 screens remain finite in multi-world and
  many-articulation single-world layouts.
- Two related contact-solve traffic experiments were rejected: keeping
  generalized delta in shared memory was neutral (+0.1%), while forwarding
  parent body delta through registers lost about 0.5% from added live
  registers and shuffles.


### Lossless compact primitive contact sort (2026-07-03)
- Primitive-only deterministic pipelines now rank valid multi-world shape pairs
  into a lossless positive int32 key. Prefix and suffix global shapes retain
  full-width second-shape intervals; world-local shapes reuse only the
  per-world rank interval. Three low bits preserve every analytic and GJK/MPR
  manifold row. Mesh, convex-mesh, heightfield, SDF, external-shape, oversized,
  and expert-component pipelines automatically keep the original int64 sort.
- The G1 contact sort drops from eight int64 radix passes to four int32 passes.
  A matched 300-replay bracket improves 1.574M to 1.599M physics steps/s
  (+1.54%). Short graph-leapfrog training measures 877.7k/s versus the retained
  875.9k/s (+0.2%, within short-run noise), so this is retained as a modest,
  general collision-pipeline win rather than a headline training breakthrough.
- CUDA-graph tests exhaust all 90 valid pair intervals across prefix globals,
  three worlds, suffix globals, and subkeys 0-7. The active full-contact path
  is bit-identical to the int64 fallback, including sorted keys and sticky
  matching. Both long deterministic collision tests pass; the 40-test reduced
  suite passes (one transient graph-instantiation error passed in isolation).
  Contact-rich Anymal/H1/G1 fleets remain finite in 512 multi-world and
  many-articulation single-world layouts.


### Symmetric-packed live body inertia (2026-07-03)
- The pose-transformed per-body spatial inertia is also symmetric. Store its 21 independent entries once, then reconstruct on load in factor and both ABA advances; this reduces initialization stores and three hot tree-pass reads without changing equations.
- Reversed 300-replay G1 bracket: 1.565M to 1.593M physics steps/s (+1.8%). Short graph-leapfrog training improves 851k to 862k samples/s (+1.3%).
- All 40 reduced CUDA-graph tests pass; contact-rich 512-robot Anymal/H1/G1 screens remain finite. Together with the other two factor-layout wins, current same-session measurements improve about 1.539M to 1.593M physics (+3.5%) and 829k to 862k training (+4.0%).

### Symmetric-packed reduced inertia (2026-07-03)
- Articulated-body reduction matrices are symmetric by construction. The internal child-to-parent cache now stores their 21 independent upper-triangle entries and reconstructs the symmetric 6x6 matrix on load, instead of moving 36 floats between tree depths.
- Reversed 300-replay G1 bracket: 1.548M to 1.572M physics steps/s (+1.6%) on top of dead-store removal. Short graph-leapfrog training improves 842k to 851k samples/s (+1.1%).
- All 40 reduced CUDA-graph tests pass, including mass-matrix/ABA parity, Featherstone comparisons, long-horizon momentum and energy, loops, self-contact, and live mass updates. Contact-rich 512-robot Anymal/H1/G1 screens remain finite.

### Remove unused articulated-inertia stores (2026-07-03)
- Both reduced factor kernels wrote a full 6x6 `articulated_inertia` matrix per body, but no solver path ever read that buffer. Removing the dead output is exact: factor equations, reduction order, and public state are unchanged.
- Reversed 300-replay G1 bracket: 1.539M to 1.556M physics steps/s (+1.1%). Device use falls by 32 MB at 8192 worlds. Short graph-leapfrog training improves 829k to 842k samples/s (+1.6%).
- All 40 reduced CUDA-graph tests pass; contact-rich 512-robot Anymal/H1/G1 screens remain finite.
- Related experiments rejected: kinematics/factor initialization fusion lost 0.6-1.6%; lane-0 row-velocity/solve fusion lost 23%; a FeatherPGS-inspired 64-thread contact solve lost 7%. Real G1 training does not launch the response transpose, and relax must retain global row caches, so a large row-build/solve megakernel has no credible traffic-elimination case.

### Warp-parallel reduced momentum capture (2026-07-03)
- The momentum-conserving split captures every articulation's pre-Coriolis world momentum before the second ABA pass. The original CUDA kernel assigned one thread per tree and serially traversed every link; the retained kernel assigns one deterministic 32-lane tile per tree and reduces mass, system COM, linear momentum, and angular momentum in a fixed tree reduction. CPU keeps the serial path.
- Post-commit nsys measures capture at 69.6 to 13.4 us per launch (5.2x, 1.9% to 0.4% of GPU time). Final matched G1 physics bracket: 1.528M to 1.543M steps/s (+1.0%). Graph-leapfrog training improves 813.5k to 821.9-826.5k samples/s (+1.0-1.6%). Contact-rich 512-robot fleets improve about 5% on Anymal/H1 and 3% on G1; many articulations in one world also pass.
- Physical validation is load-bearing because the parallel sum changes FP rounding order: all 40 reduced CUDA-graph tests pass, including strict long-horizon energy/momentum, internal loops, self-contact, dense contacts, hybrid mode, and Featherstone comparisons.
- A fresh from-scratch policy passes the full frozen gate at only 91.75M samples (iteration 175), before fine tuning. Seed 1000/2000: zero falls, battery tracking 0.9057/0.9035, jerk 0.1434, roll/pitch rate 0.2038, yaw rate 0.1172, leg speed 0.9728.
- Rejected alternatives: removing the post-contact Coriolis ABA pass gained 6% physics but failed six momentum/loop tests (up to 5.6% long-horizon drift); compile-time full/velocity publish specialization was exact but neutral (+0.1%).

### Topology-selected packed reduced-contact gather (2026-07-03)
- NCU showed the gather at 67% memory SOL, 128 registers, 33% occupancy, and roughly 57% sector-utilization headroom. The old launch dedicated one 32-lane warp to each articulation even when only a few contact points were live.
- A specialized 8-lane mapping packs four independent articulations per warp. It preserves every contact, all 32 points per page, arbitrary column counts, and the existing multi-page graph loop. Contact math remains identical.
- The packed path is selected once at construction only when articulation_count is at least 8 times the device SM count. Smaller fleets keep the original kernel byte-for-byte; there is no user tuning knob.
- A final 300-replay 8192-world bracket improves 1.417M to 1.512M steps/s (+6.7%); an earlier cooler bracket measured 1.625M to 1.717M (+5.6%). Steady graph-leapfrog training improves 795.6k to 803.7-805.6k samples/s (+1.0-1.3%).
- The complete 40-test reduced CUDA-graph suite passes. The arbitrary-contact-count test explicitly forces the packed kernel through more than two 32-point pages, zero-contact replay, and contact restoration in the same captured graph. Anymal/H1/G1 512-world screens stay on the unchanged serial path.

### Resident generalized-contact rows (2026-07-03) - REJECTED
- A graph-static `resident32` path cached the first 32 Jacobian and articulated
  response rows in shared memory. Per-articulation masks routed sparse blocks
  through the resident kernel and denser blocks through the reference kernel,
  avoiding batch-global fallback and conditional launches.
- A CUDA-graph regression with several worlds and multiple articulations per
  world exercised the resident kernel and produced bit-identical generalized
  coordinates, body poses, and velocities relative to the reference path.
- Four steady G1 graph-leapfrog intervals measured **801.7k samples/s versus
  877.1k/s reference (-8.6%)**. The extra disjoint launch, fixed shared loads,
  and reduced occupancy cost more than repeated reads of already cached global
  rows.
- The prototype is preserved at commit `9d810d4d` on branch
  `twidmer/experiment-phoenx-resident-contact-rows`. Do not retry a standalone
  row cache; revisit only as part of a fused pipeline that removes launches and
  global row materialization altogether.

### Reduced factor/contact multi-stream overlap (2026-07-03) - REJECTED
- Full begin_substep overlap is incorrect: ABA advance publishes link velocities read by contact warm-start tangent construction. A race-free prototype overlapped only kinematics/factorization with ingest and joined before advance.
- It captured and replayed correctly inside the leapfrog trainer, but lost twice: 778.7k vs 800.9k samples/s, then 774.7k vs 796.1k in reversed order (about -2.7%). Concurrent factorization contends with the already-overlapped learner more than it hides setup. The implementation was removed completely.

### ABA generalized-acceleration publication removal (2026-07-03) - REJECTED
- Both advance kernels publish per-DOF acceleration to the shared inverse-mass response buffer even though normal advance/publication does not consume it. Removing the argument and stores was exact but neutral/slightly slower: matched 500-replay G1 physics was 1.443M candidate vs 1.445M baseline. The compiler already hides or benefits from these stores; fully removed.

### Path-sparse packed contact Jacobian (2026-07-03) - REJECTED
- Prototype stored only source-path Jacobian values plus DOF indices, removed the dense zero fill, and used dynamic `tile_extract` reads from shared generalized velocity with three manual warp reductions per contact point.
- Exact equations and graph capture worked, and memory use fell, but G1 physics regressed from 1.593M to 1.569M steps/s (-1.5%). Indexed shared reads and repeated warp reductions moved more latency into every PGS iteration than the builder saved. Fully removed.
- Do not retry without a materially cheaper indexed tile-dot/gather primitive. A future sparse path must preserve the current dense tile solve or amortize indexing across many more operations.

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
- **Resolved 2026-07-10: implemented and REJECTED** — see "Substep-0
  articulation/ingest multi-stream overlap" under *Tried and reverted*. The
  design's ~10-15% estimate ignored memory-bandwidth contention between the
  overlapped branches; measured end-to-end win was +1.6%. Full design details
  removed (see git history of this file if ever needed).

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

### Recipe iteration reduction (2026-07-15) - REJECTED; recipe is near-minimal
- G1 recipe is 3 substeps x 2 PGS x 1 velocity(relax) iter. Tested reducing each
  through the frozen gate (seed 42):
  - solver_iterations 2->1 (vel=1): fails the iter-75 screen (0.675 < 0.68) — the
    PGS iterations are load-bearing.
  - velocity_iterations 1->0: **+6.3% clean G1 training throughput** (1.019M vs
    958k, ABAB) and holds the iter-75 screen (0.706, seed 11 even improves to
    0.701), BUT the known-passing seed 42 then **fails the iteration-150 screen
    (0.767 < 0.80** required; baseline 0.844). The bias-off relax pass provides
    real long-horizon drift removal that the early screen doesn't reveal.
  Conclusion: the recipe is near-minimal for quality; naive iteration reduction
  can't hold the full frozen gate. (SOR over-relaxation to accelerate a reduced
  iteration was not pursued — parameter tuning, not the target.) Lesson: an
  iter-75 screen pass is necessary but NOT sufficient; the iter-150/0.80 screen is
  where relax-removal is caught. This maps the recipe's quality floor precisely.

### Factor + kinematics refresh stride across substeps (2026-07-15) - REJECTED
- Amortize the reduced ABA articulated-inertia factor (12.6% GPU) + local
  kinematics (8.4%) across a control step's substeps, like prepare_refresh_stride
  does for contacts. Freeze only config-dependent quantities (factor, motion
  subspaces, anchors) on continuation substeps; keep forces/velocities/biases live.
- stride=1 byte-identical; contact-free floating-tree momentum/energy essentially
  unchanged at stride 2/3 (max|dq|~4e-5). BUT: isolated G1 physics (8192, contact,
  substeps=3) stride=2 +1.4% (noise), **stride=3 NON-FINITE (NaN diverges)**.
- Root cause: the reduced CONTACT-block solve builds its generalized response rows
  directly from the factor (joint_d_inv/joint_u/joint_s). A stale factor makes the
  contact response geometrically inconsistent with the current pose → PGS contact
  iteration blows up. Only safe on contact-free articulations; G1 training is
  contact-laden. The predicted +10-14% never materialized (stride-2 noise, stride-3
  diverges). Reverted. (A "freeze only when a world has no active reduced contacts"
  variant is possible but useless for contact-laden G1.)

### Build-time patch-reduced contact rows (2026-07-11/15) - DEFINITIVELY REJECTED
- Third and most thorough attempt (after the 2026-07 separate-phase try that lost
  -7.4%). Emit P normal + 2 coupled tangent rows per eligible convex contact
  column (P+2 instead of 3P) to shrink the bandwidth-bound row stream.
- M1 (foundation, side-data): eligibility + descriptor, byte-identical, confirmed
  **G1 is 100% patch-eligible** (foot-box vs ground plane) with **33% projected row
  reduction** on the standing snapshot. Committed then reverted.
- M2 (in-loop, per-articulation dual-dispatch): correct P+2C solve (box-on-plane
  stable, momentum/energy pass, deviation <5e-3) but gated per-ARTICULATION
  single-page → never activated on G1 (33 pts / 2 pages / 16 cols); ON paid pure
  overhead -5.3% physics / -6.7% training from the dual-dispatch solve kernel's
  register/occupancy bloat (both branches compiled in).
- M3 (pure-patch fleet path): removed every M2 confound — fleet-level all-eligible
  flag, per-COLUMN page-aware layout, pure single-purpose patch kernels (no classic
  branch, no per-articulation decide), patch-aware transpose. Patch ACTIVATED on G1
  (30.3% row reduction realized), default OFF byte-identical, correct. Still
  **-35.3% isolated G1 physics** (1.401M vs 2.167M steps/s). Precomputing the
  response diagonal changed nothing.
- **Definitive root cause:** the row reduction is real and physically correct, but
  realizing it requires a serial per-articulation emit pass + a DIVERGENT
  column-iterating 2x2 coupled solve with data-dependent control flow. That maps
  terribly to the warp-cooperative uniform-tile model the classic per-point solve
  exploits. **Row-COUNT is the wrong lever for the reduced contact solve: the classic
  uniform per-point tight-loop is structurally optimal on this hardware; fewer rows
  don't help when the rows were cheap uniform tiles and the reduction demands
  per-column branching.** Do not revisit patch/row-count reduction for the reduced
  solver. (Contrast: fp16 byte-reduction on the SAME uniform rows works — +9.4%
  phase — because it keeps the uniform tile structure.)

### Wide vectorized loads for per-cid body fields (2026-07-11) - REJECTED standalone
- Warp scalarizes all vector loads; `PHOENX_BODY_WIDE_LOADS` reroutes the
  hot per-cid body reads (velocity+angular pair as one 24 B interleaved slot,
  orientation as v4.f32, inverse_inertia_world sym6 as 3x v2.f32) through
  aliased u64 views + `wp.func_native`. Bit-identical (verified), PTX shows
  48x ld.global.v2.f32 + 6x v4.f32 in the fast-tail prepare+iterate.
- Isolated physics is a real but small win: dr_legs @4096 +1.4%, h1 @4096
  +0.9% (`all`; `vw` alone +0.6% both), consistent across repeats. But the
  **G1 graph_leapfrog training bracket is neutral** (958.5k baseline vs
  957.8k all, ABAB, -0.07%): G1 rollout is the critical path and body-field
  loads are a small share there, especially after fp16x2 cut the dominant
  contact cost. Wide loads only cut request COUNT at equal bytes (weaker than
  the vec6 byte-shrink that bought +22%), and only help joint-dense scenes.
- Rejected as a standalone (not worth a permanent BodyContainer layout change
  for a sub-noise training delta). The `func_native` wide-load helpers are
  retained default-OFF because the color-ordered packed constraint stream
  reuses them on a *contiguous* stream, where the wide-load payoff should be
  larger (locality fixed, not just request width).
- **Follow-up (same day): ABA advance+factor wide loads also REJECTED.** The
  reduced spatial inertia (21-float sym mat66) padded to 22 floats (8B-aligned)
  and read as 11x v2.f32 in the factor/advance warp kernels — the biggest
  training GPU phase (~15%). Bit-identical, ZERO register regression / no
  spill (the occupancy risk did not materialize; factor stayed 144 regs, PTX
  gained 40x ld.global.v2.f32), but G1 training -0.22% (both ON runs below
  both OFF). These kernels are latency-hidden / L2-resident under the
  concurrent learner, NOT request-count-bound, and the +4B/body padding
  offsets any saving. Reverted (not left in tree).
- **Conclusion after three wide-load experiments (reduced rows, contact-iterate
  body fields, ABA): the per-thread request-WIDTH axis is exhausted on the
  reduced path** — all neutral-to-negative on G1 training. Cutting request
  count at equal-or-greater bytes is a weak lever when loads are latency-hidden
  or already sector-efficient. The remaining bandwidth levers are COALESCING
  (access pattern — the NCU 5.1/32-bytes-per-sector problem, biggest and
  untested at the layout-redesign level), byte reduction (fp16, shipped for
  rows), and row-count reduction (patch rows).

### Reduced two-substep cold start (2026-07-10) - REJECTED
- G1 recipe at sim_substeps=2 from scratch: paired iteration-75 screens
  0.657 / 0.698 / 0.625 (vs 3x2's 0.708 / 0.659 / 0.674); the one screen
  survivor (seed 11) missed the frozen 0.80 second screen at 0.785 with zero
  falls. 0/3 full-path survivors vs 1/3 at 3x2 — expected cost cannot
  improve. The retained 3->2 substep transition at 99.6M samples stands.
- Related (same day): projected-maximal at 3 substeps has solved anchors via
  the position-level tree projection (see MAXIMAL_JOINT_TREE_PROJECTOR.md)
  but loses the training race decisively (0.255/0.390/0.403 vs
  0.708/0.659/0.674); extra PGS iterations at 3 substeps are provably useless
  (3x4/3x6/3x8 anchor RMS flat at 5.3-5.6e-2 pre-pass). Maximal-mode learning
  failure is now narrowed to drive/actuation dynamics — contact convergence
  and positional closure are both ruled out.

### Substep-0 articulation/ingest multi-stream overlap (2026-07-10) - REJECTED
- Implemented the designed side-stream fork (begin_substep split into factor /
  advance halves, advance event-ordered after the contact warm-start gather —
  the audit found that gather is the only pre-substep reader of articulated
  ``bodies.velocity``), plus a ``begin_step_async`` variant hiding state
  import + eval_fk + factor under ``model.collide``. Deterministic and
  bit-identical to the serial path.
- G1 shared-physics 8192-world brackets: in-step fork **-1.2%**, pre-collide
  async **+1.6%**. Node-level nsys (``--cuda-graph-trace=node``; the default
  graph granularity only shows eager warmup steps) confirms the captured graph
  truly parallelizes the branches.
- Both branches are memory-bound, so concurrency pays half its theoretical
  ~13%: under overlap eval_fk slows 173 -> 320 us, local kinematics 44 -> 100,
  factor 128 -> 160. Contact row build (884 us) + tile solve (~810 us) still
  serialize after the join — ~61% of the 3.8 ms standing-pose step is
  untouched by any front-section overlap.
- Rejected and fully reverted: +1.6% does not justify a side stream, four
  events, an async API contract (no state/control mutation between the calls),
  and per-env wiring. Streams stay where they obviously pay (leapfrog
  rollout/update/copy). Do not re-try front-section overlap; the lever on this
  pipeline is making the row build / contact solve cheaper, not scheduling.

### Upper-triangle-only reduced factor correction (2026-07-03) - REJECTED
- Reduced inertia is stored as 21 symmetric upper-triangle values, so two exact
  prototypes stopped computing the discarded lower triangle. The stronger
  version also kept the local result packed throughout, removing the final
  repack and a duplicate 36-float local matrix.
- The factor kernel improved only 129.04 to 128.27 us (-0.6%), while a reversed
  300-replay G1 bracket was neutral (1.633M candidate versus 1.631M baseline;
  the simpler form had regressed about 0.4%). Factorization remains dominated
  by inertia traversal and memory latency, so both forms were fully reverted.

### Conditional empty-reset FK/observation (2026-07-03) - REJECTED
- G1 observation produced an exact device-side `any_done` reduction, and a CUDA
  graph conditional skipped masked FK plus the second observation when no world
  terminated. Reset state clearing, RNG advancement, and command sampling stayed
  unconditional, so the experiment preserved deterministic reset semantics.
- The same captured graph adapted correctly between empty and nonempty reset
  sets. Existing FK masking and periodic-command graph tests passed.
- At 8192 worlds the random-policy graph-leapfrog benchmark was neutral:
  1.15385M samples/s candidate versus 1.15325M baseline, because at least one
  world terminated on almost every step. Isolated high-reset stepping regressed
  about 1.7% (2.724M versus 2.771M samples/s). A low-fall benefit remained
  plausible but unproven, so the conditional and its extra reduction were fully
  reverted rather than retained speculatively.

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
  layout** (`bench_g1_response_basis_aba`, 2026-07-03): compressed
  prepare+solve 495 us vs direct 563 us = **1.14x**, with response error up to
  2.0e-6. This is an optimistic hard-coded G1 case with exactly two source
  bodies; a general path needs active-body deduplication, variable-width basis
  dispatch, and fallback for arbitrary contact diversity. The projected gain
  is too small to justify a scene-specific production specialization.
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

## Register-forwarded ABA parent state (2026-07-03)

- The one-launch Nsight Compute report for reduced ABA advance measured 80% of
  issue delay as long-scoreboard waits on global/L1TEX operations, only 13.03%
  of cycles with an eligible warp, 22.3% achieved occupancy, 86.1% L2 hit rate,
  no spills, and only 17.5% compute throughput. The 8,192-articulation grid was
  just 0.30 full resident waves on the large Blackwell GPU.
- Widening G1 advance tiles from 8 to 16 lanes to expose more warps was exact
  but lost 2.4% physics throughput (1.546M versus 1.584M steps/s) because idle
  lane work outweighed latency hiding; it was fully reverted.
- The retained path precomputes each joint parent lane within the previous tree
  depth. All subgroup lanes execute a small 6-float native shuffle, forwarding
  parent velocity and Coriolis registers directly between depths while keeping
  the original global outputs for later phases. This removes two dependent
  parent-state loads per non-root joint without changing arithmetic or order.
- Bounded Nsight Systems comparison: advance median 127.2 to 117.0 us (-8.1%);
  mean 134.3 to 116.8 us (-13.0%). Short graph-leapfrog training measures
  875.9k versus 873k samples/s (+0.3%). All 40 reduced CUDA-graph tests pass,
  including serial parity across 8/16/32-lane topologies; Anymal/H1/G1 contact
  fleets remain finite.
- A refreshed concurrent-training trace still ranks advance at 9.1%, packed
  row build 7.1%, contact solve 5.7%, publish 5.4%, factor 4.7%, and narrow
  phase 2.6%; overlap hides much of the isolated advance gain. Extending the
  shuffles to final acceleration/twist improved isolated median another 2.5%
  but reduced training to 869k/s, so that extension was fully reverted.

## Real G1 training trace after factor compaction (2026-07-03)

- A delayed five-second Nsight Systems window captured steady graph-leapfrog
  training in ``/tmp/phoenx_g1_train_after_dinv.nsys-rep`` without recording
  initialization. The training-weighted ranking is: learner dense forward
  10.8%, reduced ABA advance 8.9%, packed contact-row build 7.2%, generalized
  contact solve 5.8%, reduced publish 5.3%, factor 4.7%, and narrow phase 2.6%.
- ABA advance is therefore the largest in-scope solver kernel. A prepared
  one-launch Nsight Compute run
  (``analysis_tools/ncu_profile_reduced_advance.sh``) is required after the
  compact D-inverse layout; the old counter report is unavailable and stale.
- The warp advance still writes ``generalized_rhs`` even though its fused local
  recurrence does not consume that output after the kernel. Treat removal as a
  measured candidate only: eliminating the analogous generalized-acceleration
  store was previously neutral, so counters should decide first.

## Reduced inverse-factor DOF-row layout (2026-07-03)

- Replaced the fixed ``[joint_count, 6, 6]`` inverse-factor allocation with
  ``[dof_count, 6]``. Row ``r`` of a joint now lives at its global DOF index;
  the six columns retain the identical dense per-joint values and arithmetic
  order. Fixed joints allocate no rows, 1-DOF joints consume six floats rather
  than 36, and 6-DOF joints retain all 36 entries.
- G1 at 8,192 worlds uses about 47 MiB less GPU memory. The isolated short
  graph-leapfrog training benchmark improves 862k to 873k samples/s (+1.3%).
  The contact-rich physics benchmark is approximately neutral within run noise
  (1.584M measured versus the preceding 1.593M reference).
- All 40 reduced CUDA-graph tests pass, including ABA/mass-matrix parity,
  6-DOF roots, tiled loops, Featherstone comparisons, contacts, deterministic
  momentum/energy, and live mass updates. Contact-rich 512-robot Anymal/H1/G1
  screens remain finite.
- A full fixed 96-row response slab in shared memory was also tested and fully
  reverted: 1.375M versus about 1.593M physics steps/s (-14%). The footprint
  reduced occupancy and loaded unused rows; the existing repeated row loads
  are already served effectively by cache.

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
