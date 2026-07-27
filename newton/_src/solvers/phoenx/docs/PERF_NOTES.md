# PhoenX Performance Notes

This file records current performance conclusions. Keep only reproducible wins,
constraints that affect future work, and rejected ideas likely to be retried.
Detailed experiment histories belong in benchmark output, not here.

## Measurement rules

- Compare identical scenes, contact policy, substeps, iterations, capture mode,
  and settled workload. Report contact counts when PGS ordering can change the
  manifold.
- Separate kernel, captured-frame, physics-step, learner-update, and
  wall-to-quality results.
- Reverse A/B order, warm up, and repeat. Small single-session differences are
  often thermal noise.
- Treat modeled bandwidth as a lower bound unless hardware counters were used.
- Preserve deterministic graph replay and run analytical/trajectory tests after
  arithmetic or scheduling changes.

## Current architecture conclusions

- Production G1 is latency- and dependency-bound. Sparse articulation and
  contact traversals dominate; tensor throughput and peak DRAM bandwidth do not.
- Recent production counters show only 4--10 useful bytes per 32-byte sector,
  but dependent L1TEX waits account for 53--75% of issue gaps and eligible warps
  remain 0.17--0.57 per scheduler. Wider storage alone is not a solution.
- Tree depth and sparse parent/child reuse limit reduced-coordinate kernels.
  More occupancy, fewer launches, or fewer bytes can regress by increasing live
  state, code size, or dependency latency.
- Optimizations must be evaluated end to end. Graph leapfrogging can hide learner
  gains behind rollout, and faster isolated kernels may not improve training.

## Key decisions

| Area | Change | Evidence / constraint |
| --- | --- | --- |
| Launches | Select lean Warp launch bounds only for kernels proven safe | Avoid blanket launch changes. |
| Contacts | FP16x2 immutable packed contact rows | Removed in `b374c0c4e`; the roughly 1% training gain did not justify the extra path. Production rows are FP32. |
| Contacts | Register-resident generalized contact delta | Removes unnecessary global staging. |
| G1 RL | Mask observation work after reset | Safe because reset worlds discard the omitted values. |
| Collision | Statically omit provably empty GJK/MPR stages | Topology-selected; no runtime branch. |
| Collision | Compact primitive-contact sort keys | Lossless key/tiebreak contract. |
| Inertia | Symmetric-packed body and reduced inertia | Preserve exact unpacking and alignment. |
| Reduced ABA | Remove unused inertia stores; parallel momentum capture | Retained by full physics tests. |
| Reduced contacts | Topology-selected packed gather and recomputed path Jacobian | Better than resident/path-sparse row storage. |
| Reduced contacts | Warp-shuffle scalar broadcasts in one-warp contact solves | Removes shared-tile barriers without changing arithmetic. Point-contact G1 improved about 2.82% end to end. |
| Coloring | Deterministic per-world greedy/direct endpoint ownership | Large gain for many small worlds; retain fallback above 64 colors. |
| Scheduling | Adaptive threads per world and block-world scheduler | Select by workload; no universal scheduler wins. |
| Solver | Fast-tail specializations, zero velocity-iteration support, per-substep inertia refresh | Required correctness/performance behavior. |
| Contact preparation | Defer tangent effective masses after sticky rejection | Avoid work for rejected rows. |
| Soft bodies | Drop the unused fourth tet vertex from contact coloring | Keep eight-endpoint capacity for other interactions. |
| Coloring | `wp.capture_while` greedy MIS loop | Default for the applicable single-world path. |
| Reduced factor | DOF-type split | About +2.82% physics in the measured production case. |
| SDF | Reuse accepted edge endpoints | Retained after parity tests. |
| RL rollout | FP32 cuBLAS for large dense contractions | 8192x91x128: 0.0287 -> 0.0082 ms; 8192x256x384: 0.3467 -> 0.0402 ms; about +1.82% matched training throughput. |
| RL optimizer | cuBLAS Muon Gram contractions | Bounded update trace 43.9 -> 27.6 ms; same-session update graphs 98.18 -> 95.30 ms. |
| RL recurrence | Four-step sparse FP32 checkpoints with register recomputation | MinGRU forward 53.2 -> 43.9 us; backward neutral at 110.9 -> 111.2 us; randomized reset-boundary outputs and gradients are bitwise exact. |
| RL activations | BF16 MinGRU projection slabs with FP32 recurrence/accumulation | Production storage 151.0 -> 75.5 MB; isolated forward/backward 0.776 -> 0.730 ms; full A/B/B/A 1.887M -> 1.902M samples/s. Seed 42 passed the 131.072M-sample frozen gate. |
| Diagnostics | Fixed-order rollout reductions | Deterministic and much faster than contended atomics at production scale. |

## Representative production profiles

Before cooperative patch traversal, a refreshed 8,192-world, three-substep G1 trace attributed approximately:

- 13--14% to fused reduced advance/publication;
- 9--10% to packed contact-row construction;
- 9--10% to the two patch solves;
- 5--6% to external advance;
- about 10% to MinGRU forward/backward.

FP32 SGEMM was about 1.3% of GPU time after the retained cuBLAS rollout change. Later patch-traversal profiles are recorded below; percentages are snapshots, not a permanent ranking.
The next broad target is reuse/locality in reduced articulation and contact
traversals, not another dense-math rewrite.

## Rejected or low-value directions

Do not retry without new evidence or a materially different design.

| Direction | Result |
| --- | --- |
| Resident generalized-contact rows | Extra storage/traffic outweighed reuse. |
| Reduced factor/contact stream overlap | Dependencies and capture scheduling prevented a useful gain. |
| Remove ABA generalized-acceleration publication | No end-to-end win. |
| Path-sparse packed contact Jacobian | Irregular traffic lost to recomputation. |
| Tiered contact sort | Brittle and not robustly faster. |
| Depth-major/level-synchronous ABA | Refuted the occupancy hypothesis; dependency latency remained. |
| Reduced block GS with Hessian cross-terms | Fixable, but low expected value for G1. |
| APGD/Nesterov colored GS | Structural convergence mismatch. |
| Fewer recipe iterations | Current recipe is near the acceptable quality floor. |
| Reuse factor/kinematics across substeps | Physics-changing and failed qualification. |
| Build-time patch-reduced rows | Extra staging and traffic lost decisively. |
| Standalone wide body-field loads | More bytes/instructions without useful latency reduction. |
| Two-substep cold start | Quality/performance tradeoff rejected. |
| Substep-0 articulation/ingest stream overlap | No useful overlap after dependencies. |
| Upper-triangle factor correction | Bookkeeping outweighed saved arithmetic. |
| Conditional reset FK/observation | Branching and launch complexity lost. |
| Coarser preparation refresh | Physics-changing. |
| Higher occupancy by register caps | Spills or recomputation worsened latency. |
| Further lossy inertia packing | Precision/range risk without a measured gain. |
| Contact-major/AoS body layouts | Larger footprint or conversion passes lost. |
| Cooperative grid-sync iterate megakernel | Direct CUDA cooperative launch was bitwise correct but 3.0--3.3x slower: captured launches took 12.65 us for 9 phases and 112.65 us for 90, versus 38.54 and 371.27 us with hardware grid barriers. |
| Island sleeping on the Kapla tower | Slower *and* destabilising. Matched 240-frame runs: 116.23 FPS off, 110.10 at threshold 0.15, 109.94 at 0.3. Quality collapses -- drift 0.0655 -> 0.2582 m, max speed 0.229 -> 4.632 m/s, max angular 17.7 -> 76.0 rad/s, tower top 2.788 -> 2.502 m, contacts 256,619 -> 206,297. The sleeping pipeline's per-step cost is never repaid because the 7,100-body island does not quiesce. Thresholds 0.15 and 0.3 give bit-identical quality, which points at the sleep/wake logic rather than the cutoff; root-cause that before any dormancy work. `--sleeping-threshold` on `bench_phoenx_kapla`. |
| Single-block resident subdomain block GS | Morton-partition census of the settled tower (73,572 live contact columns) measured 55.3% interior contacts at `P = 188`, below the 60% gate; `P = 94` reaches 64.6% but idles half the SMs. Shared memory was never the limit (3.1 KB/part). Surface-to-volume rejects one block per part. `analysis_tools/kapla_subdomain_feasibility.py`. |
| Eight-lane cooperative contact manifold | Bitwise-exact scalar and cooperative probes were neutral at 4.095 and 4.099 us; allocating lanes to cached row streams reduces independent columns without helping scattered endpoints. |
| Cross-item Kapla prefetch by shrinking the grid | Settled regular colors have 4,294--5,529 rows and overflow has 32,247, all below the 48,128-thread production grid. Creating two-row regular-color work would use fewer than 87 of 188 SMs; the prior 4-blocks/SM A/B also lost. |
| Separate regular-color launch grid | Reducing regular colors from 48,128 to 11,360 threads measured 82.40 FPS versus fresh 82.76--82.78 FPS baselines. Lower launch waste did not repay lost latency hiding; removed. |
| Indexed AoSoA8 contact rows | A no-padding point-major layout reduced a synthetic 24-plane gather from 10.86 to 7.62 us, but the required dependent row-index load reduced full Kapla throughput to 78.17 FPS versus 82.76--82.78. Removed. |
| Per-substep packed endpoint mass/inertia | Corrected staging improved matched Kapla throughput only about 1.9% (82.76--82.78 to 84.02--84.62 FPS), required 14 planes and a new launch, and changed the floating-point trajectory; removed. Step-level staging is incorrect because world inertia refreshes every substep. |
| One-block-per-world all-substep megakernel | Register pressure and limited parallelism. |
| Multiple fused inner sweeps | Live-state cost exceeded launch savings. |
| Single-world multi-sweep iterate | Same register/dependency problem. |
| Small algebra/function-extraction rewrites | Neutral or negative generated code. |
| Lean greedy fixed-iteration coloring | Incorrect or slower on difficult graphs. |
| Patch-row cooperative traversal/layout rewrites | Sparse dependent access remained dominant. |
| Warp `array_noalias` annotations | Generalized solve improved about 1.8%, but contact rows were neutral, packed gather regressed about 1.3%, and settled G1 physics improved only about 1%. Do not depend on the experimental Warp branch. |
| Forced vec4 response layouts | Wide PTX loads/stores were emitted, but the best isolated vector-load gain was below 1% and every tested PhoenX response layout regressed end to end. |
| Always use coalesced packed-response transpose | Production throughput fell about 6.6%. |
| Tensor-core rollout crossover at 8,192 rows | Faster iterations but about 11% worse wall-to-quality; removed. |
| Active-action-only PPO likelihood | Changed the objective and failed training evidence. |
| Publication traversal fusion | Enlarged live state; fused advance/publish regressed about 2%. |
| Skip advance outputs overwritten by publication | Suppressing both outputs averaged only +0.14% with a divergent branch; suppressing only dead internal twists regressed about 2%. Removed. |
| Pack scalar joint work into inverse-factor rows | Same 28-byte footprint but only +0.08% in the production graph; retained SoA layout. |
| Depth-local reduced scalar/factor repacking | Scalar depth order was neutral; global inertia-component SoA was 75% slower; compact depth AoSoA and split joint-u/d storage were neutral. |
| Depth-major reduced-factor workspace | Bitwise-correct, but matched G1 patch throughput improved only 0.35%; removed. |
| Reuse checkpoint MinGRU kernel for read-only sequences | Only a 1.1% forward-subphase change and no measurable full-training gain. |
| Fuse next-layer BF16 shadows into MinGRU recurrence | Isolated recurrent graph improved 0.730 -> 0.713 ms, but the extra dependent store reduced full A/B/B/A throughput 1.940M -> 1.927M samples/s; removed. |

## Open ideas

- Sub-island contact dormancy. The settled tower leaves 63.9% of live contact
  columns with both endpoints at rest (v<0.01, w<0.05) and 86.5% at
  (0.02, 0.1) -- against an iterate kernel that is 52.3% of GPU time. The
  shipped `sleeping_velocity_threshold` path cannot reach it: the contact graph
  is 10 islands with one 7,100-body giant (62.6% of bodies) that a few spinning
  bricks keep permanently awake, capping island sleeping at 28.2% of bodies at
  threshold 0.15 and 31.2% at 0.3. Granularity is the defect, not the idea --
  a contact is skippable iff *both* endpoints are dormant, which is per-body
  state, and dormant endpoints behave as infinite mass (what copy-state slots
  already express). `analysis_tools/kapla_census.py --mode dormancy`.
- Cluster-scale resident subdomain block GS. The census that rejected one-block
  subdomains measures 83.8% interior at `P = 12` and 78.7% at `P = 24` -- part
  sizes reachable only with Blackwell clusters (16 blocks sharing distributed
  shared memory, 48 KB per cluster), which keep all 188 SMs busy. Blocked on a
  Warp cluster launch path (`cudaLaunchKernelEx` cluster dims plus a sync
  intrinsic). Changes GS ordering; gate on invariants and quality.
- Depth-ordered joint descriptor, attempted 2026-07-27 and reverted unfinished.
  Packing `(joint, parent_lane, child, dof_start | dof_count << 24)` into one
  `vec4i` indexed by the unit-stride depth slot replaces a dependent
  `articulation_depth_joint` fetch plus four scattered per-joint gathers, and
  hoists `parent_lane` -- which gates both `_shuffle_reduced_spatial` calls --
  to the first dependent level. Plumbing is wider than it looks: the device
  func has four kernel wrappers, including
  `_make_biased_contact_advance_publish_kernel`, whose argument order differs,
  plus consumers in `reduced_contact_block.py`. The attempt left 18 failures
  and 4 errors in `tests/test_reduced_articulation.py` (baseline: 1 error), so
  the nsys timings taken against it are void -- four modules failed to compile,
  so the measurement was not an A/B. Redo by converting one wrapper at a time
  and running the suite after each.
- Test a depth-ordered reduced-joint descriptor that coalesces invariant child,
  DOF, type, parent-lane, and child-range metadata only after auditing which
  fields are not already derivable or packed. Require at least 5% on fused
  advance/publish and bitwise trajectory parity.
- Improve reuse of topology/factor data within dependent packed-row traversal.
- Test truly world-interleaved hot joint/body fields only if profiling identifies
  repeated scalar transactions; depth-local packing and vec4 padding already lost.
- Benchmark FP32 cuBLAS backward contractions for non-production FP32 learner
  configurations.
- Benchmark complete Muon Newton--Schulz steps before routing its remaining
  fused GEMMs through cuBLAS.

### Kapla launch floor and work scaling

Measured 2026-07-27, RTX PRO 6000 Blackwell, captured-graph replay at the
production single-world geometry (1,504 blocks x 32 threads):

- An empty kernel of identical geometry costs **1.14 us/launch** (90 launches)
  to 1.37 us (9 launches). Against the 5.25 us production iterate that is a
  ~22% dispatch floor, so fusing launches is capped near 22% and is not the
  dominant lever.
- A synthetic three-level pointer chase (slot -> header -> two body states)
  over the same grid costs 1.39 us at 500 live items, 1.48 us at 5,000, and
  2.67 us at 32,000. A regular color (~5k rows) therefore sits ~0.34 us above
  the empty floor.

Do not size the persistent grid from the empty-kernel floor. Re-measured
2026-07-27 at matched settings and bit-identical physics: 8 blocks/SM (the
production 1,504) gives 115.84 FPS, 6 gives 116.02 (noise), 4 gives 113.60
(-1.9%). The floor model predicts a gain from shrinking; it is wrong because it
ignores the 32,222-row overflow partition, which at 4 blocks/SM reaches 1.34
items/thread and serialises for more than the eight ~5k colors save. Partition
sizes on the settled tower are [5513, 5338, 5203, 5075, 4925, 4775, 4553, 4305,
32222] -- overflow alone is 44.8% of rows and, by the scaling above, roughly
37% of iterate time in one launch of nine.

The production iterate is 5.25 us, i.e. ~3.8 us above floor at comparable item
counts. The address chase does not explain it -- the per-column solve does
(1-5 point manifold, sequential GS over points, friction cone). Two
consequences: do not model this kernel as address-latency-starved, and expect
work removal to pay mainly on the large overflow partition (~32k rows), where
item count is well above the floor knee, rather than on the ~5k regular colors.

## Correctness traps

- CUDA-graph state ping-pong requires either an even captured substep count or an
  explicit copy-back. Odd swaps can leave consumers reading stale state.
- Coloring changes alter floating-point PGS order. Compare invariants and quality,
  not bitwise trajectories, unless order is intentionally fixed.
- Never claim a throughput win from runs with materially different contacts.
- Do not infer a bandwidth bottleneck from sector utilization alone.

## Profiling

```bash
uv run python -m newton._src.solvers.phoenx.benchmarks.profile_phoenx
```

Use `nsys` for phase attribution and privileged `ncu` counters for dependency,
cache, and sector analysis. Record hardware, clocks, Warp/CUDA versions, command,
scene, contacts, and capture mode with every retained result.

## Detailed evidence and coverage

### Acceptance regimes

| Regime | Throughput case | Required gate |
| --- | --- | --- |
| Reduced robot fleets | G1, H1, and ANYmal at 8,192 worlds | Reduced CUDA-graph tests, finite-state screen, G1 exploration screen |
| Maximal robot fleets | G1 with patch friction | CUDA capture/replay regression |
| Large single world | 11,340-brick Kapla | Drift, speed, finite state, and contact count |
| RL training | G1 graph-leapfrog | Learning screen and wall-to-quality check |

Compact manifold friction rows improved captured reduced physics by 5.5% on
ANYmal, 3.7% on H1, and 1.8% on G1. Extending block-world scheduling to large
maximal patch-friction fleets improved G1 from 3.617M to 4.298M world-steps/s
(+18.8%); small fleets retain fast-tail scheduling.

For Kapla, device-selected 32-pair SAP chunks reduced dense sweep time 290 to
173 us and improved 107.14 to 109.36 FPS with identical diagnostics. Sparse
46k-body towers and 3,600-box grids stay on the legacy sweep. PGS accounts for
about 66% of kernel time; collision is about 8.4%. A contact-refresh stride of
two gained about 3% but changed trajectories and remains rejected. Eight
colored partitions is the measured stability floor for the production tower.
On a settled 11,340-brick snapshot, deterministic Morton partitions retained
75.3%, 67.8%, 58.2%, and 47.7% of contact points as partition-interior for
47, 94, 188, and 376 partitions. This supports a block-resident subdomain
oracle; it does not yet establish convergence or an end-to-end win.
The oracle rejected explicit shared caching at practical occupancy. With 188
subdomains, 61 bodies per subdomain, one 128-thread block per subdomain, and
58.2% interior points, two repeats measured 558.9/558.7 us with global body
state versus 553.1/552.2 us with explicit shared state (about 1.1%). The 42%
cache gain seen for 376 one-warp subdomains only recovered their insufficient
latency hiding; both shared variants converged near 2.5--2.6B simplified
point-iterations/s, close to production's useful iteration rate.

### Predicting training value

Use:

```text
expected training gain ~= isolated phase gain * phase share of training GPU time
```

The original graph-leapfrog trace was rollout-bound: about 0.45 s rollout versus
0.14 s update and 96% union-busy. Historically, FP16x2 rows improved their phase by 9.4%; that phase was about
11% of training GPU time, predicting roughly 1%. Measured G1 training improved
1.06% across three ABAB comparisons. The path was later removed because this
small gain did not justify its maintenance and qualification cost. Optimize phases that are
both large and limited by the proposed resource; do not extrapolate isolated
kernel speedups directly to training.

### Contact architecture laws

- At 8,192 G1 worlds, global response-row streaming beat on-chip
  warp-per-world by 13% excluding gather; the on-chip design wins at 512 worlds
  but loses residency at scale. Matrix-free per-sweep tree solves reached only
  0.36x because each warp walks a dependent joint chain.
- Warp emitted scalar global loads for hot vector types in the inspected PTX.
  Aliased wide loads help random per-row gathers only when fewer requests offset
  unpacking and footprint. They were neutral for already sector-efficient
  reduced rows at 8,192 worlds.
- FP16 without packing gained nothing. FP16x2 reduced requests and gained 9.4%;
  FP16x4 measured 8.3%. Request count, not only stored bytes, explained the win.

### Retained solver details

- Greedy single-world coloring selects the smallest free color during MIS
  commits, uses an `int64` forbidden mask, and falls back to JP above 64 colors.
  Fusing compact/scan into CSR construction saved about 1.5% on Kapla. Removing
  remaining-list compaction saved about 16% of step time.
- The parallel `int32` color-tag mirror reduced the hot greedy launch about 5%
  while the `int64` view remains required by JP fallback.
- Tail fusion drains only small trailing colors. Multi-world PGS preserves one
  global visit to every color per iteration; local multi-sweeps changed the
  coupled contact/joint fixed point and remain rejected.
- Revolute-only static specialization removes mode loads/branches from common
  all-revolute worlds but is only a modest standalone win.
- Cable rows use combined implicit-Euler PD coefficients, Nyquist stiffness
  clamping, a full quaternion log map, and no cross-substep bend/twist warm
  start. This is why `velocity_iterations=0` is valid.
- World inverse inertia is refreshed after every substep. Symmetric `vec6`
  storage reduced its hot footprint from 36 to 24 bytes; measured gains were
  +22% for dr_legs@4096 and +13% for H1@4096, neutral on Kapla.
- Auto threads-per-world uses 8, 16, or 32 lanes from topology/workload.
  Block-world scheduling improved representative 512/1024-world robot fleets
  by roughly 1.11--1.25x. Generalizing its selector to joint-heavy,
  contact-light fleets improved dr_legs@4096 from 2.18M to 2.57M env FPS.
- Greedy coloring always emits family offsets. Large mixed fleets consume joint
  and contact ranges directly, avoiding the per-row family branch.
- Contact preparation defers tangent effective masses until after sticky
  rejection. Soft-tet contacts omit the unused fourth tet vertex from coloring,
  but `MAX_BODIES` remains eight for other interaction families.

### Retained reduced-coordinate changes

- Register-forwarded ABA parent acceleration/twist reduced external advance
  94.14 to 85.46 us and fused advance/publication 226.22 to 220.61 us; a
  production bracket improved 1.718M to 1.739M samples/s.
- Removing dead first-pass link-twist stores later reduced fused
  advance/publication 222.0 to 214.6 us and external advance 85.9 to 81.0 us.
  The production leg-action trajectory remained JSON-identical.
- The reduced inverse-factor layout is DOF-row-major. Splitting factor work by
  DOF type retained a measured +2.82% physics gain.
- Cooperative patch-row construction keeps parent responses in registers and
  passes them through subgroup shuffles. Row construction fell 240.1 to
  150.3 us (-37.4%); production physics improved about 6.1% and full PPO about
  4.4%. Random 8/16/32-lane rows and a production G1 graph trajectory matched
  bitwise.
- That audit found a correctness bug for depths wider than 32 joints: a
  62-child tree differed from serial ABA by up to 0.604 rad/s. Reference paths
  now use exact serial advance for wide trees; persistent paths reject them
  until a correct multi-wave implementation exists.
- Lane-local patch application then reduced row-building solve 137.25 to
  126.45 us and cached solve 74.30 to 67.20 us. Production physics improved
  about 1.88% and PPO about 1.47%, with bitwise G1 trajectory parity. Shared or
  fully register-resident 32x36 response slabs each regressed about 10%.
- Specializing fused advance/publication for its topology-selected 8/16/32-lane
  width reduced the matched G1 kernel median 211.84 to 206.75 us (-2.4%).
  Eight interleaved 256-replay pairs improved environment-step throughput by
  1.03%. External advance did not benefit and retains its general kernel.

### Retained learner details

- Large BF16 cuBLAS contractions use FP32 accumulation and graph capture. FP32
  cuBLAS uses an 8,192-row/64-output crossover and retains the Warp fallback.
  Its maximum reported recurrent-projection difference was 4.96e-5.
- Fixed-order rollout diagnostics changed a production-scale reduction from
  about 1,000 to 41 us and removed reporting nondeterminism; these values do not
  feed PPO.
- Reusing the MinGRU backward candidate sigmoid reduced 114.6 to 109.5 us
  (-4.5%) with bitwise randomized-gradient parity.
- cuBLAS Muon Gram products changed ten-step parameters by at most 3.73e-8
  versus Warp. Full graph-leapfrog throughput was neutral because rollout hid
  the faster update; retain it for learner headroom, not a current training
  throughput claim.
- Active-action-only PPO likelihood improved short training statistics but
  failed the frozen-policy screen (`battery_perf=0.650`, collapsed +0.8 m/s
  tracking). Ignored action dimensions had acted as an implicit trust-region
  constraint; any retry needs dimension-normalized trust-region tuning.
- Parallel scan MinGRU recurrences lost at the production 512x64x128 shape:
  log-space scan 0.189 versus 0.047 ms and affine scan 0.174 versus 0.057 ms.
  Horizon 64 already exposes sufficient independent work.

### Profiler evidence

Production NCU measurements showed:

| Kernel | Time | DRAM | Eligible warps/scheduler | Main issue |
| --- | ---: | ---: | ---: | --- |
| Fused advance/publish | 235.20 us | 35.82% | 0.22 | 69.9% L1TEX dependency gaps; 101 registers |
| Packed contact rows | 234.75 us | 26.66% | 0.20 | 76.1% L1TEX dependency gaps |
| Patch build/solve | 175.68 us | 19.63% | 0.49 | 52.9% L1TEX dependency gaps |
| Cached patch solve | 91.46 us | 37.50% | 0.49 | 66.9% L1TEX dependency gaps |
| External advance | 107.74 us | 26.12% | 0.17 | Only 0.30 full-GPU waves |
| Factor | 66.56 us | balanced | 0.57 | Dependency latency despite 62.31% compute/cache |

The first counter wrappers accidentally used stale four/five-substep overrides;
the stall diagnosis is valid, but those timings are diagnostic rather than the
authoritative three-substep baseline.

- A later 8,192-world factor-initialization capture measured 72.3 us, 62.6%
  memory/L2 throughput, 9.4% compute, and 10.27M excessive sectors out of
  12.71M. Its 6x6 spatial inertia was already symmetric-packed to 21 floats.
- A component-parallel 21-float writer improved coalescing but reduced complete
  G1 physics throughput about 0.4%; the consumer-friendly AoS layout remains.
- Packing the source symmetric 3x3 inertia from nine to six floats reduced the
  initializer median 52.4 to 51.5 us, but was neutral end to end and added a
  persistent cache. Both experiments were removed.
- The fused advance/publish capture used 98 registers/thread, reached 22.9%
  occupancy, and spent 66.3% of its issue interval on long-scoreboard stalls.
  Memory/L2 throughput was 50.8% versus 22.9% compute, with 72.81M excessive
  sectors out of 87.66M; scalar accesses used about 4.3 bytes per 32-byte sector.
- Splitting momentum capture reduced the fused median only 210.4 to 208.7 us but
  added an 11.9 us kernel, so it was removed. An exact rigid-body operator using
  mass, COM, and local 3x3 inertia reduced the fused median to 206.3 us, while
  complete G1 physics stayed neutral. Serial-oracle comparisons passed on
  branched 8/16/32-lane trees, but the floating-point reorder had no pipeline
  payoff and was removed.

- Body-index component SoA regressed fused advance to 213.6 us. Depth-packed
  canonical scratch reached 208.8 us but slowed external advance 80.8 to
  85.4 us; shared scratch reached 241.3 us. All were removed. Restricting depth
  packing to the static fused recurrence retained canonical storage elsewhere:
  two Nsight repeats measured 205.2 and 206.2 us versus the 210.35 us baseline,
  external advance stayed near 80.9 us, and complete physics reached 2.04M
  steps/s. Serial-oracle 8/16/32-lane and production G1 analytical tests pass.
- A depth-indexed SoA topology descriptor removed joint-to-metadata dependent
  lookup chains and improved an isolated real-G1 three-pass metadata probe
  17.1%. The production integration passed analytical, warp-versus-serial, and
  fused-momentum tests, but a longer matched 8,192-world run regressed complete
  physics from 7.43M to 7.30M steps/s (1.8%). Its seven extra arrays increased
  cache traffic more than the pointer chasing cost; the integration was removed.

### Configuration and experimental boundaries

- Key knobs live in solver configuration: greedy coloring, tail-fuse threshold,
  threads per world, scheduler selection, solver/velocity iterations, substeps,
  and preparation refresh. Change defaults only with cross-regime qualification.
- Warp-local no-coloring PGS and color-grid actual-solve schedulers remain
  experimental oracles, not production alternatives.
- Multi-stream reduced-pipeline overlap is a design note, not implemented
  production behavior.
- Speculative coloring remains opt-in; deterministic greedy/JP paths are the
  supported defaults.

### Current learner and pipeline chronology

These later results supersede the early FP16/contact-row prioritization:

- Production rollout was about 273 ms per 524,288 samples versus nanoG1's
  270 ms; the initial PhoenX update was 250 ms versus about 94 ms. Enabling the
  configured BF16 MinGRU contractions reduced update to 231 ms and leapfrog
  iteration time 0.423 to 0.408 s (+3.6% samples/s).
- Reusing pre-update values instead of a redundant post-optimizer policy
  forward restored snapshot consistency and reduced update to about 209 ms;
  leapfrog throughput improved 1.284M to 1.364M samples/s (+6.2%).
- Giving rollout higher CUDA stream priority while launching update first
  improved 1.361M to 1.382M samples/s. Rollout-first measured 1.191M and was
  rejected. Save/resume and Anymal repeatability/finite-training gates passed.
- Reusing BF16 MinGRU inputs/weights reduced update 204.1 to 200.4 ms. Removing
  an unused FP32 combined-gradient materialization reduced 203.1 to 197.2 ms
  and improved leapfrog throughput about 1.7%.
- Grouping three compatible 128x384 Muon matrices reduced isolated update
  181.3 to 166.1 ms; grouped and independent updates were bit-identical. The
  end-to-end gain was only about 0.5% because rollout hid update work.
- The optional BF16 `cublasGemmEx` bridge reduced a 32768x128x384 forward,
  weight-gradient, and input-gradient contraction from 0.092/0.166/0.068 ms to
  0.025/0.019/0.017 ms. Isolated G1 update fell 165.6 to 112.6 ms; graph
  throughput improved about 10%. Warming Ant improved about 23%, with a finite
  120-iteration learning run.
- Seed 42 passed the frozen G1 gate at 131.07M samples (`battery_perf=0.9306`,
  zero falls). Seed 11 missed at 100M under both backends; this supports
  numerical safety, not equal multi-seed sample efficiency. Changing gate chunk
  size resets worlds/recurrent state and changes the trajectory.
- A 256-way priority CDF reduced priority sampling 14.2 to 2.5 ms/update and
  isolated update 123.0 to 113.9 ms. End-to-end improvement remained about
  0.5% under leapfrog overlap.
- Deterministic multi-block optimizer norm reductions reduced isolated update
  about 9.5% and the norm phase about 12.8 to 2.5 ms. Reducing log-standard-
  deviation partial chunks from 256 to 16 rows cut its partial kernel 6.27 to
  0.49 ms and brought isolated update to 99.3--99.8 ms. Both were neutral for
  rollout-bound G1 but improved Ant.
- Replacing the optimizer gradient-norm block reduction with a two-stage
  warp-shuffle reduction cut its barriers from eight to one and improved the
  isolated kernel 8--14%. Matched 8-update G1 runs reduced graph-captured
  PPO updates 96.94 to 96.39 ms and eager updates 104.89 to 103.96 ms;
  NumPy parity and exact graph replay pass.
- Reusing contact transforms during reduced gather removed a second per-column
  geometry traversal without changing hard-Hertz bias arithmetic. Long 8,192-world
  runs improved point contacts 1.378M to a 1.400M candidate mean (+1.62%) and
  patch contacts 1.478M to 1.503M steps/s (+1.72%) at identical contact counts.
  Bitwise patch trajectory, multipage, refresh-stride, fused-apply, and deterministic
  momentum regressions pass.
- Static omission of unused external-pass Coriolis publication reduced external
  advance 112.176 to 104.736 us. Reusing invariant patch effective masses cut
  cached patch solve about 18% and total patch solve 6.8%, with bitwise
  production trajectories.
- Removing abandoned response-basis metadata cut page counting 16.94 to 4.05 us;
  removing redundant internal twist stores reduced external advance about 7%
  and fused advance/publication about 2.4%.

### Current quality and audit caveats

- FP16x2 production contact rows are removed; production rows are FP32.
- Seed 42 was also the strongest early seed in a 11/29/42/47/73 screen. This is
  evidence of seed selection, not equal sample efficiency across simulators.
- Preserving MinGRU state across rollouts is supported but did not fix the
  acquisition gap and reduced throughput in the tested ablation.
- Exact weaker nanoG1 reward weights improved seed-42 acquisition; a later ramp
  toward robust weights improved tracking but narrowly missed stability gates.
  This motivates a multi-seed curriculum study, not a single-seed default change.
- Corrected production drive-convergence settings are three substeps, two
  iterations, reduced articulations, and explicit torque. Against a 20x8
  PhoenX reference, the tested perturbation had zero falls, 0.00059 rad joint-q
  RMS, 0.00665 rad/s joint-qd RMS, and 0.35 mm base-state RMS.
- Full-training throughput was broad and flat near 8,192--10,240 worlds:
  4,096/6,144/8,192/10,240/12,288 measured
  1.415/1.574/1.671/1.664/1.608M samples/s. Device-side batch search has no
  demonstrated payoff on this GPU.
- A 12-output policy gained only 1.4% and hurt acquisition robustness; removed.
  Adaptive KL reacted to noisy early observations and blocked acquisition;
  future controllers need normalized signals, confidence gates, and hysteresis.
- Single-GPU PBT workers execute through a Python loop, not concurrently. The
  eager G1 continuation once discarded interval history and produced `-inf`
  fitness; the shared corrected loop now has finite-history and continuation-
  equivalence regressions.
- Shared row slabs, resident row caches, and register-capped advance kernels are
  specifically rejected: they regressed 14%/8.6% or spilled 80-byte writes and
  44-byte reads per thread for a noise-level +0.19% median.

- A one-launch Kapla Nsight Compute capture attributed the worst sector
  inefficiency to scalar indexed endpoint-property loads in the persistent
  contact solve. Each instruction in the inverse-mass/inertia cluster reported
  about 10,213 excessive L2 sectors, and its consumers dominated sampled
  long-scoreboard stalls. Staging both inverse masses and both symmetric
  six-float world inverse inertias into the existing color-ordered header
  changed a matched 80-frame run from 83.03 to 85.30 median FPS (+2.7%) and
  82.09 to 83.49 mean FPS (+1.7%). It is fused into prepare, adds no launch,
  passed 20 packed-row/contact-force/tower tests, and was retained in
  `c29693f4f`.

- A post-integration Nsight Systems A/B localized that end-to-end gain: over
  60 steady frames the repeated PGS kernel fell 360.39 to 345.28 ms (-4.2%)
  and relax fell 38.53 to 34.43 ms (-10.6%), while prepare rose 68.66 to
  72.78 ms (+6.0%). A 5,500-contact layout oracle then measured 4.51 us for
  node-major random velocity reads/writes versus 3.40 us for coalesced
  contact-local reads with scattered successor writes (1.33x). This is only an
  upper bound: a production successor layout must preserve mass-splitting
  average/broadcast semantics and both deterministic sweep directions.

- Ordinal-major contact rows looked 1.23x faster in a 5,556-column isolated
  load/store oracle, but a capacity-safe hybrid implementation regressed the
  matched Kapla run from 84.10 to 80.67 FPS (-4.1%). Nsight Systems showed
  that the regression was inside the solver, not only its extra map build:
  repeated PGS rose 5.33 to 5.50 us, prepare 11.23 to 11.48 us, and relax
  5.31 to 7.01 us. The wider physical row extent and address/control overhead
  outweighed better sector use in the full working set. The implementation was
  removed; do not infer production wins from the small row-layout oracle.

- Row-local AoS packing also failed its isolated gate. Against the current SoA
  row stream at 8.14 us, scalar AoS took 10.39 us (-22%) and six aligned vec4
  chunks took 10.03 us (-19%). Cross-thread SoA coalescing is more valuable
  than per-thread row locality for this access pattern; no production code was
  changed.

- An eight-lane cooperative manifold oracle improved row fetch and sequential
  projection about 19.5% using width-eight shuffles. The real regular-color
  implementation preserved the packed-row dynamics tests but reduced matched
  Kapla throughput from 83.7 to 78.2 FPS (-6.6%). Assigning eight lanes per
  manifold reduced available manifold-level parallelism and increased register
  demand more than coalescing repaid. The implementation was removed.

- A fresh whole-frame Kapla trace assigned 49.2% of GPU time to repeated PGS,
  9.7% to prepare, 6.3% each to rigid mass-splitting average/broadcast and
  relax, 13.9% to collision sort/broad/narrow phase, and about 4% to graph
  selection/commit. An eight-lane rigid average/broadcast specialization
  reduced its real-copy-state kernel from 6.23--6.30 to 4.10--4.30 us
  (32--35%). Source-lane-ordered shuffle gathers are byte-exact to the scalar
  sum for randomized copy counts 0--33. Candidate/control/candidate Kapla
  measured 86.35/84.20/86.21 FPS, a 2.47% candidate-mean gain with identical
  contact counts and motion diagnostics. This path is selected by rigid
  velocity-level mass splitting, independent of whether its slots were
  produced by joints or contacts. Reduced-coordinate articulation blocks and
  the generic mixed/deformable synchronization path are unchanged.

- Kapla's production grid launches 48,128 threads although its eight regular
  colors contain only 4,287--5,535 constraints; overflow contains 32,256.
  Reducing residency to 4/2/1 blocks per SM measured 83.78/82.28/75.99 FPS
  versus 85.09--86.58 at 8 blocks per SM. A real-distribution next-column
  prefetch oracle was neutral with one item per thread and 25--33% slower with
  2--3 items per thread; its PTX also had substantially more live registers.
  Both production ideas were rejected and removed.

- A fresh current-recipe G1 trace at 8,192 worlds assigns about 30% of GPU time
  to reduced advance/publication, 13.9% to packed patch-row construction,
  19.9% to the two contact-solve variants, 9.5% to factorization, and 6.2% to
  factor initialization. Omitting the final internal twist-workspace store
  from fused publication reduced that kernel's median 200.35 to 198.09 us
  (1.1%), but complete throughput remained about 2.52M environment steps/s in
  the matched bracket. The candidate was removed as too small.

- Specializing the packed patch-row kernel for its already-selected 8-thread
  articulation tile did not improve G1 throughput. Seven candidate/control
  runs measured medians of 2.49936/2.49967M environment steps/s (-0.01%).
  Runtime tile-width division, masks, and shuffle width are therefore not a
  material bottleneck; the factory specialization was removed.

- Replacing the packed-row topology chase with full-size depth-ordered arrays
  increased its median from 150.24 to 160.59 us (+6.9%) and reduced complete
  G1 throughput about 1.1%. It preserved the number of PTX topology loads while
  increasing address state and footprint, so it was removed.

- Deduplicating repeated topology signatures and packing local joint offset,
  DOF offset/count, and parent lane into one aligned `vec4i` descriptor did
  emit `ld.global.v4.u32` and reduced the packed-row median from 150.24 to
  143.05 us (-4.8%). It passed byte-exact G1 scalar trajectory parity,
  heterogeneous-template checks, 8/16/32-wide scalar parity, mixed
  reduced/maximal contacts, analytical G1 torque, full-coordinate
  determinism, and momentum conservation. The final close bracket improved
  complete throughput only from 2.51573 to 2.52157M environment steps/s
  (+0.23%), too little for the added representation and native accessor. The
  production candidate was removed; reuse by the larger reduced-advance walk
  is the only justified follow-up.
