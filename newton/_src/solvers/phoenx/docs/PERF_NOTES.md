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
| Cooperative grid-sync iterate megakernel | Synchronization and residency constraints lost. |
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
| Reuse checkpoint MinGRU kernel for read-only sequences | Only a 1.1% forward-subphase change and no measurable full-training gain. |
| Fuse next-layer BF16 shadows into MinGRU recurrence | Isolated recurrent graph improved 0.730 -> 0.713 ms, but the extra dependent store reduced full A/B/B/A throughput 1.940M -> 1.927M samples/s; removed. |

## Open ideas

- Improve reuse of topology/factor data within dependent packed-row traversal.
- Test truly world-interleaved hot joint/body fields only if profiling identifies
  repeated scalar transactions; depth-local packing and vec4 padding already lost.
- Benchmark FP32 cuBLAS backward contractions for non-production FP32 learner
  configurations.
- Benchmark complete Muon Newton--Schulz steps before routing its remaining
  fused GEMMs through cuBLAS.

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
