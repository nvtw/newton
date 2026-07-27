# PhoenX + PhoenXRL Optimization Charter

Standing goal for solver and RL performance work. Supersedes ad-hoc prioritization.
Evidence lives in [PERF_NOTES.md](PERF_NOTES.md); this file states the objective,
the acceptance policy, and the ranked backlog.

Established 2026-07-27, after the incremental lane was declared exhausted.

## 1. Goal

Two primary metrics, tracked separately and never traded silently:

1. **Wall-to-policy** — median wall-clock from random init to a frozen G1 gate,
   over a frozen seed set. Decomposes into `samples-to-gate ÷ samples/s`.
2. **Physics throughput across the scene matrix** — steps/s on every regime, not
   just the one being optimized:

   | Regime | Representative scene | Why it is in the matrix |
   | --- | --- | --- |
   | Large single world, contact-dominated | Kapla 11,340 bricks | Latency-starved, 9-color barrier |
   | Many small worlds, reduced coordinate | G1 flat @ 4096/8192 | Drives RL wall-clock |
   | Many small worlds, maximal coordinate | H1 flat, dr_legs @ 4096 | Different register/dispatch regime |
   | Deformables / particles | cloth, soft-body drop | Different coloring and copy-state pressure |
   | Articulated stress | Anymal PBT fleet | Convergence quality under load |

A change is characterized by its effect on **all** of these, but is not required
to win on all of them (see §3).

## 2. Non-negotiable constraints

- Hard Hertz contact model and high-fidelity dynamics preserved.
- Deterministic replay for identical seeds; CUDA-graph capture compatible.
- No scene or robot topology baked into kernels. Static kernel factories and
  feature-axis `wp.static` specialization are fine; runtime if-elif sprawl and
  scene detection are not.
- Physics-quality gates are binding. Speed never buys quality.
- Root-cause fixes with regression tests, not workarounds.

## 3. Acceptance policy (revised 2026-07-27)

The old policy — "reject unless it wins the production bracket" — is what
produced the plateau. It systematically rejects the first step of any two-step
architectural change. Revised rules:

- **Keep a clear, isolated win on one regime that is neutral elsewhere.** A
  measured Kapla-only or cloth-only gain with no regression on the rest of the
  matrix is a KEEP, not a reject.
- **Enabling value counts.** A change that is neutral on its own but is the
  precondition for a named, credible follow-up may be kept on a branch and
  merged, provided the follow-up is written down with its own accept/reject
  metric. Neutral + no named follow-up = revert.
- **Neutral on RL wall-to-policy is not a rejection** for a physics-side win.
  Training is stochastic and rollout/learner overlap hides gains; physics
  steps/s is the primary signal, the training bracket is confirmatory only
  (see `feedback_keep_clear_solver_speedups`).
- **Negative results are deliverables.** Every rejected experiment lands as a
  `PERF_NOTES.md` row with numbers and the structural reason, on a commit that
  can be resurrected.
- **A regression anywhere in the matrix still blocks a merge to the trunk
  branch** unless it is explicitly traded and recorded.

Every experiment gets its own branch `twidmer/phoenx-<experiment>` and at least
one commit, so a failed swing is one `git checkout` away from being revisited
with new evidence.

## 4. The regime split (the central reframing)

The three hot regimes want **opposite** prescriptions. Most cross-applied
lessons in the rejected-ideas table are explained by ignoring this.

| Regime | What counters say | Prescription |
| --- | --- | --- |
| **Kapla single-world** | ~1 item/thread, 5–6% achieved occupancy, 91% no-eligible-warp cycles, 60% long-scoreboard, 15–17% DRAM. ~140 registers/thread of unused budget. | Latency-starved. Buy **memory-level parallelism** or **eliminate work**, not bytes and not occupancy. |
| **G1 reduced** | 14.6 of ~19 warps/SM, no spare registers, dependent address chase on the critical path, 53–75% L1TEX dependency gaps. | Spend nothing. **Remove dependency levels.** |
| **RL learner** | ~99 ms update fully hidden behind ~273 ms rollout under leapfrog overlap. | Throughput work has ≈0 marginal value for G1. Optimize **rollout (= physics)** or **sample efficiency**. |

**Previously measured and rejected** — treat as prior, not proof: cooperative
grid sync (3.0–3.3x slower), eight-lane cooperative manifolds (4.095 vs
4.099 us), cross-item prefetch via a shrunk Kapla grid (82.40 vs 82.76–82.78
FPS), separate regular-color grid, indexed AoSoA8 rows, per-substep packed
endpoint mass/inertia, register-cap occupancy hunting, world-interleaved
reduced ABA.

### Provenance rule

**Nothing in the recorded evidence is authoritative, including this file.**
Every table row is a past measurement under a past code state, hardware, Warp
version, scene configuration, and — often — a past understanding of the
mechanism. Two failure modes are already visible in the record: a rejected
result whose *stated reason* does not follow from its own numbers, and a
recommendation (the 2026-07-27 external review's top item) produced without GPU
access that contradicts a measured row it never read.

So:

- A recorded rejection lowers the priority of an idea. It does not forbid it.
- **Re-test rather than cite** when any of these hold: the mechanism has changed;
  the prior result was within noise; the reason given is structural rather than
  measured; the idea is now a *component* of a larger change rather than a
  standalone one; or the rejection predates a relevant refactor.
- When a re-test contradicts a recorded row, **correct the row** and note both
  measurements. Do not append a second contradictory row.
- Prefer a fresh 30-minute measurement over an hour of reasoning about an old
  one. Cite numbers with their date and code state, or do not cite them.

The 2026-07-27 external review is a useful hypothesis generator whose factual
claims are explicitly marked as derived, not measured. Its §0 reframing (the
Kapla kernel is latency-starved with spare register budget) is worth testing
directly even though its concrete E1 proposal collides with a recorded row —
the recorded row tested *one* implementation of that idea, not the idea.

## 4b. Method: the loop is the goal

The deliverable is not a list of optimizations. It is a **measure → reason →
change → re-measure** loop run with research discipline, indefinitely, across
the scene matrix.

1. **Measure.** Profile the current production configuration. Attribute time to
   kernels; get counters (`nsys` for phase attribution, privileged `ncu` for
   dependency/cache/sector analysis) for whatever dominates *now*.
2. **Reason.** Form a mechanism hypothesis that explains **all** the counters,
   not the one that fits the idea already in hand. Write down the number that
   would falsify it.
3. **Change.** Smallest edit that tests the mechanism. Prefer an isolated probe
   or an ablation over a production integration when the probe can decide.
4. **Re-measure.** Full scene matrix. Accept, reject, or — most often —
   discover the mechanism was misidentified and return to step 2.
5. **Record and repeat**, then re-profile: the hot spot moves after every win,
   and the next cycle's target must come from the new profile, not the old
   backlog.

Two failure modes to avoid, both visible in the project history: optimizing the
kernel that *was* hot, and accepting a mechanism story that explains one counter
while contradicting another (bytes-and-occupancy reasoning on a
latency-starved kernel).

## 5. Current hypothesis queue

**Not a mandate.** This is the state of the loop as of 2026-07-27 — the
hypotheses that survive the evidence on hand. Any of them may be displaced by
the next profile, and a fresh measurement outranks any entry here. Ordered by
information-per-engineering-hour, not by expected size. Tier 0 prices the rest.

### Tier 0 — cheap measurements that re-rank everything (hours)

- **T0.1 Launch-floor bound.** Replay the captured Kapla graph with the 90
  iterate launches per substep replaced by a geometry-identical no-op kernel.
  The GPU time is the exact dispatch floor. Prices every fusion/barrier idea
  before anyone builds one. Floor ≤20% of the 5.5 us average → close the whole
  barrier-removal line permanently.
- **T0.2 Per-partition element histogram** for the settled tower. The unrolled
  dispatcher disables tail fusion and issues a full grid for every partition.
  If the tail partitions are a few hundred rows, they are pure launch floor.
- **T0.3 `(K partitions, I iterations)` stability surface.** Cost per substep is
  `(K+1) x I` launches; production is `(8,10) = 90`. "Eight partitions is the
  stability floor" was measured at fixed `I`. Benchmark time only, zero code.
- **T0.4 Sleep/idle census on settled Kapla.** What fraction of the 11,340
  bricks are below a rest threshold after settling, and for how long? Prices
  T3.1 exactly.

### Tier 1 — bitwise-exact structural (days)

- **T1.1 Depth-ordered joint descriptor + fleet-wide topology dedupe.** One
  coalesced 32 B structure load replaces the `joint -> child -> dof_start`
  pointer chase in the three depth traversals; for a homogeneous fleet, store
  **one** table (~1 KB, permanently L1-resident) instead of 8,192 copies
  (~7.8 MB). Bitwise exact. Already an Open Idea with a >=5% bar on fused
  advance/publish. Audit `reduced_contact_block.py:_build_packed_generalized_row`
  for the same chase first.
- **T1.2 Morton body renumbering at build time.** Independently useful (tightens
  the locality the existing within-color sort exploits), and it is step 1 of
  T2.1 — a textbook enabling change under §3.

### Tier 2 — ordering-changing architectural (weeks, gated)

- **T2.1 Resident subdomain block Gauss–Seidel.** Within a color every body
  appears once, so there is **no body-state reuse inside a color** — on-chip
  body caching is worthless unless several colors run inside one block, which
  requires spatial partitioning. Morton-partition bodies into `P ~ 2-4 x SMs`
  parts, stage each part's body state into shared memory (coalesced after
  T1.2), run all local colors x iterations with `__syncthreads()`, and route
  boundary constraints through the existing copy-state machinery.
  **Gate first, offline, no kernel:** interior/boundary constraint fraction vs
  `P`. Below ~60% interior at `P = 188`, drop it. Changes the fixed point
  (block-GS / additive-Schwarz), so it gates on invariants and quality, not
  trajectory equality.
- **T2.2 Blackwell thread-block clusters.** `cluster.sync()` plus distributed
  shared memory across up to 16 blocks is the only barrier mechanism materially
  cheaper than an atomic spin. Warp has no cluster API; `wp.func_native` is the
  entry point. Pure enabling change: it lets a T2.1 part span 16 blocks and
  directly fixes the surface-to-volume problem that is T2.1's main risk. Only
  after T0.1 shows a large floor and T2.1's offline gate passes.

### Tier 3 — algorithmic work elimination (the largest untried lever)

Everything above tunes a fixed amount of work. These change how much work
exists, and neither is in the rejected table.

- **T3.1 Sleeping / island stepping.** `islands/island_builder.py` and
  `sleeping_kernels.py` exist and are reserved for this. A settled Kapla tower
  is the canonical case: bricks that are at rest cost full solver iterations
  every substep forever. Priced by T0.4. Generic (granular, debris, any large
  quasi-static scene), CUDA-graph-compatible if the sleep mask is a device-side
  array and the launch geometry stays fixed. Determinism requires a fixed-order,
  device-side wake/sleep rule — no host readback.
- **T3.2 Multilevel / coarse correction for stacks.** Colored GS converges
  poorly on the low-frequency modes of tall stacks — that is *why* the recipe
  needs 10 iterations x 8 partitions. A coarse-level correction is the standard
  fix and reduces iteration count rather than iteration cost. The `multilevel/`
  package is currently empty. High risk, highest ceiling: iterations are a
  multiplicative factor on 66% of Kapla GPU time. Must be gated on convergence
  quality, not FPS.

### Tier 4 — RL wall-to-policy

- **T4.1 Censored-data measurement protocol.** Samples-to-gate is
  right-censored; seeds that never reach the gate have no finite value, so means
  over finishers are biased — which is exactly the recorded seed-42 selection
  effect. Adopt: >=8 seeds frozen before comparison, Kaplan–Meier estimate of
  `P(gate <= N samples)` with non-reachers entered as censored, log-rank for A/B,
  bootstrap CI on the KM median, reach-rate reported separately from
  time-to-reach. With 8 seeds this detects roughly a 2x hazard ratio — any claim
  of a 10–20% sample-efficiency win from 8 seeds is not measurable, and should be
  labeled so up front. Prerequisite for T4.2 producing a decision rather than an
  anecdote.
- **T4.2 AMP / motion prior.** The standing big swing (see
  `project_phoenx_amp_motion_prior`): attacks the reach *rate* (seeds that never
  walk), which is a different failure mode from time-to-reach and the one
  throughput work cannot touch.
- **T4.3 Learner throughput: frozen for G1.** Fully hidden behind rollout.
  Remaining open ideas (FP32 cuBLAS backward contractions, Muon Newton–Schulz
  through cuBLAS) are headroom for Ant/PBT and code simplification, not G1
  wall-clock. Do not measure them against G1.

## 6. Per-experiment protocol

1. Branch `twidmer/phoenx-<experiment>`. State the hypothesis, the accept/reject
   number, and the falsifier **before** measuring.
2. Benchmark solo on the GPU (poll `nvidia-smi` idle first). CUDA + graph
   capture, reversed A/B order, drop trial 0. Quality/convergence metrics may
   run under contention; timings may not.
3. Correctness gate by class:
   - bitwise-exact changes → exact trajectory equality + capture/replay tests;
   - ordering changes → invariants and quality (drift, settle, contact count,
     stacking, analytical anchors), never bitwise;
   - both → a regression test for the property the change could break, verified
     to fail without the change.
4. Report the full scene matrix, including the neutral entries.
5. Land the result either way: code + `PERF_NOTES.md` row, or `PERF_NOTES.md`
   row alone on a preserved branch.

## 7. Stop conditions

- Kapla: if T0.1 shows a small launch floor **and** T2.1's offline gate fails
  **and** T3.1/T3.2 are rejected on quality, then record that 11,340 bricks
  underfill a 188-SM Blackwell by ~10x, that the scene is at its structural
  ceiling, and stop.
- G1 physics: if T1.1 lands and the next profile shows no single kernel above
  ~10% with an identified dependency defect, stop and move all effort to T4.
- RL: throughput work stops when rollout no longer dominates. Sample efficiency
  work stops when the KM curves of two arms are statistically indistinguishable
  at the frozen seed count.
