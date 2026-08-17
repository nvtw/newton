# FlashSAC engineering notes

This document records implementation, correctness, and performance findings for
the pure-Warp PhoenX RL FlashSAC trainer.  Keep unsuccessful experiments here as
well as retained optimizations: a fast microbenchmark is not sufficient evidence
that a training change is safe.

## Authorities and acceptance gates

The algorithm reference is the local FlashSAC checkout at
`/home/twidmer/Documents/git/FlashSAC`, commit
`87edc9061150ae9e962dd84e6544e27a1554b3ab`.  In particular, its
`flash_rl/agents/flashSAC/layer.py`, `network.py`, and `update.py` define the
network equations, autocast scope, and update order.

Performance measurements must use an otherwise idle GPU, warm up CUDA graphs,
and report repeated windows rather than one launch.  The main throughput
benchmark uses 1,024 G1 worlds and one captured cadence containing two
interactions and four learner updates.  Learning quality uses the full 29-action
G1 recipe, a fixed 0.8 m/s command, deterministic 200-step evaluation, and
requires two consecutive evaluations satisfying all of:

- tracking score at least 0.30;
- command-aligned velocity at least 0.4 m/s;
- fall fraction at most 0.06.

Standing is not locomotion.  The compact 12-action task and tracking score alone
can reward a stationary policy and must not be used as the quality gate.

## CUDA graph coverage

The steady-state graph includes policy inference and exploration, the PhoenX
step and auto-reset, n-step replay insertion, reward normalization, replay
sampling, actor/temperature/critic updates, target EMA, Adam, learning-rate and
seed progression, and counters.  Graph/eager state equivalence, checkpoint
continuation, termination/truncation handling, and pre-reset next observations
have focused regressions in `tests/test_flash_sac.py`.

All graph-owned arrays are allocated during setup/capture.  Replay work is
negligible compared with the learner; optimize the network before replay.

## Retained FP32 optimizations

The initial captured implementation measured approximately 16.09 ms per mixed
learner update and 68.39 ms per full cadence, or 29.9k transitions/s.  The
following exact FP32 changes reduced this to approximately 4.96 ms per update
and 23.2 ms per cadence, or 88k transitions/s:

- fused twin-critic dense contractions;
- setup-owned reduction workspaces;
- packed Warp tile reductions for BatchNorm, RMSNorm, bias, affine, and weight
  normalization;
- fused normalization/ReLU forward kernels;
- cuBLAS contractions with FP32 accumulation;
- complete learner and collector CUDA graph capture.

Validate every optimization with equation fixtures, finite-difference backward
checks where applicable, eager/graph state comparison, deterministic learning,
and the full PhoenX FlashSAC test module.

## Mixed-precision findings

Upstream uses CUDA autocast with `torch.float16` and `GradScaler`.  BF16 is not a
drop-in substitute on the measured RTX PRO 6000 Blackwell.  Persistent-operand
FP16 GEMMs are substantially faster and more accurate than BF16 for the dominant
FlashSAC shapes, while isolated conversion-heavy BF16 forwards were slower than
FP32.

Autocast is not a uniform whole-network dtype.  A direct PyTorch 2.10 CUDA trace
of the authoritative source found:

- Autocast is active only in actor and critic learner updates.  Environment
  interaction and evaluation use FP32.
- The scalar actor input BatchNorm receives and returns FP32.  Dense block
  linears return FP16; block BatchNorm/ReLU, residuals, and actor RMSNorm remain
  FP16.
- In the fused critic, each ensemble linear returns FP16.  The custom ensemble
  BatchNorm computes mean and variance with FP16 result dtype, `rsqrt` promotes
  to FP32 only after the FP16 `variance + epsilon` addition, and the FP32 affine
  parameters make its output FP32.  The centered subtraction is FP16 before it
  is multiplied by the FP32 inverse standard deviation.  Critic residuals,
  RMSNorm, categorical logits, probabilities, values, and losses remain FP32.
- The actor predictor and critic categorical predictor have different promotion
  behavior.  Actor mean and raw log standard deviation are FP16; its bias add,
  tanh, affine log-standard-deviation map, and stored log standard deviation are
  also FP16, while `exp(log_std)` returns FP32.  The critic head linear is FP16,
  but adding its FP32 bias promotes categorical logits to FP32.
- Master parameters, optimizer state, running statistics, and parameter
  gradients remain FP32.

The first opt-in AMP implementation incorrectly retained the whole critic
residual branch in FP16.  It was fast (about 3.08 ms/update and 15.57 ms/cadence,
131.5k transitions/s) but failed the G1 quality gate through 15.1M transitions.
Correcting the visible critic output boundaries retained a useful speedup (about
3.39 ms/update and 16.99 ms/cadence, 120.5k transitions/s) but still failed the
quality gate.  A later arithmetic audit found that the critic moment calculations
were still retained in FP32 rather than reproducing the upstream FP16 result
rounding.  Do not enable AMP by default until a fixed PyTorch-versus-Warp fixture
matches forward values and selected gradients and the full G1 gate passes.
Forcing only collector/evaluation inference back to the authoritative FP32 path
did not repair learning: seed 0 still failed through 15.1M transitions (tracking
0.109, aligned velocity 0.171 m/s, fall fraction 1.0).  A committed NumPy fixture
must preserve the internal actor-head and ensemble-moment values traced from
PyTorch without requiring PyTorch at test time; whole-network actor/critic output
and selected-gradient fixtures remain the stronger acceptance requirement.

The subsequent staged implementation matched the traced actor-head, ensemble
moment, custom BatchNorm backward, and RMSNorm equations.  On a mapped tiny
network, the actor forward differed by at most 1.2e-7 and its input and parameter
gradient directions agreed to floating-point precision.  The critic forward and
summed input-gradient cosines were 0.9999993 and 0.9999816, respectively;
isolated custom BatchNorm input gradients matched PyTorch exactly at widths 4
and 16.  The remaining critic error was consistent with FP16 GEMM/reduction
ordering.

This closer implementation measured 3.46 ms per update versus 5.01 ms for FP32,
and 17.14 ms per complete cadence versus 23.08 ms for FP32 (119.5k versus 88.7k
transitions/s).  It nevertheless failed the seed-0 quality gate through 15.09M
transitions: the final evaluation had tracking 0.010, aligned velocity 0.038 m/s,
and fall fraction 1.0.  It must remain opt-in and is not a quality-preserving
replacement for FP32.  Exact upstream gradient-scaling behavior and accumulated
FP16 contraction drift remain candidates for the discrepancy.

An authoritative PyTorch oracle at batch size 2,048 confirmed that scaling is
material even without widespread underflow.  Across three seeds, autocast
without scaling lost 121--145 actor gradient entries and 4,055--4,104 critic
entries, compared with 15--20 and 0--1 under `GradScaler(65536)`.  After
unscaling, the worst per-layer cosine between the two paths was only 0.915 for
the actor and 0.976 for the critic.  A faithful graph-safe implementation must
therefore scale before FP16 network backpropagation, unscale FP32 master
gradients before Adam, and keep overflow detection, skipped steps, and scaler
state device-resident.

A fixed-scale placement experiment then confirmed causality.  Scaling critic
logit gradients before critic backpropagation, scaling both Q and entropy paths
of the actor loss before their first FP16 boundaries, and unscaling FP32 master
gradients before Adam restored seed-0 learning.  The fixed-scale run first
passed at 8.096M transitions (tracking 0.796, aligned velocity 0.448 m/s, no
falls) and sustained at 8.595M (tracking 0.846, aligned velocity 0.796 m/s, no
falls).  This is evidence for scaler placement, not an accepted implementation:
dynamic growth/backoff, global overflow detection, skipped optimizer state, and
checkpointed scaler state still require graph-native coverage.

The retained graph-native scaler starts at 65,536, grows by two after 2,000
successful optimizer steps, and backs off by one half on overflow.  Actor and
combined-critic overflow reductions are global: Adam parameters, moments, and
step count are skipped together, while upstream's parameter normalization,
learning-rate scheduling, and target EMA still run.  The shared scaler advances
after the actor and critic optimizer calls exactly as upstream does, and its
scale and growth tracker are checkpointed.  Forced-overflow, eager/graph
continuation, checkpoint continuation, and exact growth/backoff equations have
focused regressions.  The seed-2 sustained checkpoint had scale 16,777,216 and
growth tracker 1,646, confirming that long captured runs exercise dynamic
progression rather than retaining the initial scale.

With dynamic scaling, the captured learner measured 3.67 ms per update and the
complete cadence measured 17.84 ms, or 114.8k transitions/s.  The corresponding
FP32 cadence was 23.08 ms, or 88.7k transitions/s.  Three independent full-G1
seeds all first passed at 7.596M transitions and sustained at 8.096M.  Their
sustained training-only times were 73.38, 71.35, and 70.34 seconds; tracking was
0.840, 0.742, and 0.838; aligned velocity was 0.664, 0.664, and 0.540 m/s; and
all had zero falls.  This establishes both quality preservation and a roughly
23 percent end-to-end throughput improvement for the measured configuration.

Loss scaling should not be assumed to fix a forward mismatch.  The fixed
PyTorch/NumPy fixtures for actor heads, critic moments, custom BatchNorm
backward, and RMSNorm remain required even though the end-to-end quality gate
now passes.

## Profiling history

Before mixed precision, the optimized learner was dominated by GEMMs, followed
by BatchNorm moments/backward and normalization.  An Nsight Compute report for
the earlier normalization kernels is stored outside the repository at:

- `/tmp/newton_flash_sac_norm_ncu.ncu-rep`;
- `/tmp/newton_flash_sac_norm_ncu.txt`;
- `/tmp/newton_flash_sac_norm_ncu_details.csv`;
- `/tmp/newton_flash_sac_norm_ncu_raw.csv`.

The report showed severe underfill for narrow per-column BatchNorm reductions
(roughly 0.09--0.11 waves/SM for widths 98 and 128), while wide elementwise
normalization was healthy.  Chunked partial reductions and single-block grouped
tile reductions were measured and rejected: preserving the exact two-pass
centered variance required enough staging to lose end-to-end time.

The retained alternative transposes each BatchNorm input with a coalesced 32 by
32 tile into a setup-owned feature-major workspace, then evaluates the same
two-pass equation with the same lane assignment and tile reduction order.
Multi-shape FP32 and FP16 tests confirm bit-exact means, variances, and inverse
standard deviations, including row and column tails.  On the corrected AMP
graph this reduced complete cadence time from 17.55 ms to 15.68 ms and raised
throughput from 116.7k to 130.6k transitions/s.  A projection/EMA-to-FP16-mirror
fusion was also rejected after focused correctness passed because its measured
full-cadence improvement was only about 0.3 percent.
Fresh seed-0 quality first passed at 8.096M transitions and sustained at 8.595M
in 66.40 seconds of training, with tracking 0.860, aligned velocity 0.738 m/s,
and zero falls.
The staged AMP BatchNorm backward now reuses the feature-major forward input and
transposes its output gradient once, preserving every FP16 rounding boundary and
reduction result.  This further reduced cadence to 15.00 ms (136.6k transitions/s).
Fresh seed-0 quality first passed at 7.596M and sustained at 8.096M transitions
in 63.92 seconds, with tracking 0.826, aligned velocity 0.608 m/s, and zero falls.

The ensemble BatchNorm, residual-add, and RMSNorm producers now write their
FP32 activations and exact FP16 mirrors together.  Forward and weight-gradient
GEMMs reuse those setup-owned mirrors, removing separate activation casts while
retaining the original FP32 state.  Bit-exact producer fixtures cover training
and inference normalization, residual addition, and RMS normalization.  The
complete cadence measured 14.23 ms (143.9k transitions/s).  Fresh seed-0 quality
first passed at 7.596M transitions and sustained at 8.096M in 57.07 seconds,
with tracking 0.863, aligned velocity 0.466 m/s, and zero falls.

The opt-in multi-stream cadence uses a separate FP32 rollout-policy snapshot
and two fixed learner-batch phases.  Each preparation graph runs only after the
previous rollout and update graphs have joined: it pre-samples the next learner
batches from stable replay state and copies the updated actor into the rollout
snapshot.  The following rollout and learner graphs can therefore overlap
without concurrent replay reads/writes or learner/collector parameter races.
The public handle drains all streams explicitly for checkpoints and teardown.

After warming both construction paths in one process, combined setup measured
0.919 s and overlap setup measured 1.165 s.  Including phase preparation, batch
copies, joins, rollout, and all four learner updates, their cadences measured
14.136 ms (144.9k transitions/s) and 11.780 ms (173.8k transitions/s),
respectively.  The additional setup cost breaks even after roughly 105 cadences,
or 0.215M transitions at 1,024 worlds.

Three fresh seeds preserved the fixed full-G1 quality gate.  Seeds 0 and 1
first passed at 7.596M transitions and sustained at 8.096M; seed 2 first passed
at 8.595M and sustained at 9.095M.  Their sustained tracking was 0.770, 0.849,
and 0.761; aligned velocity was 0.458, 0.696, and 0.921 m/s; all had zero falls.
Training times including setup were 57.96, 47.49, and 53.13 seconds.  The mode
remains opt-in so callers explicitly choose the one-cadence replay eligibility
delay and additional policy/batch storage.

The CUDA distributional critic loss and backward pass assigns one 128-lane
tile block to each replay row.  The block reduces the two softmax maxima,
normalizers, and cross-entropy loss in parallel while writing the same
per-atom gradients; the scalar implementation remains the CPU fallback and
test oracle.  Multi-shape tests cover 7 and the default 101 atoms.  With AMP
and overlapped rollout, complete cadence improved from 11.780 ms to 11.365 ms
(173.8k to 180.2k transitions/s).  Fresh seed-0 quality first passed at 7.096M
transitions and sustained at 7.596M in 53.52 seconds, with tracking 0.653,
aligned velocity 0.476 m/s, and zero falls.  A preceding ReLU/BatchNorm
backward fusion was rejected despite a repeatable 1.4 percent cadence gain:
its isolated equations passed, but the full-G1 gate failed through 15.09M
transitions.

### Environment-count and batch-size sweep

An early scaling diagnostic changed a nonexistent standalone ``WORLDS = 1024``
assignment while the harness actually declared a tuple.  Phases previously
labelled as 2,048, 4,096, or 8,192 worlds before the ``*_corrected_*`` phases
therefore all ran with 1,024 worlds and are invalid as world-scaling evidence.
The raw profiler results were separate invocations with an explicit world-count
argument and remain valid.

The corrected harness asserts both the requested and constructed environment
counts, two interactions per cadence, and four learner updates per cadence.
At batch size 2,048, seed-0 sustained-gate results were:

| Worlds | Samples | Updates | Training wall [s] |
| ---: | ---: | ---: | ---: |
| 1,536 | 13,120,512 | 17,084 | 53.204 |
| 2,048 | 15,591,424 | 15,226 | 48.690 |
| 2,560 | 20,070,400 | 15,680 | 51.392 |
| 3,072 | 20,505,600 | 13,350 | 51.537 |
| 4,096 | 28,585,984 | 13,958 | 49.390 |
| 8,192 | 50,077,696 | 12,226 | 51.625 |

The 2,048-world configuration was then repeated at seeds 1 and 2 and sustained
the gate in 50.236 and 50.469 seconds; its three-seed median is 50.236 seconds.
For comparison, three phases that were mislabeled during the initial sweep but
actually used 1,024 worlds sustained in 48.168, 51.104, and 54.158 seconds
(median 51.104 seconds).  Raising the 2,048-world batch size to 4,096 restored
four replay items per transition but reduced warm throughput from 328.8k to
208.3k transitions/s and sustained only after 81.249 seconds.  It was rejected.

### Full-action G1 learning-rate sweep

An asserted 2,048-world sweep retained batch size 2,048, two interactions and
four updates per cadence, policy delay 2, AMP, overlapped rollout, reward
normalization, n-step 3, target rate 0.01, and a 100,000-update cosine schedule.
Raising the actor, critic, and temperature learning rates together from
``3e-4`` to ``6e-4`` sustained the fixed 0.8 m/s gate at seeds 0, 1, and 2 in
40.755, 34.327, and 32.980 seconds including setup but excluding separately
recorded evaluation overhead.  The corresponding transitions were 13.093M,
11.094M, and 10.594M; all sustained evaluations had zero falls.  The median
34.327 seconds is 31.7 percent below the prior three-seed median of 50.236
seconds.  The tuned rate is exposed only through
``isaaclab_flat_g1_flash_sac_config()``; the generic FlashSAC defaults continue
to match the upstream recipe.

For seed 0, ``4.5e-4`` sustained in 42.197 seconds, n-step 5 in 44.237
seconds, and target rate 0.015 did not sustain within the 15M-transition cap.
The standard non-G1 distributional continuous-control learning regression
also passed after the sweep.

An Nsight Systems trace of the selected 2,048-world, batch-2,048 cadence is at
``/tmp/flash_overlap_2048_b2048.nsys-rep``.  It measured 12.379 ms per cadence.
The two alternating rollout graph instances each averaged about 11.56 ms, while
the complete four-update learner graph averaged 5.55 ms and phase preparation
averaged 0.025 ms.  The learner is therefore hidden behind rollout at this
world count; reducing learner kernels alone cannot shorten the critical path.
Increasing the batch to 2,560 or 3,072 moved the learner onto the critical path,
raising cadence to 14.016 and 16.242 ms respectively.  Their extra replay
density would need to reduce updates-to-gate by at least 12 and 24 percent just
to break even.  The batch-4,096 quality run reduced updates-to-gate by only
about three percent, so these intermediate batches were not quality-run.

### Champion/challenger execution layouts

The retained bounded search uses one shared replay and identical setup-owned sampled
batches for both learners. In the production-shaped 2,048-world benchmark, the
normal overlapped P1 path measured 335.9k transitions/s. Two independent learner
streams measured 210.9k transitions/s, or 1.593x the P1 cost, while remaining
about eight percent faster end-to-end than the fused population backend on this
GPU. Once search converges, the controller returns to the preallocated P1 graph
and recovers full single-learner throughput. The independent-stream backend is
therefore retained for bounded, coarse-cadence search rather than rejected as a
continuous execution layout.

### Population training foundation

The population implementation stacks arbitrary compatible actors and critics and
owns all optimizer moments, AMP scaler/overflow state, online and target
parameters, scalar counters, and update workspaces at setup. Forward/backward
activation, reduction, and AMP mirror storage retains stable addresses across
CUDA graph replay. Per-member BatchNorm affine and running state remain logical
checkpoint views over contiguous population storage.

P2 fused updates preserve the scalar FlashSAC order: actor, deterministic
temperature, distributional critic, and parameter-only target EMA. Per-member
seeds, scaler decisions, overflow skips, optimizer state, and policy-frequency
cadence are deterministic. Repeated independent allocations produce bitwise
identical P2 state, while the scalar oracle bounds the expected FP16 batched-GEMM
layout difference.

At batch 2,048, the complete fused P2 micro-update measured 2.165 ms versus
2.298 ms for two scalar updates. The larger production overlap benchmark favored
two independent learner streams by about eight percent because concurrent exact
scalar graphs overlap better with rollout on this GPU. The fused implementation
remains useful as a deterministic population primitive; backend selection must
be based on full end-to-end cadence rather than the isolated update microbenchmark.

### Bounded automatic learning-rate search

The LR controller starts from reliable configured actor, critic, and temperature
rates and applies deterministic bounded log-space proposals. It first probes a
linked rate, then explores individual coordinates. Both members consume the
same sampled batch, so paired differences are not replay-sampling noise. Search
uses 10 percent of rollout worlds for the challenger and 90 percent for the
champion, with all storage allocated before capture.

A device-side per-world guard evaluates champion and challenger actions using
the same exploration seed. Challenger actions are used only when both policies
are finite and their RMS and maximum action differences stay within configured
bounds; otherwise the champion action is routed and the fallback is counted.
This protects shared replay from an immature challenger without adding host
synchronization to the steady-state captured path.

Paired policy evaluation runs in isolated held-out G1 environments with identical
commands, seeds, and reset state. The device objective multiplies upright/alive
tracking success by nonnegative command-aligned velocity normalized to the fixed
0.8 m/s command, so standing still scores exactly zero. Nonfinite state is always
unsafe. Early falls use relative paired safety, allowing equally immature
policies to remain comparable while rejecting a challenger whose fall rate is
worse than the champion beyond a small margin.

Promotion and best-state updates require repeated windows. Candidate evidence
tracks the actual superior member before promotion or reset, including its exact
optimizer, AMP, target, and rate state. A setup-owned frozen best snapshot
supports rollback after live regression. Bounded quality proofs finalize with
the repeatedly confirmed best, restore it into the active P1 learner, and
evaluate that restored state. A one-window live-policy finalization exists only
as an explicit diagnostic mode.

Two measurement pitfalls materially changed the apparent result. A benchmark
replay configuration with a roughly 2k-item warmup and 16k capacity failed even
for the known-good fixed-rate control; the production quality protocol requires
a 100,000-item warmup and 10,000,000-item capacity. An earlier shaped/Gaussian
score also rewarded standing. Neither setup is valid evidence for locomotion
learning or search quality.

Starting all three rates at 3e-4, the confirmed discovery run processed 18.022M
transitions. It promoted the temperature rate to 3.200226e-4 and subsequently
confirmed 3.413815e-4. The restored best achieved objective 0.872365, mean
command-aligned velocity 0.843517 m/s, and zero falls at 198.2k transitions/s.
This supports the narrow claim that bounded coordinate search discovered a
useful temperature-rate improvement while preserving learning from a safe
default. It does not establish a global optimum or show that this run discovered
the separately validated linked 6e-4 G1 recipe.

## Reproducible quality evidence

The corrected FP32 seed-0 result is stored in
`/tmp/g1_time_to_gate_20260817/full29_corrected_results.json`.  It first passed
at 8.595M transitions and 100.56 s of training wall time and sustained the gate
at 9.095M transitions and 106.53 s.  The same results file contains named phases
for rejected AMP experiments.  Evaluation overhead is recorded separately from
training wall time.

The same file contains the three accepted dynamic-scaler phases named
`full29_amp_dynamic_seed{0,1,2}_0.8`.  Each phase records every deterministic
evaluation and its checkpoint path.

Do not treat `/tmp` as permanent archival storage.  Promote the final comparison
configuration and concise results into this document when the AMP implementation
passes multiple seeds.
