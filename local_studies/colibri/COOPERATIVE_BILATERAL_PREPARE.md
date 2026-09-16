# Cooperative bilateral preparation

The CUDA optimization is integrated, with a scalar CPU fallback. The paired
short-run result is 89.99 to 98.70 FPS with byte-identical Colibri histories;
preparation, CPU smoke, Kapla, and G1 checks pass. The 300-second integration
gate passed at 98.6414 FPS with all ten saved arrays byte-identical. See the later sections for evidence and the regression plan
for remaining coverage.

## Prototype design and investigation history

The candidate uses eight CUDA lanes per joint. Each active row computes the existing FP32 body responses and ordered FP64 dot products. Responses are broadcast from registers, avoiding store/read visibility dependencies. LDL pivots advance in original row order; independent later rows perform their own increasing-index subtraction and division. All eight lanes participate in subgroup shuffles, including unused lanes. The leader writes the complete factor structs once. Zero-row joints preserve stale response/factor fields and only clear validity. This initial prototype did not modify production.

The proposed test harness covers existing 84 active-row/geometry/mass-ratio/copy-count cases at 32- and 64-thread blocks, current captured full Colibri inputs, seven ragged joints with counts zero through six, nonzero output sentinels, singular pivots, and partial final CUDA blocks.

## First CUDA result: exactness gate failed

Session 63853 compiled the kernel and failed during the first block-size pass. Count 4, random wrenches, mass ratio 400, split copies: diagonal entry 3 is 7.182241436714367 in the scalar reference and 7.182241436714364 in the candidate (difference 3.55271368e-15). Responses and the lower factor passed prior assertions in that case. The remaining cases have not run; there is no performance result. Log: `/tmp/cooperative_bilateral_prepare_checks.log`.

This is not yet localized to matrix-dot contraction versus LDL arithmetic. Identical source operation ordering does not prove identical compiler contraction when values are distributed across lanes. No tolerance was relaxed. A future tiny diagnostic should capture the first differing matrix/pivot intermediates before selecting a code change. GPU released to the queued adapter tests.

## Instrumented diagnostic: observer effect, no valid first-difference localization

Session 6591 completed. Capturing lower-triangle matrix entries and pivot values before/after each subtraction makes all instrumented scalar/cooperative outputs byte-identical. The scalar instrumented kernel also preserves its original outputs. The cooperative instrumented kernel changes its original diagonal output, exactly the field that failed previously. Therefore the instrumentation-equivalence gate rejects this as a localization result: debug stores alter cooperative code generation/evaluation. It supports investigating compiler contraction but does not yet identify the precise original differing operation. There is no evidence of a response visibility race; responses are communicated in registers.

Artifacts: `/tmp/cooperative_prepare_first_failure.npz`, `/tmp/cooperative_prepare_first_difference.json/.npz`, and `.log`. Production unchanged; no performance measurement or tolerance relaxation. GPU released for the floor-distance long run.


## Resolved contraction difference and canonical integration

The initial exactness rejection is resolved without changing tolerances. High-precision reconstruction of the captured Gram matrix reproduces the scalar pivot with a rounded FP64 square followed by fused subtraction, and reproduces the original cooperative pivot with separate multiplication/subtraction. Generated code inspection supports the contraction difference. An explicit FP64 helper now uses `__dmul_rn(lower, lower)` and `__fma_rn(-squared, diagonal, pivot)`; FP32 response calculations are unchanged. No global compiler flags were changed.

All 84 frozen cases pass byte equality at both 32- and 64-thread blocks, including signed zeros. The independent Gram/LDL diagnostic reports identical scalar and cooperative maximum relative backward error of 2.501240851646906e-16. Current 38-joint inputs, ragged counts 0 through 6, singular pivots, stale output sentinels, partial blocks, and CUDA graph capture also pass. Artifacts: `/tmp/cooperative_prepare_all_explicit_cases.json`, `/tmp/cooperative_bilateral_prepare_explicit_checks.log`.

Canonical integration uses eight CUDA lanes per joint with 32-thread blocks and retains the scalar CPU path. Six production preparation and block-policy tests pass, including finite-drive momentum/property-transition tests. No configuration, temporal count, collision policy, or iteration count changed. Production files are `constraints/bilateral_joint.py`, `articulations/block_joint_system.py`, and `tests/test_bilateral_preparation.py`; the existing bilateral optimization changelog fragment records the change.

## Isolated performance and trajectory gates

Frozen current-input timing (128-launch graphs, 12 forward/reverse interleaved samples): scalar 25.75350 microseconds, cooperative block32 11.08350 microseconds, block64 11.09613 microseconds. Artifact: `/tmp/cooperative_prepare_frozen_timing.json`.

Fresh integrated paired public Colibri runs use 30 temporal substeps per 120 Hz collision refresh, one solver iteration, the current default final relaxation, group4, geometric admission, and no search floor. After 30 warmup frames, 300 measured frames give scalar 11.112763 ms (89.9866 FPS) and cooperative 10.131364 ms (98.7034 FPS): 8.83% less frame time. Both runs preserve all ten reference arrays byte-for-byte, including the complete 330-frame pose/velocity histories; source and assets remain unchanged during each run.

Exact paired artifacts (each has matching `.npz`, `.source.json`, and `.log`):

- `/tmp/colibri_prepare_integrated_scalar330.json`
- `/tmp/colibri_prepare_integrated_cooperative330.json`

Reference: `/tmp/colibri_spatial_priority_validation_velocity1_first.npz`. The scalar control is a process-local restoration of the prior scalar preparation launch on otherwise identical current source.

Cross-scene gates completed with exit 0: Kapla 180 measured frames preserves all six saved arrays against `/tmp/kapla_rebased_packed_fixed.npz`; learned G1 20 policy frames preserves body_q/body_qd bytes against `/tmp/g1_cache_only_candidate.npz`. Artifacts: `/tmp/colibri_cooperative_prepare_kapla.json` and `/tmp/colibri_cooperative_prepare_g1.json/.npz`. These establish short regression equivalence, not new long-run stability claims or matched cross-scene speedups.

GPU and canonical source were released/frozen for the parent integrated 300-second Colibri reference comparison. That longer gate subsequently passed (session 59043, exit 0):
18,000 frames / 300 simulated seconds, 98.6414 FPS, 10.137728 ms mean,
11.220050 ms p95. All ten arrays, including every q/qd/time history sample,
match `/tmp/colibri_spatial_priority_velocity1_long18000.npz` byte for byte;
source/assets and the reference hash remained unchanged. Peak depth is
0.762952 mm, anchor error 0.794343 mm, and axis error 0.0289467 rad.
Artifacts: `/tmp/colibri_cooperative_prepare_velocity1_long18000.{json,npz,source.json,log}`.
The clean runner's original `scope` text incorrectly described increased
velocity work even for count 1; the recorded count and extra-sweep fields
confirm default work, and the runner now labels that case correctly. The
original result artifact is preserved.

## Real CPU fallback smoke

The actual canonical scalar preparation kernel compiled and executed on CPU after integration (not a mocked dispatch). Current 38-joint, ragged seven-joint counts 0 through 6, and invalid-pivot fixtures pass finite-output, expected validity, and count-zero stale-field checks. Independent Gram/LDL reconstruction maximum relative errors are 2.09264e-16 and 6.20507e-17 for current/ragged valid fixtures. Artifacts: `/tmp/cooperative_prepare_cpu_smoke.json/.log`; local harness `check_cooperative_prepare_cpu.py`. This checks preparation and module compilation, not an entire CPU simulation. No GPU kernel launched.
