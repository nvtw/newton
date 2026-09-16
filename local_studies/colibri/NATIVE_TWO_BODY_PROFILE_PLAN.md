# Matched native two-body profiling plan

Local-only instrumentation: `profile_native_conditioned_step.py`. No solver parameters or kernels are replaced.

## Capture

Run only after the accepted friction configuration is known and the GPU is exclusively available. Run direct and reduced sequentially, with the same environment and source/asset hashes.

For each `MODE` (`direct`, then `reduced`):

```bash
COLIBRI_STAGE_RUNNER=local_studies.colibri.profile_native_conditioned_step \
COLIBRI_PROFILE_FRAME=331 \
COLIBRI_PROFILE_METADATA=/tmp/colibri_native_MODE_profile.capture.json \
nsys profile --trace=cuda,nvtx --sample=none --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=true --output=/tmp/colibri_native_MODE_profile \
  uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -u \
  -m local_studies.colibri.native_conditioned_two_body \
  --native-path MODE --body-count 2 --frames 332 --warmup 330 \
  --substeps 30 --iterations 1 --save-history \
  --output /tmp/colibri_native_MODE_profile.json
```

Replace MODE literally in all paths and `--native-path`. If the FP32 differential-reference candidate is accepted, set `COLIBRI_DIFFERENTIAL_REFERENCE=1` identically for both runs. Otherwise retain the accepted earlier configuration. Do not mix old/new friction controls or normalized/raw variants.

The range covers **only frame331 example.step**: two120Hz collision intervals, each30substeps, one iteration, SOR1. It includes collision and actual captured CUDA graph nodes, but excludes post-step contact/joint test queries. Native wrappers retain source/assets fingerprints, ownership checks, geometry/joint bounds and saved trajectories. Nsight-affected harness FPS must not be reported as isolated timing.

## Correctness and attribution

1. Run an unprofiled332-frame control for each path with identical settings. Compare that path's q, qd, initial_q, full q_history/qd_history/history_times bytewise with its profiled result. Direct and reduced trajectories need not equal each other.
2. Use `nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv` and export SQLite for exact range durations and launch gaps.
3. Attribute actual names to collision/broadphase/narrowphase/reduction/matching, body/contact preparation, joint factorization/response preparation, joint-conditioned contact sweeps, integration, and transfers.
4. Count graph launches and constituent kernel nodes. Report GPU kernel sum separately from range elapsed and CPU CUDA API time: overlapping durations must not be added. Attribute unaccounted idle gaps explicitly instead of calling them arithmetic.
5. Record active contact/owned row sizes for the sampled frame. Different trajectory contact counts can explain work differences; this is not an equal-operator microbenchmark.
6. Only if necessary, profile a second matched frame to resolve an observed attribution ambiguity.

Prior20.9/47.5FPS numbers motivate this capture; different run lengths/settling states prevent treating their ratio as an isolated architectural speedup. No GPU profile has been run by this preparation.
