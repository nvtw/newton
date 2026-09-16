# Cached response: measured rejection and next target

The first serial-native response candidate is **not accepted**. Its 600-frame
run measured 20.4978 FPS against 20.4885 FPS for the corrected native baseline.
Base creep over seconds 5–10 increased from 2.25892 to 2.96354 micrometers/s.
Neither the small timing difference nor the isolated impulse-response parity
establishes an improvement in the simulation.

## Nsight Systems evidence

Both profiles capture 30 warmed frames, 60 collision updates and 1,800 substeps,
with CUDA graph node tracing. Times below sum kernel execution times and divide
by 30; they are not independent end-to-end frame timers.

| Operation | Native GPU ms/frame | Candidate GPU ms/frame |
| --- | ---: | ---: |
| Contact iteration | 27.468 | 26.214 |
| Contact mobility refresh | 7.342 | 8.326 |
| Contact warm start | 3.155 | 1.420 |
| Extra response cache construction | — | 1.294 |

The candidate's contact iteration still occupies 55.9% of GPU kernel time;
mobility refresh occupies 17.8%. Removing the cooperative per-point scatter
did not remove the dominant cost. Warm-start improvement is offset by the new
cache and slower refresh. Collision detection is not the dominant cost in
this two-body configuration; this does not establish its share for 37 bodies.

The profiled candidate's `q_history`, `qd_history` and `history_times` are
byte-identical to the unprofiled 600-frame candidate prefix. Its valid-cache
flag is set, and comparison-mode duplicate work is disabled.

Artifacts:

- `/tmp/colibri_cached_serial_warm_profile.nsys-rep`
- `/tmp/colibri_cached_serial_warm_profile_stats.csv`
- `/tmp/colibri_cached_serial_profile65.npz`
- `/tmp/colibri_cached_serial600.json`
- `/tmp/colibri_no_skip_warm_profile_stats.csv`

## Interpretation and next discriminating experiments

The remaining iteration executes a serial point sweep in one active thread,
with repeated global body/joint state accesses. Register-resident component
state is a candidate for eliminating that work while preserving the actual
point equations. This is a hypothesis requiring timing and physical gates.

An offline sm120 assembly of the candidate PTX reports no local-memory spills;
this is evidence against a spill explanation, not a driver-JIT resource audit.
The final captured state's 67 tangent blocks select the FP32 metric branch,
including its 14 loaded contacts. This snapshot cannot establish branch
frequencies throughout the run, but does not support blaming the conditional
FP64 friction fallback for the measured whole-kernel cost.

A separate experiment revisits full-coordinate colored joint/contact solving.
Its timing counts only after original joint/drive residuals, friction,
conservation and base motion pass. No result here justifies relaxing those
requirements, reducing contact fidelity or using an inaccurate fast baseline.
