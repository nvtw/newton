# Identical-input point-kernel attribution

`profile_point_replay.py` is a diagnostic runner, not a solver change. Launch after the corrected owned callbacks, normalized FP32 material history, and direct no-skip overlay:

```bash
COLIBRI_DIFFERENTIAL_REFERENCE=1 COLIBRI_DIFFERENTIAL_NORMALIZED=1 \
COLIBRI_STAGE_RUNNER=local_studies.colibri.profile_point_replay \
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python \
-m local_studies.colibri.direct_relax_no_skip \
--native-path direct --body-count 2 --frames 331 --substeps 30 --save-history \
--output /tmp/colibri_point_replay331.json
```

The physical trajectory uses the serial-global cached response control. At frame 331 only, disable the simulation graph and clone the complete inputs to the first biased and first relaxation point-kernel launches. Physical integration continues with the original kernel. No diagnostic output is scattered into the simulation.

For each frozen launch, execute the full original kernel and save every output array. A recording variant saves each point's actual three row velocities, metric-friction result, and post-response body velocities/drive impulse. Its complete array outputs must be byte-identical to the full kernel. Independent variants then replace Jv, metric projection, or response application with these recorded exact values. Every array must remain byte-identical, so downstream branch decisions and contact interactions remain those of the full solve. An additional original cooperative-native kernel control uses the same physical inputs; its physical outputs must match, while its private response scratch may differ.

All arrays reset before every timed replay. CUDA events time one completed launch; reset copies, synchronization before start, collision, factorization, and cache construction are excluded. Five warm launches precede 40 samples per variant. Results are counterfactual latency reductions, **not additive exclusive component times**: generated code/register allocation, replacement trace loads, and scheduling can interact. This diagnostic does not infer performance from static PTX counts, extrapolate two-body results to the full mechanism, or establish trajectory stability.

Actual SDK work differs: the parent's matched PhysX contact report exposes six normal points and two friction anchors, while the PhoenX capture has 67 pointwise friction contacts (52 base,15 zero-load Frame candidates). The SDK contact report may omit untouching speculative pairs and therefore does not establish the total internal GPU row count. Preserve that scope distinction when comparing total times.

## First capture and timing-method control

`/tmp/colibri_point_replay331.point_replay.json` passes all frozen full/record/substitution byte gates, including same-input native physical outputs. Both phases visit 67 points. The biased phase executes 67 metric calls; the relaxation phase executes **zero** (all frozen rows speculative).

The first timer enclosed a Python `wp.launch`, so its event interval can include device idle during host argument packing. Its latency numbers are preliminary and must not be presented as exact GPU component costs. The runner now captures each one-kernel replay graph before timing and launches that graph inside the event interval. This removes per-replay Python struct packing; an isolated repeat is pending the GPU queue.

Offline `ptxas -arch=sm_120 -v` on the generated variant PTX reports zero spills/stack for every variant: record128 registers, Jv-substitution102, metric-substitution96, apply-substitution106, all-substitution48. This is offline compilation, not inspection of the driver-JIT binary. `/tmp/colibri_point_replay_ptxas.log` preserves the report. The metric-removal latency difference even when no metric call executes is a warning against interpreting counterfactual differences as additive runtime routine costs.

## Completed graph-only replay

`/tmp/colibri_point_graph_replay331.point_replay.json` passes every byte gate. Its full 331-frame q/qd/time histories are also byte-identical to `/tmp/colibri_cached_serial600.npz`, including the eager capture frame.

Median microseconds, 40 measured one-kernel graph replays:

| Variant | Biased | Relaxation |
|---|---:|---:|
| Full serial candidate | 382.640 | 331.088 |
| Native cooperative, same physical inputs | 394.352 | 333.072 |
| Substitute all three Jv values | 80.848 | 36.272 |
| Substitute metric result | 154.976 | 132.832 |
| Substitute impulse-response writeback | 380.352 | 329.440 |
| Substitute all three sections | 54.656 | 34.720 |

The row-velocity dependency is the dominant measured target: replacing its outputs saves 302/295 microseconds. Replacing the response saves only approximately two microseconds in these snapshots. This is narrower than a general claim about arbitrary trees.

**Metric substitution also deletes its prerequisites.** Both tangent velocities feed only the tangent RHS and metric projection. Substituting the final metric result makes these two generic Jv evaluations dead, allowing compiler elimination. The relaxation metric saving (198 microseconds) is approximately two-thirds of the three-Jv saving (295 microseconds), despite zero runtime metric calls. It must not be described as friction-projection execution time. A follow-up `full_keep_jv`/`metric_keep_jv` pair writes computed Jv into identical observable sinks to preserve those computations and isolate projection cost; implemented, validation pending.

The implementation problem is more specific than repeated impulse tree application: every point evaluates three generic joint-coordinate row velocities, including FP64 force/torque construction, parent traversal and repeated body reads. Those evaluations still happen for the frozen speculative relaxation rows even though their impulses remain unchanged. Production changes must preserve cancellation protection and actual physical output gates; these trace substitutions are diagnostics only.

## Projection cost isolated with live Jv sinks

`/tmp/colibri_point_sink_replay331.point_replay.json` passes all 16 phase/variant byte gates and the full 331-frame trajectory-prefix byte gate. With identical observable Jv sinks, biased full is381.984µs and metric-substitution356.176µs: **25.808µs** counterfactual projection saving. Relaxation is333.136 versus335.024µs, consistent with no executed projection and small codegen differences. Thus the earlier approximately200µs metric-removal saving was principally deletion of its two tangent-Jv prerequisites. Generic three-direction Jv substitution saves302.192µs; response substitution saves2.352µs in the biased snapshot.

Complete frozen arrays, scalar/null values and pointer aliases are saved in `/tmp/colibri_point_sink_replay331.point_inputs_{biased,relax}.{npz,json}`. `COLIBRI_REPLAY_FROZEN_PREFIX=/tmp/colibri_point_sink_replay331` skips trajectory generation and restores those typed inputs. Per-point traces are saved as `.point_trace_{biased,relax}.npz`: rows0:3 nativeJv,3:6 metricresult,6:12 postpointroot(v,w),12:18 postpointchild(v,w),18:20 driveaccumulation,20 normalimpulse. Before-point body state is the preceding visited point's poststate; the first uses the initial archive.

The first pure-load command completed all replay gates and exports (`/tmp/colibri_point_loaded_replay.*`) but its outer `native_conditioned_two_body` wrapper subsequently exited1 on its correct 'Configured constructor was not invoked' guard, because pure replay deliberately builds no world. Do not call that process a passing live-run/provenance test. The replay-only module results and trace exports remain successful; use a dedicated overlay-only entry point for a clean future replay process.
