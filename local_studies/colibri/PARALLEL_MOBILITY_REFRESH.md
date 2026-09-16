# Exact parallel native mobility refresh

`parallel_mobility_refresh.py` changes only contact ownership within the existing64-lane component launch:

```diff
-    if lane != wp.int32(0):
-        return
...
-        for offset in range(count):
+        for offset in range(lane, count, wp.int32(_TREE_WIDTH)):
```

Every contact still calls the identical `_write_exact_contact_mobility` Warp function. The six conditioned mobility calculations, FP32 order, internal/common-ancestor cancellation path and output layout are unchanged. Factor and geometry arrays are read-only. Different lanes write different contact slots. There is no component-body-count specialization or new physical approximation.

## Correctness

Both CPU and CUDA pass11 cases:

- actual frozen biased and relaxation inputs from frame331;
- external contact column counts0,1,63,64,65,129;
- internal-contact helper counts1,65,129.

Every output byte equals the serial native result; all input factor/geometry bytes remain unchanged. Untouched output slots retain a sentinel, and the final point of every nonempty synthetic column is written. The real67-point capture consists of52+15 points in separate columns, so the synthetic65/129 cases are necessary to test more than one stride in a column.

CPU uses an isolated verbatim serial-source kernel because compiling the entire native GS module on CPU includes unrelated CUDA-only `__syncthreads`. CUDA compares against the actual native kernel. A strict source gate verifies that both refresh and helper source are identical to the corrected owned-callback staged module. A deliberately wrong control retains the lane0-only return while applying the strided loop; the parity gate fails on the first actual snapshot.

Artifacts:

- `/tmp/colibri_parallel_refresh_cpu.json` and `.log`:11PASS.
- `/tmp/colibri_parallel_refresh_cuda.json` and `.log`:11PASS.
- `/tmp/colibri_parallel_refresh_negative.log`:expectedFAIL, missed-lane output detected.

## Isolated GPU timing

One-kernel CUDA graph, five warm launches plus40 samples, output reset and synchronization outside event interval:

| Actual snapshot | Native serial | Parallel | Ratio |
|---|---:|---:|---:|
| Biased |113.536µs|5.824µs|19.49×|
| Relaxation |113.600µs|7.056µs|16.10×|

This measures refresh alone. Approximately107µs saved per refresh would be approximately6.4ms/frame at60 refreshes, but that is an extrapolation, **not a measured live speedup**. No stability or convergence claim follows from the timing.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python \
-m local_studies.colibri.parallel_mobility_refresh --device cpu \
--output /tmp/colibri_parallel_refresh_cpu.json

uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python \
-m local_studies.colibri.parallel_mobility_refresh --device cuda:0 \
--output /tmp/colibri_parallel_refresh_cuda.json
```

Default input prefix is `/tmp/colibri_point_sink_replay331`, whose NPZ/JSON preserve typed arrays, scalar values and aliases. No simulation reset/warmup or geometry approximation is needed for these frozen gates.

## Integration gate

Canonical code is unchanged by this local experiment. A production change needs actual live output parity and whole-step timing. The corrected native benchmark imports a staged owned-callback module via `OwnedFinder`; editing canonical source alone does not update that executed callback. Regenerate the stage/manifest or explicitly bind the new kernel, and verify actual object provenance before making a live performance claim. The generated source path in the CUDA JSON preserves both the old serial-source clone and parallel variant for later comparison.
