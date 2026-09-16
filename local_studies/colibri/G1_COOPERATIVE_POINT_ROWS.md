# Local G1 cooperative point-row experiment

The fresh current-source profile attributes 3.028 ms per policy step to packed
contact-row preparation (216 calls over two policy steps;9.32% GPU kernel time).
The local factory distributes independent six-DoF projection/matrix rows across
eight lanes. Scalar sums, ordered wrench updates, forward body propagation,
inverse-mass accumulation, and physical settings remain unchanged. Root stores
have unique lane/DoF ownership; other stores use the leader. Subgroup barriers
publish predecessor responses. Dynamic-pair rows retain scalar fallback.

## First correctness gate

Session21669 passed the corrected20-frame trajectory byte gate and separate
eager-capture versus uninstrumented step21 byte gate. Two actual input snapshots
each passed15 variants against the scalar kernel, comparing all70 reachable
buffers. Source/assets unchanged. Timing disabled.

Actual inputs contain49 generalized DOFs/44 joints, page0, seven points/21 rows
and eight points/24 rows, prepare=True, no second dynamic endpoint. Clamped
counts above24 did not test the response-layout transition above32 rows.

Artifacts: /tmp/g1_cooperative_point_rows.json,
.capture0.npz, .capture1.npz, and
/tmp/g1_cooperative_point_rows/candidate.py.

## Additional gates prepared

Synthetic32/33/96-row pages repeat real three-axis geometry into distinct
contact slots. They test storage boundaries, not physical manifold quality.
Pair/stale-pair/stale-body cases cover fallback and ownership transitions.

V1 PTX declares a96-byte local array depot caused by dynamic vec6 lane indexing.
V2 replaces projected/RHS/matrix lane temporaries with scalars before shuffling
into the same vector components. Arithmetic is preserved by construction;
CUDA byte gates and timing remain required. V1 is retained unchanged.

Use local_studies.colibri.check_g1_cooperative_point_rows with --output
/tmp/g1_cooperative_point_rows_boundaries.json for v1, and
--scalar-lane-values --output /tmp/g1_cooperative_point_rows_scalar_lanes.json
for v2. Both default to timing-disabled. Only after both gates pass, use
--timing-repeats during an isolated GPU window. Event brackets exclude buffer
restore copies; every frozen input/scratch buffer is restored before each
replay. Kernel timings do not establish controller speed.

No production edit, live candidate rollout, or speedup claim.


## Terminal expanded gates and timing: do not promote

Both versions passed42 cases (two actual captures times21 variants), including
explicit32/33/96-row storage boundaries and pair/stale-owner transitions.
All70 buffers, trajectory gates, and source/asset hashes remained exact.

After these gates, isolated frozen timing used CUDA graphs containing complete
buffer resets followed by external start-event, kernel, and end-event nodes.
Event brackets exclude both resets and Python enqueue gaps. Each kernel had ten
warmup replays and100 measured replays. Actual launch attributes came from
cuFuncGetAttribute on the loaded block-size variant.

| Actual rows | Scalar median us | V1 median us | V2 median us |
| --- | ---: | ---: | ---: |
| 21 | 28.672 | 30.720 | 30.720 |
| 24 | 30.720 | 32.768 | 32.768 |

| Kernel | Registers/thread | Local bytes/thread | Static shared bytes |
| --- | ---: | ---: | ---: |
| Scalar | 64 | 0 | 192 |
| V1 indexed vector lanes | 64 | 96 | 512 |
| V2 scalar lane values | 56 | 0 | 512 |

V2 removes the depot and lowers register use, but does not beat the original.
Timing is quantized at this duration; the approximately7% slower medians are
not a full-controller regression claim. Means also favor scalar in both
captures. More lanes and synchronization may offset parallel root arithmetic,
but this experiment alone does not isolate those costs. Do not integrate this
prototype or claim a speedup. No full live candidate hook was installed.

Artifacts:
- /tmp/g1_cooperative_point_rows_boundaries.json
- /tmp/g1_cooperative_point_rows_scalar_lanes.json
- /tmp/g1_cooperative_point_rows_v1_timing.json
- /tmp/g1_cooperative_point_rows_scalar_timing.json
- /tmp/g1_cooperative_point_rows_timing_summary.json

An initial timing attempt stopped at a resource-query lookup error after its
correctness checks; it produced no accepted timing result. The corrected lookup
uses the actual loaded launch-block module. Terminal accepted sessions84428
and90853 passed. GPU released to root after90853; production remains unchanged.
