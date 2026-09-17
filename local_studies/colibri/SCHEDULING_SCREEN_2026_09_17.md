# Single-world scheduling screen, 2026-09-17

The 9.4 ms/frame full-minute physics goal remains open. These are frozen
kernel screens, not end-to-end speedups. The scene uses the original contact
policy (before the adjacent-tail filtering change), one world, 60 warmup
frames, and unchanged solver counts on the RTX PRO 6000 Blackwell.

Each sample restores 400 state arrays, then times one biased sweep using GPU
events. Restore copies are outside the timed interval. Discard 20 warmups
and measure 200 samples. Check velocity, angular velocity, contact impulses,
and joint impulses bitwise against the original sweep.

| Scheduling | Mean sweep time (microseconds) |
| --- | ---: |
| Control, component experiment | 137.196 |
| Independent connected groups | 125.418 |
| Control, block-size experiment | 136.906 |
| 32 threads | 301.757 |
| 64 threads | 196.573 |
| 128 threads | 144.799 |

All candidates passed the frozen bitwise check. Smaller blocks preserve
row coverage by adjusting contact and joint strides; they are rejected.
The connected-group screen partitions each color slab into disjoint body
components and preserves row order within each component. Its 8.6% kernel
saving excludes component construction, computed on the CPU for this
upper-bound experiment. It does not establish a full-trajectory gain and
is not retained in production. A GPU scheduling implementation would need
to justify its setup cost and complexity first.

Raw results: `/tmp/colibri_component_frozen.log` and
`/tmp/colibri_blocks_frozen.log`. Experimental drivers and kernel copies:
`/tmp/colibri_profile_components.py`, `/tmp/colibri_component_sweep.py`,
`/tmp/colibri_profile_blocks.py`, `/tmp/colibri_sweep_block{32,64,128}.py`.
Both drivers run against `/tmp/newton-phoenx-optimization` via `PYTHONPATH`,
using Newton's venv Python and `--output /tmp/<report>`.

A suspected frozen-benchmark kernel lookup issue was also checked. Both
omitting and explicitly passing the default world count intercepted the
sweep successfully in fresh processes. Both passed the restored-state
bitwise check. The proposed tooling change was unnecessary and removed.
Results: `/tmp/colibri_factory_before.log` and
`/tmp/colibri_factory_after.log`.
