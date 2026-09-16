# Matched two-body runtime profiling

Both Nsight Systems traces capture 30 warmed frames at two collision steps per
frame and 30 position iterations/substeps per collision step. GPU kernel time
is the sum of measured kernel durations; it is not end-to-end frame latency.
SDK profile initial, post-warmup and final states and six physical-property
arrays pass byte-equality checks against the existing PhysX accuracy reference.

| Measurement | PhysX SDK | Corrected native PhoenX |
| --- | ---: | ---: |
| Unprofiled completed frame | 4.70–4.97 ms | 48.81 ms |
| Sum of profiled GPU kernel durations/frame | 1.182 ms | 47.479 ms |
| Kernel executions/frame | 488 | 3,116.23 |
| Principal contact/joint iteration kernel | 5.079 us, whole island | 443.0 us, conditioned contact iteration |

The last row compares different units of work, not identical algorithms.
PhysX also executes `solveStaticBlockTGS` at 3.876 us/call. Whole-island plus
static solve kernel time totals 0.545 ms/frame. Its whole-island kernel executes
1,800 times and static kernel 1,860 times in the captured range. PhoenX executes
its conditioned contact kernel 1,860 times, taking 27.468 ms/frame; contact
mobility refresh adds 7.342 ms and warm starting 3.155 ms.

The cached serial PhoenX candidate still spends 46.904 ms/frame in GPU kernels,
with 3,358.23 executions/frame. Its principal contact kernel takes 422.811 us.
This candidate has no accepted speed improvement and worsens measured creep.

## Actual contact counts, separately measured

An optional diagnostic USD layer adds only `PhysxContactReportAPI` and zero
report threshold on the two dynamic bodies. At frame 600, the SDK reports:

- One contact pair.
- Six normal contact points, five with nonzero reported impulse.
- Two friction anchors.

The diagnostic's initial, post-warmup and final q/qd and physical-property
arrays remain byte-identical to the original reference. Its timing is not
substituted for the uninstrumented benchmark. The saved PhoenX candidate state
has 67 tangent/contact blocks, 14 carrying normal load. These are settled
snapshots, not a count histogram throughout the profile window; the trajectories
and discrete contact/drive laws differ. The count discrepancy is a measured
workload difference, not proof that arbitrary contact removal is safe.

## Source interpretation

The actual runtime invokes `solveWholeIslandTGS` and `solveStaticBlockTGS`.
In the inspected PhysX checkout, `solverMultiBlockTGS.cu:1688` loads island
velocity/delta state into shared memory, walks constraint partitions, and
dispatches joint/contact blocks before writing state back. This is stronger
evidence for the active strategy than assuming which named solver path runs.
The installed SDK binary is not claimed to use the exact checkout revision.

Contact-count disparity, repeated conditioned response work, and repeated
kernel/state dispatch are now explicit hypotheses with measured workload
evidence. Their individual causal timing contributions require identical-input
counterfactuals; do not add a guessed factor for each or claim the full gap is
already explained. Those counterfactuals and the D6-style full-coordinate
implementation are the next tests. The profiles do not justify weakening
accuracy, conservation, drive equations or friction capacity.

## Artifacts

- `/tmp/colibri_physx_two_body_warm_profile.nsys-rep`
- `/tmp/colibri_physx_two_body_warm_profile_stats.csv`
- `/tmp/colibri_physx_two_body_profile30.json`
- `/tmp/colibri_physx_two_body_contacts600.json`
- `/tmp/colibri_physx_two_body_contacts600.contacts.json`
- `/tmp/colibri_no_skip_warm_profile_stats.csv`
- `/tmp/colibri_cached_serial_warm_profile_stats.csv`

Runner: `benchmark_physx_two_body_timing.py --profile` captures only its measured
steps; `--contact-report` enables the separate reporting diagnostic.
