# Colibri performance target: measured PhysX comparison

The matched two-body SDK timing measured PhysX at 201.36 and 212.63 FPS
(4.966 and 4.703 ms per frame) in two independent processes. The current
PhoenX corrected native path with the joint-relaxation skip disabled measured
20.49 FPS (about 48.8 ms per frame). The throughput gap is roughly 9.8–10.4x.

Both use two 1/120-second collision steps per reported frame, with 30 PhoenX
substeps or 30 PhysX position iterations plus one velocity iteration per step.
The PhysX timer excludes setup, warmup, rendering and state readback. Its state
and property gates match the existing awake, plane-1-mm two-body reference.
Contact/drive discretizations differ; timing agreement is not an accuracy claim.
The PhoenX path still creeps, so it is not a completed stable configuration.

Artifacts: `/tmp/colibri_physx_two_body_isolated300.*`,
`/tmp/colibri_physx_two_body_isolated300_repeat.*`, and
`/tmp/colibri_native_direct_no_skip600.*`. Both SDK processes have separate
provenance records; their frame-time variation is approximately5.3%.

## Explicit all-awake full-scene target

The new37-body SDK companion measured **119.6676FPS /8.356482ms per frame**
(P95 8.992219ms),300 measured frames after30warmup. All37 bodies author
sleepThreshold0; stabilization is explicitlyfalse; position iterations are
pinned min/max30 and velocity iterations min/max1, at120Hz. The composed-USD
diff allows only those numerical flags. All other attributes, relationships,
APIs, physical properties, collision geometry and filters remain unchanged.

All six saved physical-property arrays and body paths are byte-identical to
the historical full-scene control. Initial/final pose and velocity arrays
also match over this330-frame run. This is one isolated timing sample, not
a claim that disabling sleeping improves performance or that the full scene
has passed stationary-base/conservation acceptance.

Artifacts: `/tmp/colibri_physx_full37_awake_isolated300.{json,npz,log,provenance.json}`
and `/tmp/colibri_physx_full37_awake.fixture.json`.
The earlier refreshed historical configuration measured105.7821FPS, but only
5of37 bodies explicitly disabled sleeping; it is NOT the all-awake target.
That report is `/tmp/colibri_physx_full37_isolated300_refresh.*`.

## Warmed PhoenX GPU profile

Thirty frames, sixty collision steps, 1,800 substeps:

| Operation | GPU ms/frame | Share |
| --- | ---: | ---: |
| Joint-conditioned contact iteration | 27.47 | 57.9% |
| Contact mobility refresh | 7.34 | 15.5% |
| Contact warm start | 3.16 | 6.6% |
| All GPU kernels | 47.48 | 100% |

Those three operations consume 80% of GPU time. Collision detection is not the
primary bottleneck in this configuration. Even eliminating these operations
entirely leaves roughly 9.5 ms/frame of GPU work, above the whole PhysX step.
Removing observer launches showed no meaningful speedup and is not a priority.

The profile uses CUDA graph node tracing over warmed steps. The entire profiled
65-frame pose/velocity/time prefix is byte-identical to the unprofiled native
600-frame run. Artifacts: `/tmp/colibri_no_skip_warm_profile*`.

## Next architectural experiment

Replace repeated per-point joint-tree response traversals with a cached coupled
response for a small connected component. Aggregate warm-start wrenches, iterate
the original point contact laws against the cached response, and scatter the
resulting body/drive update once. Retain the original soft-normal, friction and
drive equations, common force points, SOR=1 and all physical acceptance tests.

This is a replacement for dominant work, not an additional final correction.
The final-only correction was rejected as a creep fix: 2.69 micrometers/second
versus 2.26 for the no-skip baseline, with 287/1,200 proposals accepted. Its
small final velocity did not undo displacement integrated during biased steps.

The cached-response experiment must pass physical and convergence regressions
before its FPS can count toward the target. A valid full-scene PhoenX/PhysX
speed ratio remains unavailable until full-scene quality is established.


## Source-backed architectural distinction

Inspected PhysX checkout: `/home/twidmer/Documents/git/PhysX`,
commit `ed6e5ca2474c9c80ad4f4826591b88476779c6ef`. The pinned SDK binary
is not claimed to be built from this exact revision.

For the rigid-body GPU TGS path, `solverBlockTGS.cuh:152–498`
(`solveContactBlockTGS`) loads both endpoints' linear/angular velocities into
local variables, iterates the contact rows using precomputed inverse-mass and
angular response terms, and updates these local velocities immediately.
Normal linear impulses are accumulated across the manifold before the linear
endpoint update (around319–327); angular updates happen perpoint. Friction
updates the same two endpoints (around431–440), and velocities are returned
once at494–497. Optional slab response factors are explicit. These are direct
endpoint rigid-body responses, NOT exact elimination of all connected joints.

Joint rows are solved separately in `solve1DBlockTGS` at744 onward.
`solverMultiBlockTGS.cu:1745–1831` walks constraint partitions, dispatching
contacts or joints with synchronization between partitions. There is no
mechanism-wide exact joint solve nested inside each rigid contact row on this
path. This source observation does not prove which whole-island scheduling
branch the measured SDK chose, nor cover the separate articulation path.

In contrast, PhoenX `maximal_contact_gs.py` evaluates row velocities through
joint-coordinate branches, builds each point impulse, then calls
`_apply_accumulated_impulse` after each active point (around528–539).
That helper calls `apply_maximal_contact_impulse_thread`, updates constrained
body velocities and the compliant-drive accumulated reaction, and synchronizes.
The stronger conditioned response is real work, not merely an inefficient
version of the PhysX local endpoint update.

The actionable lesson is to amortize that response while retaining its chosen
equations: cache the small component's full physical/compliant response for
the current geometry, aggregate warm-start wrenches, execute original contact
laws against it, then scatter body and drive reactions together. This motivates
Kepler's current prototype. It does NOT justify dropping joint response,
point contacts or torque capacity, freezing changing geometry, copying PhysX
friction compression, or predicting speedup before physical and timing gates.
