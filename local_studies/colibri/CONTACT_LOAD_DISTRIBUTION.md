# Corrected native final contact load distribution

CPU-only analysis of `/tmp/colibri_native_direct_owned_fixed3600.native_state.npz`
with its matching history/provenance. Actual staged owned callback binding and
normal-capacity/broken-state gates passed; production source was unchanged.
This is the final state after relaxation, not a time-averaged force measurement.

| Quantity | FrameGround / ground | Frame / ground |
|---|---:|---:|
| Generated points |52|15|
| Static coefficient |0.5|0|
| Positive normal impulse |24|0|
| Zero normal impulse |28|15|
| Latest broken bit, all |37|15|
| Latest broken bit, loaded |9|0|
| Loaded points at >=98% local cap |10|0|
| Sum normal impulse |1.62384227e-4 Ns|0|
| Equivalent normal force at h=1/3600 |0.584583 N|0|

All28 unloaded base points have broken set. A projection with zero capacity
can set this bit; it is not evidence of28 physically sliding loaded contacts.
The9 loaded broken points carry32.229% of the normal impulse. Near-cap status
and broken status are distinct: a retained impulse can be near its cap without
a current unconstrained projection crossing the cap.

Normal load is distributed across24 points, with effective participation
`(sum lambda)^2/sum(lambda^2) =14.754`. The largest point carries13.638% and
the largest four39.858%. Loaded impulses range0.2294 to22.1464 micro-Ns,
median5.4227 micro-Ns. The two physical masses total0.575110 N of weight at
source gravity1 m/s². Snapshot normal force is1.647% higher; this is not a
long-time load-balance claim and does not include phase integration history.

## Tangential distribution and full normal wrench

Total point capacity `mu sum(lambda_n)` is8.119211e-5 Ns.
`sum ||lambda_t_i|| / total_capacity =0.889748` while
`||sum world_lambda_t_i|| / total_capacity =0.0426794`.
Loaded local utilization median0.917283, maximum1.000000049 (FP32 rounding).

On the base, summed normal impulse is `(0,0,1.62384227e-4)` Ns;
normal angular impulse about COM is
`(-2.53468955e-7,-1.16481299e-5,0)` Nms.
Summed tangential impulse is `(2.98568736e-7,3.45234384e-6,0)` Ns;
its angular impulse about COM is
`(9.71807648e-9,-8.40505866e-10,3.01339564e-7)` Nms.
All per-point values and origin-based normal moments are saved in JSON.

These opposing point impulses suggest a useful distribution diagnostic, but
are **not proof** of spurious friction, unused feasible capacity, or drift's
cause. Different point forces can intentionally generate a required torque.
A patch change alters the admissible friction wrench and its anchor placement;
a scalar net-force ratio does not certify that the same wrench remains feasible.

## Persistence observables and limits

The52 base points contain two distinct14-float reference-pose signatures:
49 points (21 loaded), and3 points (all loaded). Their stored body-COM reference
translations differ from the current COM by18.96 nm and36.81 nm respectively.
There is no timestamp, and stationary/revisiting body trajectories are not
injective in pose. These values do not establish history lifetimes.

Raw collision match indices:11 base points nonnegative,41 equal-2. This is the
sticky **geometry** map, not the separate history-only identity map used by
the candidate. Negative indices alone must not be called lost history.

## Paper comparison and reusable interface

`/tmp/colibri_rigid_body_friction.txt`, Patch Friction, pp9-11, describes at most
two separated persistent anchors, normal-load sum over the patch, and equal
allocation among anchors with improved patch friction. It explicitly warns
that even two-anchor load division can delay static friction. Thus pooling is
not automatically a convergence or constitutive-law fix. GPU TGS processes
normal rows then friction anchors; Kepler's literal reference targets that law.

`audit_contact_load_distribution.py --prefix PREFIX --output JSON` exposes the
same per-point, positive/zero-mu, load-utilization, broken-state, resultant force
and moment metrics for another compatible native snapshot. It assumes the
saved27-row reference layout and column friction offsets3/4. A hybrid patch
must separately expose its patch impulses/capacities: point rows alone cannot
measure its patch friction. No solver changes or GPU work.

Artifact: `/tmp/colibri_corrected_final_contact_distribution.json`.
