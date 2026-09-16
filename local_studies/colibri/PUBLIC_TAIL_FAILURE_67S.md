# Public group-four failure at 67.9167 seconds

The group-four/grid-64 public example at 30 temporal steps per 120 Hz
collision update fails the TailMount/TailPinion axis check at 67.9167 s.
An instrumented repeat reproduces all 4,074 successful pose samples and the
final failed poses and velocities byte for byte. No physics changes or
pose-only restart were used.

Artifacts:

- `/tmp/colibri_public_group4_tail.trace.npz`: last 128 collision updates,
  generation/postsolve poses and velocities, cached contacts and impulses,
  physical masses/inertias/COMs, and final copy counts.
- `/tmp/colibri_public_group4_tail.ring.json`: cached witness transport summary.
- `/tmp/colibri_public_group4_failed_pose_contacts.json`: independent fresh
  contacts at the final eight successful poses and the failed pose.
- `/tmp/colibri_public_8147_pre_sdf.json`: actual generation-pose SDF queries
  at the subsequently penetrating witness locations.

## Observed sequence

At update 8147, ending at 67.900 s, the rack/pinion-thick pair has 14 cached
points. All their transported final separations are at least **+549.8 um**;
all normal impulses are zero. Fresh geometry at that same final pose finds
**422.8 um penetration** on the opposite tooth flank. Every cached normal
has negative dot product with the worst fresh normal (range -0.673 to -0.065).
The pair was present; its represented features did not constrain this approach.

At update 8148, newly detected rack/pinion penetration is corrected to roughly
27 um in the cached planes. Pinion angular speed drops from 10.97 to 3.94 rad/s.
At update 8149, fan/pinion-thin contacts arrive with up to 950 um cached
penetration. The largest normal impulse reaches 0.000468 Ns, pinion speed
reaches 17.39 rad/s, and its joint axis error jumps to 0.03994 rad.
The preceding successful frame's joint axis error was only 0.000646 rad.

Actual pre-update SDF queries of the future worst rack/pinion witnesses give:

| Target | Distance | Initial normal velocity | Linear prediction after 8.33 ms |
| --- | ---: | ---: | ---: |
| Rack | 684.9 um | -1.476 mm/s | 672.6 um |
| Pinion | 1100.6 um | -3.398 mm/s | 1072.3 um |

Mesh queries agree within 0.63 um. Both witnesses are initially closing, but
their initial velocities predict substantial remaining separation. This
differs from the earlier initially receding Cylinder33 case, while sharing
the limitation that contact and joint forces change motion during an update.
The measurements establish missing effective feature coverage before the
large correction impulse; they do not yet isolate raw endpoint rejection
from predictive admission or reduction. Bilateral validity/pivots were not
captured, so they cannot be inferred from this trace.

## Other grouping failures are not identical

Group two fails at 55.55 s on a feather joint's axis. Its strongest ten final
contact impulses are rack/mount contacts (maximum 0.003964 Ns), with endpoint
copy counts 22/13 and several capped normal recovery biases of -2 m/s.
Group three instead fails at 44.93 s on fan/pinion-thin fresh penetration
(1.376 mm), without a comparably large global impulse spike.

The physical tail mass ratio is moderate: mount 31.494 g, pinion 3.187 g,
rack 3.754 g. Rotational response is more demanding: pinion principal inertias
are 1.205–1.850e-7 kg m² versus mount 5.738–12.235e-6 kg m²; the rack itself
has a principal-inertia ratio of 44.6. These are authored physical properties.

No production change follows from this diagnostic. A fresh raw-contact audit
with the endpoint-ownership correction is the next discriminating check.

## Frozen endpoint correction comparison

Exact update8147 pose/velocity/offsets/dt, repeated twice with identical sorted witness sets:

| Mode | Raw count | Reduced count | Minimum transported final gap |
| --- | ---: | ---: | ---: |
| Original predictive | 44 | 14 | +549.8 um |
| Corrected predictive | 68 | 16 | +549.8 um |
| Original geometric envelope | - | 37 | -362.3 um |
| Corrected geometric envelope | - | 40 | -348.4 um |

The endpoint fix does NOT directly restore the missed flank under public admission, even without reduction. Longer passing corrected trajectories therefore cannot establish direct recovery of this particular old-trajectory feature. Initial-twist admission still misses the force-driven approach here. The geometric envelope remains diagnostic, with independent rotating-box coverage limitations.

All candidate points/normals/pre/post gaps: /tmp/colibri_public8147_endpoint_8147_VARIANT.npz. Summary: /tmp/colibri_public8147_endpoint_coverage.json. Worst witnesses: /tmp/colibri_public8147_endpoint_worst_candidates.json. Original sampler loaded privately from /tmp/sdf_contact_before_endpoint_fix.py; no canonical edits.
