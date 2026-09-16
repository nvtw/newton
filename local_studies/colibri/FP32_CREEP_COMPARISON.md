# FP32 creep comparison and current causal checks

The target is PhysX-level bounded base motion while retaining the analytical
spring equilibrium, physical impulse balance, Coulomb friction, and SOR 1.
No candidate is accepted merely because it is faster or passes geometry bounds.

## Import-provenance correction

The phase observer established that the native runs below loaded canonical
maximal_contact_gs.py, not the staged owned-contact callback. The native
wrapper installed its import finder too late. Hashing staged files did not
prove that their functions executed. The 27-row FP32 reference preparation
did execute, but these runs did **not** include the intended owned total-normal
load and explicit break-state corrections. Their measured trajectories remain
valid observations of that partial configuration. They cannot reject or
validate the intended combined fixes. Corrected runs must assert actual loaded
module paths and function provenance before simulation.

## Matched two-body results

Both controls use a dynamic FrameGround plus Frame, source gravity of 1 m/s²,
120 Hz contact updates and 30 temporal substeps. PhysX sleeping, stabilization,
and damping are disabled. These are not full-sculpture results.

| Control | Translation, 10–60 s | Net translation / elapsed time |
| --- | ---: | ---: |
| PhysX awake plane control | 0.933 µm | 0.0187 µm/s |
| PhoenX native conditioned, normalized FP32 reference | 120.590 µm | 2.4118 µm/s |

PhysX's maximum excursion from its 10 s pose was 1.101 µm. Its small net
displacement is not evidence of a constant creep rate or exact zero motion.
PhoenX's 50–60 s displacement was 24.413 µm, indicating persistent motion.
PhysX settled near 15.729 degrees; PhoenX near 18.2074 degrees, consistent
with the independent nominal spring equilibrium at 18.2073 degrees.

Artifacts: `/tmp/colibri_physx_two_body_awake_plane1mm60.*` and
`/tmp/colibri_native_direct_normalized3600.*`.

Increasing both biased and final velocity sweeps did not solve the problem.
Over the same 5–10 s window, 1/2/4 sweeps gave 2.210/5.576/2.418 µm/s.
The extra work is not a justified production change.

## Current discriminating experiments

1. Capture the native biased solve, final relaxation, and contact refresh with
   a mandatory byte-identical trajectory check. A final zero broken bit alone
   does not prove that an anchor did not slide earlier. The 5.5 s trace had no
   break transitions but did not yet contain the later large old-anchor bias.
2. Measure joint position-recovery velocity removed before conditioned contacts
   and restored before pose integration. This can leave part of the integrated
   motion invisible to contact friction. The split protects nearly blocked
   mechanisms from huge recovery impulses; removing it blindly is not a fix.
3. Compare material-reference derivatives with the actual friction Jacobians.
   PhoenX uses fresh common-point levers while storing older material anchors.
   PhysX uses persistent anchor levers, but copying separated impulse points
   would violate our angular-momentum requirement.
4. Run the native solver from the independently certified static initial pose.
   `native_equilibrium_start.py` composes existing overlays before importing
   the scene builder. Only initial poses change; no body is pinned and no
   contact impulses are seeded. This separates startup history from sustained
   equilibrium defects. This control is prepared, not yet measured.

PhysX GPU source also truncates small lever-arm components and uses patch
friction. These are comparison differences to quantify, not changes to copy
without checking physical consequences. The new material-reference arithmetic
is FP32. This does not certify every existing solver branch as FP32; the
pre-existing metric projection has a conditional FP64 fallback for poorly
conditioned tangent mobility. No new double-precision path is introduced here.
