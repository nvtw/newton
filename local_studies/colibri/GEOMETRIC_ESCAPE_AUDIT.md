# Geometric admission: absolute-travel assertion at 203.45 seconds

The original public run **failed** `Body escaped assembly`; this result is not
relabeled as a pass. CPU analysis of its saved history shows attached mechanism
motion together with global base translation and yaw, rather than an escaping
wing connector or a tipped base.

- Base translation from first saved frame: (0.161539, -0.257457, 0.000000238) m.
- Base yaw change: 0.626642 rad; peak tilt change: 0.004720 rad.
- Base height range: 0.2155892–0.2160688 m.
- Peak body displacement in the moving base frame: 0.3645968 m. The same
  measure already reached 0.3645856 m during the first 20 seconds.
- Final maximum speeds: 0.07854 m/s and 3.21365 rad/s.
- Tail rack/mount separation never exceeds 22.9913 mm (original limit 50 mm).
- Original reported peak joint errors: 0.9390 mm anchor, 0.028822 rad axis;
  parent independently measured much smaller final errors (0.25345 mm, 0.001866 rad).

`audit_geometric_escape.py` writes the reproducible CPU report to
`/tmp/colibri_geometric_escape_audit.json`. Its reference is the first saved
post-step pose because this old artifact does not contain the authored initial
pose. WingLeftConnector's final world displacement is about 0.5024 m while its
base-frame displacement is 0.2362 m.

This does **not** validate support accuracy. Both base and ground friction are
0.5; slow translation could be mechanism walking or numerical friction creep.
An attachment/escape test alone cannot distinguish these explanations.

## Local continuation

`check_relative_assembly.py` replaces only the inherited displacement assertion,
using the identical 0.5 m bound relative to FrameGround. Flower is an independent
prescribed body and is excluded from that assembly-relative bound. All original
finite-state, speed, joint, tail-support and fresh penetration checks remain.
The original absolute bound is independently tracked and its first crossing is
reported, including when the local continuation succeeds. No public assertion
or solver equation changes.

```bash
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python \
  -u -m local_studies.colibri.check_relative_assembly \
  --frames 18000 --substeps 30 --geometric-candidates --save-history \
  --output /tmp/colibri_public_geometric_relative18000.json
```

The separate `/tmp/colibri_relative_escape_bounds.json` records the original
absolute-bound outcome. Compare all prior saved poses and velocities bytewise
to establish that changing the diagnostic did not alter the trajectory.

## Completed continuation and example policy

The 18,000-frame / 300-second local continuation passed every remaining original
check and the base-relative bound. The original absolute bound still failed at
203.45 seconds and reached 0.656367 m. Peak relative travel remained 0.364571 m.
Peak fresh penetration was 0.753578 mm, anchor error 0.938996 mm, and axis error
0.028822 rad. All 12,206 old q/qd/timestamp history frames and the old failing
pose are byte-identical to the continuation prefix. Timing was not isolated.

The Phoenx example now selects geometric admission by default; its explicit
--velocity-filtered-candidates option retains the old diagnostic. The global
CollisionPipeline default remains unchanged. The inherited free-base escape
check now uses base-relative coordinates; fixed-base checks remain unchanged.
CPU regression fails before and passes after for rigid global motion, while
detached-body and moved-fixed-base cases still fail as intended.

This five-minute assembly-integrity result does not resolve friction creep.
Support accuracy remains under investigation, and no claim of accurate static
support follows from changing the coordinate frame of the escape assertion.
