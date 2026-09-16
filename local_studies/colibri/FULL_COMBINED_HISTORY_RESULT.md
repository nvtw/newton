# Full Colibri with combined friction history fixes

The 60-second run finished with all 36 dynamic mechanism bodies and Flower.
Configuration: 120 Hz collision updates, 30 substeps, one biased solve and
final velocity relaxation, SOR 1, existing group size 4. The base is free.

Synchronized public step timing after 30 warm-up frames: **10.5583 ms mean,
94.712 FPS**, excluding rendering and read-only physical audits. Existing
geometry/joint bounds passed, but stationarity failed:

- Base horizontal displacement: **7.9150 mm** from initial pose.
- Base rotation: **0.149181 rad (8.55 degrees)**.
- Net horizontal displacement from 10 to 60 seconds: **5.9301 mm**.
- Net horizontal displacement from 50 to 60 seconds: **1.1688 mm**.

The late displacement rules out interpreting this as initial settling alone.
No stable full-scene performance claim or production adoption follows from
this run. Friction history fixes alone do not resolve joint/contact coupling.

Artifacts: `/tmp/colibri_full_combined_history3600.json`, `.npz`, `.log`, and
`.combined_history.json`. The run uses the `combined_gap_matching` and
`total_normal_friction` overlays and retains source-file integrity checks.
