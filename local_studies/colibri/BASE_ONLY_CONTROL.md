# Isolated FrameGround control

Executed both60-second controls,3600frames,30substeps per120Hz contact update.
Uses public source-built SI geometry, same dynamic base, source gaps/materials,
SOR1,no damping,and originalgroup4/batch2 solver schedule. No attached joints:
model joint records contain only FrameGround/free and Flower/free. The separate
kinematic Flower remains inert in storage; all15of its shapes have flags0 and
cannot interact. Only FrameGround and ground participate in contact dynamics.
Harness: isolated_base.py + check_public_analytic_gradient.py --body-count1.

| Measurement | Production friction | Total-normal local candidate |
| --- | ---: | ---: |
| Maximum horizontal motion from initial pose |0.192732mm|0.192642mm|
| Final displacement from initial pose |0.041436mm|approximately0.04178mm|
| Net horizontal motion,10to60s |0.323882µm|1.696987µm|
| Maximum horizontal excursion from10s pose |0.463933µm|1.849817µm|
| Rotation change,10to60s |0.00015458degrees|0.01677364degrees|
| Mean isolated-base physics frame time |4.08244ms|4.10162ms|

Both geometric checks pass. These timings are BASE ONLY and do not establish
stable full-assembly FPS. Small startup motion is largely settling; production
base-only post-settling motion is submicrometer over50seconds. The total-normal
candidate has more small rotational creep even in this control. Neither result
proves exact rest for arbitrary loads, nor excuses full-scene drift.

Original source run: /tmp/colibri_base_only_original3600.json/.npz/.log.
Candidate: /tmp/colibri_base_only_total_normal3600.json/.npz/.log.
Paired post-settling metrics: /tmp/colibri_base_only_paired_motion.json.
The pose rotation metric normalizes quaternions and uses relative quaternion
atan2, avoiding loss of precision from small-angle acos. No simulation state
is modified by this measurement.

Next isolation step: preserve construction and add only Frame and its authored
joint, comparing production friction and the candidate before more bodies.
