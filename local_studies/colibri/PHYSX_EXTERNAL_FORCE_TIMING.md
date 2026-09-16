# PhysX external-force timing control

One-setting companion to awake two-body GPU TGS, plane contact offset1mm,
30 position/1 velocity iterations at120Hz, zero sleep thresholds and damping.

Only composed attribute difference:
`/World/Colibri/PhysicsScene.physxScene:enableExternalForcesEveryIteration=True`.
Schema/importer default is false. The importer reads the exact attribute;
no runtime getter for this flag is exposed. The pinned runtime and logged
ovphysx/ovstage environment are identical to baseline: ovphysx0.5.11 in uv
archive j4xPZTZuhlAwReXiGUPET. Mass, inertia, COM, contact/rest offsets and
material coefficient arrays are byte-identical.

| Setting | Net10–60s XY drift | Max excursion10–60s | Final hinge |
|---|---:|---:|---:|
| Default, forces once per outer step |0.933359µm|1.101434µm|15.728972deg|
| Forces every position iteration |0.077413µm|0.255019µm|15.865726deg|

Late50–60s net motion is0.060797µm, max excursion0.167033µm with the flag.
These accuracy trajectories include readbacks; no performance claim.
The control excludes force-application timing as a sufficient explanation for
the much larger native creep. It does not make the remaining algorithms or
finite-iteration drive law identical.

Files:
- prepare_physx_each_iteration_force.py: composed-attribute and input-hash gate.
- audit_physx_each_iteration_force.py: runtime property byte gate and windows.
- /tmp/colibri_physx_two_body_awake_plane1mm_eachforce.usda and.fixture.json.
- /tmp/colibri_physx_two_body_awake_plane1mm_eachforce60.{json,npz,log}.
- /tmp/colibri_physx_eachforce_comparison.json.

Exact runtime:

    uv run --offline --no-project --python 3.12 --with ovphysx==0.5.11 --with numpy python -m local_studies.colibri.benchmark_physx_two_body --usd /tmp/colibri_physx_two_body_awake_plane1mm_eachforce.usda --frames 3600 --output /tmp/colibri_physx_two_body_awake_plane1mm_eachforce60.json
