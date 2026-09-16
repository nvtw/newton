# Mixed joint/contact conservation audit

The existing grouped conservation test exercises contacts only; the block
policy test exercises a motor without external contacts. This local fixture
adds a revolute joint between two of three dynamic boxes while the third
box remains in frictional contact. It tests both body orders and color-group
widths1 and4, retaining unequal body copy counts, moving centers of mass,
and anisotropic inertia. Nonzero bilateral impulses and active contacts
are asserted.

All four cases pass conservation of total physical linear/angular momentum
in every biased and relaxation phase, cumulative momentum, and non-increasing
final kinetic energy. This does not establish high-mass-ratio convergence
or motor-work accounting. No solver code changed.

Run: `uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python
-m unittest local_studies.colibri.test_mixed_joint_contact_conservation`.
Result: `/tmp/colibri_mixed_joint_contact_conservation.log`.
