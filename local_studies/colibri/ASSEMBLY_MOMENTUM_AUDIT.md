# Colibri aggregate motion and energy audit

The saved canonical geometric-candidate trajectory stops at203.45s because
of the original absolute-displacement assertion. The CPU-only script
`audit_assembly_momentum.py` uses saved body poses, COM velocities, masses,
COM offsets and local inertia tensors. It excludes the kinematic Flower.
Quaternion rotation is evaluated in FP64; a known90-degree rotation and100
random inverse-rotation round trips agree within3e-15.

Dynamic mass is1.532687872kg. Assembly COM displacement through203.433s is
(0.157523,-0.226275,0.003448)m. Mean kinetic energy in full20-second windows
after startup stays between0.35975 and0.36658mJ. There is no sustained kinetic
energy growth in these windows. Mean horizontal momentum is approximately
0.002kg m/s and rotates gradually with the base heading.

This is a trajectory diagnostic, **not a conservation test**: saved states
alone do not include ground/Flower impulses or motor work. These results do
not establish that the observed sliding is physical rather than numerical
creep. A contact impulse/work audit or convergence comparison is still needed.

Reproduce:

```sh
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.audit_assembly_momentum /tmp/colibri_public_geometric18000.npz --output /tmp/colibri_geometric_momentum_audit.json
```

## Current integrated 300-second trajectory

CPU diagnostics were repeated on the cooperative-preparation default trajectory
(`/tmp/colibri_cooperative_prepare_velocity1_long18000.npz`). Dynamic assembly
COM displacement is (0.261911, -0.299937, 0.004187) m. Complete post-startup
20-second windows have mean kinetic energy 0.360255–0.372077 mJ; there is
no sustained kinetic-energy growth across these windows. The final 300-second
window is only an endpoint sample and is excluded from that range.

The crank stays rotating in every sampled interval, mean -3.183971 rad/s
(target -3.490659), with sampled speeds -4.225986 to -2.161369 rad/s.
Observed gear phase fits remain close to the earlier rational ratios, but
fitted ratios still do not independently prove tooth engagement or exclude
steady slip. Spinner phase variation is 0.312381 rad.

Outputs: `/tmp/colibri_current300_momentum_audit.json` and
`/tmp/colibri_current300_gear_audit.json`. These are trajectory diagnostics,
not an external impulse/work balance. Ground/Flower impulse accounting and
convergence of support creep remain required before classifying drift as
physical.
