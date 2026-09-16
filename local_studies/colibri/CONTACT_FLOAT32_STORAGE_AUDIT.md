# Contact root storage precision

`audit_island_float32_storage.py` independently checks the 789 accepted
FP64 roots from the 960-case high-mass replay. The 171 rejected roots are
excluded; this audit does not change their status or any solver tolerance.

The model of storage is explicit:

1. Round the solved impulses to FP32.
2. Accumulate their physical effect using the geometry-consistent FP64
   Jacobian and inverse mass.
3. Round the resulting body velocities to FP32.
4. Recompute the Coulomb residual from those stored velocities and impulses.

This models ideal FP64 accumulation followed by FP32 storage. It is **not**
the native GPU scatter, which also has arithmetic and accumulation error.
No production kernel or array dtype changes.

## Results

| Quantity | Maximum observed magnitude |
| --- | ---: |
| Accepted FP64 root residual | 7.843e-9 m/s |
| Stored-state residual | 6.080e-6 m/s |
| Linear momentum plus external reaction imbalance | 2.503e-6 N s |
| Angular momentum plus external reaction imbalance | 3.732e-6 N m s |
| Kinetic-energy/work identity error | 4.070e-7 J |
| Positive friction-disk violation | 2.407e-7 N s |

787 of the 789 stored results exceed the original 1e-8 m/s reference
threshold. All still dissipate kinetic energy in this particular sample.
That observation alone is not a contact admissibility test.

## Independently checked bounds

Each FP32 store is bounded by half the larger adjacent FP32 spacing.
The script propagates those componentwise bounds through the physical
mass, Jacobian, and contact impulse response. It checks:

- Linear momentum error against the mass-weighted velocity-store bound.
- Angular momentum error including COM cross linear-momentum error.
- Work error against the mass-weighted storage error dotted with midpoint
  velocity.
- Natural-map residual changes using nonexpansiveness of disk projection
  and the change in friction-disk radius.
- Friction-disk violations against normal/tangent impulse quantization.

All 789 cases satisfy these bounds. The bounds are computed independently
from adjacent FP32 spacings, not fitted to the observed errors. Small FP64
evaluation allowances are explicit in the assertions.

## Consequence for integration

Retain the strict FP64 convergence gate. A future native implementation
must separately audit its realized physical state, with its actual
arithmetic and scatter order. It cannot claim a 1e-8 residual solely from
an FP64 root before storing the result. These storage bounds do not justify
accepting an unconverged root, increasing friction, or adding damping.

Results: `/tmp/high_mass_float32_storage.json` (summary plus every case).

```sh
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.audit_island_float32_storage
```
