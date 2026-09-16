# Normal-first contact prototype: mutable scalar read

The local prototype solves the normal row with the existing helper, then computes
friction load and conditionally evaluates tangent mobility. Production contact
equations are unchanged; the earlier zero-friction specialization is disabled.

## Actual prepared-row regression

Run:

    uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m unittest -f local_studies.colibri.test_normal_first

The matrix includes 128 cases: static/dynamic friction (including positive static
with zero dynamic), PD, biased/relax sweeps, speculation, zero/nonzero old normal
impulse, and stale tangent impulses.

With the native getter for the read immediately after normal update, the first
PD/biased case writes normal lambda 0.0002 but applies zero body impulse. The
general reference applies the expected +/-0.0002 linear velocity and angular
response. Both have the same final normal lambda in memory.

Changing only

    new_n = cc_get_normal_lambda(cc, k)

to

    new_n = cc.impulses[0, k]

makes all 128 cases array-exact against the general reference. The failing source
is saved at /tmp/prototype_normal_first_before_direct_read.py; the passing source
is local_studies/colibri/prototype_normal_first.py.

The native getter routes through read2d_f32, whose C++ snippet reads a const
__restrict__ pointer. This is evidence of an unsafe read-after-write in this
larger compiled context; the precise compiler/cache mechanism is not yet isolated.
The small standalone local_studies/colibri/test_contact_read_after_write.py passes,
so do not claim that every invocation of the native getter is broken.

A final normal-first helper should return normal delta and friction load directly,
avoiding redundant global writes/reads. Mutable native accessors still warrant a
separate audit rather than hiding the discrepancy inside that optimization.

## Return-value candidate and controlled scene check

The local normal helper now returns its delta and friction load directly. It
preserves the existing bounded normal row equations and avoids any read-back.
All 128 prepared-row cases pass array-exact against the general path.

A process-local wrapper, check_normal_first.py --normal-first, replaces only the
ordinary iterate entry points. Production source is unchanged. Isolated full
37-body matched 30x1/120Hz/300-frame runs measured 62.6701 ms baseline versus
55.2611 ms candidate (11.82% improvement). Final body poses, velocities, impulses,
material anchors and derived rows are array-identical; both max depth267.295um,
final38.790um. These use the existing1mm fallback, not the later exact PhysX
offset map that parent found fails near frame297.

Reports: /tmp/colibri_normal_first_reference_300.json and
/tmp/colibri_normal_first_candidate_300.json, with matching NPZ state artifacts.

The standalone recreate script reproduce_contact_native_read.py rebuilds the
failing prepared-row variant from the current local prototype. It deliberately
expects unittest failure and keeps the actual large-kernel regression available.
The helper/native accessors in production have not been changed. Existing
production coupled helpers return their normal delta directly and write impulse
storage at the end, so this specific newly introduced read-after-write pattern
does not show an established production failure.

## Production integration

Parent authorized integration after the exact full-scene comparison. The
production ordinary rigid factory now defaults to normal-first, with its private
normal_first=False reference retained for tests. The normal delta/load and tangent
delta helpers live in contact_projection.py. Cloth and patch fallback behavior
is unchanged. All 128 actual prepared-row cases still pass array-exact, along with
ordinary kinetic-energy and sustained linear/angular momentum checks, material
anchor history/reset and mass invariance, six maximal conservation tests, and
the unchanged full11,341-body Kapla settling test. No native accessor changed.
