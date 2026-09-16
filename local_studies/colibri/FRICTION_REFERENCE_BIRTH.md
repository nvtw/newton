> **Provenance correction:** Native direct/reduced artifacts preceding
> `colibri_native_direct_owned_fixed600` used the new constraint preparation
> overlays but **canonical owned contact callbacks**. The owned import finder
> was installed after an eager package import. Thus those runs are valid
> measurements of that mixed configuration, not tests of combined total-normal
> load plus actual-break owned callbacks. Their old-anchor/all-zero broken-bit
> observations must not be interpreted as behavior of the corrected owned law.
> The local wrapper now verifies actual module paths, kernel-object identity,
> projection-helper identity, kernel source, and an executed CPU break test.
> `check_owned_import_order_negative.py` reproduces and rejects the old order.

# Newborn material-reference coherence

At the independently certified two-body static pose, the original first biased phase has41 nonzero tangential target components among134. Maximum3.2186508e-6m/s corresponds to11.176nm apparent drift with the actual0.08/dt bias rate. Values are quantized at0.9313nm displacement. No body integration preceded this phase.

Actual stored FP32 local anchors, evaluated with independently normalized FP64 transforms offline, still have a maximum3.57e-6m/s birth mismatch. Rebuilding both anchors from one common point entirely in FP64 reduces this to3.60e-14m/s. A standalone source-expression replay does not reproduce all original fused-kernel rounding bitwise; its maximum difference is4.29e-6m/s and is not represented as an exact replay certificate. The authoritative defect evidence is the captured native nonzero targets at unchanged birth, plus their incompatible tangent RHS and the candidate native gate below.

## All-FP32 candidate

`friction_differential_reference.py` stores both birth COM positions and quaternions (14 FP32 values/contact), carries them through matching and colored scatter, and refreshes them when anchors are born or geometrically reset. It defines one coherent material pair using the existing body1 anchor and relative birth pose. Translation uses separate per-body position increments. Rotation differences use the midpoint bilinear quaternion formula, retaining small increments explicitly. No new normalization, double arithmetic, damping, dead zone or changed friction cone is introduced.

History storage grows from13 to27 rows in this unoptimized local prototype. It is not a production change. The prepared FP64 correction variant was abandoned at the explicit user constraint and its CLI is disabled.

## Executed gates

- Actual helper CPU on67 captured contacts: birth delta exactlyzero; represented relative-translation error≤1.82e-12m.
- Actual helper CUDA: birth delta exactlyzero; represented relative-translation error≤3.64e-12m.
- Actual native first-phase capture:27 history rows, all134 tangent bias components exactlyzero. `/tmp/colibri_differential_birth_capture.birth.npz`.
- Previously rejected analytical-equilibrium coupled control now completes1frame/62phases with no failure. This is not a long stationarity claim.
- Native CUDA graph matching/scatter/recontact regression passes, including all14 added values, broken-history reset and unmatched recontact. `/tmp/colibri_friction_reference_lifecycle.log`.

Independent audit still finds raw quaternion norm variation and rounded-position covariance limits. Birth coherence alone does not remove those errors. The next matched native direct120/60s controls will measure actual creep and preserve the already validated finite drive law. Production remains unchanged.

## Matched native direct 60-second controls

All controls retain authored drive gains, 30 substeps at 120 Hz, one biased
iteration and the original final relaxation schedule. No integration guard
was applied. Production source guards pass; actual differential history has
27 rows and the recorded helper hash selects the intended FP32 variant.

| History | 10–60 s horizontal motion | Late speed | FPS (two bodies) |
|---|---:|---:|---:|
| Prior absolute anchors | 121.548 µm | 2.431 µm/s | 20.896 |
| Raw differential FP32 | 126.806 µm | 2.536 µm/s | 20.829 |
| Normalized differential FP32 | 120.590 µm | 2.412 µm/s | 20.617 |

The birth defect is fixed, but neither differential variant establishes
stationarity. Normalized final hinge angle is 18.2073914 degrees and its
finite-drive constitutive residual is −2.49e−9 rad/s. World yaw still changes
−0.0714477 degrees from 10 to 60 seconds. Final 50–60 s speed is 2.441 µm/s.

Actual final contact audits cover all 67 rows. Raw final state has six loaded
overlap contacts strictly inside the static cone, maximum slip 14.411 µm/s;
the last loaded contact has slip 4.15e−12 m/s. This is evidence that later
contact updates reopen earlier rows despite accurate joint closure. The
normalized final state has 13 such interior contacts and maximum slip
1.486 µm/s. These different final snapshots are not a matched frozen-operator
comparison and do not prove a convergence improvement.

Both final snapshots contain an old material reference with tangent recovery
target around 0.02 m/s, unlike most loaded targets of a few µm/s. Raw point44
retains a reference pose about 165 µm from the current base pose. This is
not the newborn roundtrip error. All final broken bits are zero, but they
record the latest relaxation projection and cannot establish that no biased
phase clipped earlier. Final relaxation omits tangent recovery and holds
positive-gap rows fixed, so the audit separates those rows explicitly.
A phase history is needed before attributing positional creep to this older
reference rather than finite contact convergence.

Artifacts: `/tmp/colibri_native_direct_{differential,normalized}3600.*`;
`audit_native_contact_residual.py` records per-row cone fraction, slip,
normal velocity, prepared bias, broken state and per-body loads. These are
residual diagnostics, not an impulse/work ledger or acceptance certificate.
