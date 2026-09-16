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

# Native conditioned two-body control

Production source and authored assets remain unchanged. Local wrapper `native_conditioned_two_body.py` combines the staged actual-normal friction law and solve-generated break state (including direct/maximal owned callbacks) with gap-history and fresh-geometry identity matching. It verifies authored mass/inertia/COM, drive gains/modes/targets, passive damping and armature remain unchanged. Initial pose arrays are byte-identical between direct and reduced controls.

Both controls use30 temporal substeps per120Hz refresh, one biased iteration, final-substep relaxation and SOR1. They use physical unsplit masses, group0, chunk0 and native preparation. These settings are required alternatives to experimental block-PGS grouping/chunking, not a matched dispatcher performance comparison.

## Drive equation

Direct equality native dynamic mass is armature + dt*passive_damping + dt*kd + dt²*ke, with reference momentum armature*v_old + dt*(ke*(target-q)+kd*target_velocity). The original finite drive is retained. Reduced dynamics uses the same implicit diagonal and linearized implicit-Euler force. Neither path replaces the spring with a hard velocity constraint.

The direct control selects native tree-conditioned contacts (`direct_tree_contacts=True`) with direct equality owning the hinge. The reduced control owns the declared tree in generalized coordinates. No identical previous combined-history run was found.

## Completed results

- Direct120 actual control: geometry checks pass, maximum base horizontal excursion69.34µm, hinge18.30666 degrees at2 seconds.
- Direct3600: all geometry/source gates pass. Final hinge18.207434 degrees, consistent with the independent flat-support analytical equilibrium18.2073 degrees. Final native drive equation residual1.776e-9rad/s from saved actual wrenches, body velocities, accumulated impulse and dynamic mass.
- Direct late10–60 seconds:121.548µm horizontal movement (2.431µm/s);50–60 seconds26.597µm (2.660µm/s). World yaw changes−.071680 degrees over10–60 seconds; world roll/pitch changes below.000031 degrees. It still creeps.
- Direct two-body synchronized step timing20.896FPS,47.8555ms/frame; not full-scene FPS and not accepted as a fast/stationary solution.
- Reduced120: geometry/source gates pass, maximum base excursion63.84µm, hinge18.30736 degrees,47.465FPS. No60-second extension yet. Final contact/body/response arrays retained for residual analysis; short settling velocities alone do not establish a steady-state Coulomb defect.

Artifacts `/tmp/colibri_native_direct_combined3600{.json,.npz,.native_conditioned.json,.native_state.npz,.motion.json}` and `/tmp/colibri_native_reduced_checked120` with corresponding files. The first `/tmp/colibri_native_direct_combined120` attempt is REJECTED as a direct configuration experiment: runpy bypassed a module-alias hook. Corrected wrapper patches the actual constructor and requires recorded ownership; `/tmp/colibri_native_direct_checked120` is the valid short control.

## Remaining interpretation

Native joint conditioning resolves the previously large drive constitutive error and wrong hinge equilibrium. It does not eliminate contact-contact convergence/history/integration errors or base creep. Next work must address that remaining physical contact behavior; neither geometry pass nor analytical hinge agreement establishes full stationarity.

## Corrected actual-owned-callback baseline

`/tmp/colibri_native_direct_owned_fixed3600.*` passes all actual staged module
paths, solver kernel object identity, projection-helper alias identity, source
load/write checks, executed CPU 99%-stick and sliding checks. Its first600
frames reproduce corrected600 q/qd/time history bytes exactly. Production
sources remain unchanged.

At60s:20.4515FPS (two bodies);10–60 drift167.000µm=3.34000µm/s,
50–60 drift33.1867µm=3.31867µm/s. World yaw10–60 changes−0.0878737deg.
Hinge18.2074486deg, final drive constitutive residual−5.02e−10rad/s.
This is not a stationary solution.

Final maximum tangent recovery target8.89681µm/s versus20,047.6µm/s in
the old mixed-callback60s control. Thus the actual history fix removes the
large old target but does not remove creep.52/67 final broken bits are set;
23loaded overlap points,15interior points, maximum interior physical slip
1.166µm/s. Final relaxation holds speculative points and omits tangent bias;
these residuals alone do not measure pose-integration velocity.

The separate old-callback bias observer (`colibri_native_direct_bias330`)
shows actual joint recovery restored after contact solve, with base mean
bias velocity(.913,−2.177,+3.599)µm/s in its final ring. This is source/mechanism
evidence only; it predates corrected owned callbacks and must be recaptured
against the corrected baseline before a causal claim.

### Matched60s provenance comparison

| Actual owned callbacks |10–60 drift speed|50–60 drift speed|Max prepared tangent bias|Final broken points|
|---|---:|---:|---:|---:|
| Canonical (old mixed configuration) |2.41179µm/s|2.44129µm/s|20.0476mm/s|0/67|
| Correct staged load/history |3.34000µm/s|3.31867µm/s|8.89681µm/s|52/67|

Neither result meets stationarity. Corrected20.4515FPS is two-body diagnostic
throughput, not full-sculpture throughput or an accepted quality result.
Neither saved60s file contains schedule.mobility, so actual counts of the
pre-existing conditionalFP64 friction branch cannot be reconstructed honestly.
Future final snapshots save that array. New reference arithmetic is FP32;
this does not imply the entire native solver uses only FP32.

Independent source review of `tangent_recovery_trial.py`: adding C_t*v_bias
to the temporary tangent RHS has the correct sign and units because the
native solve subtracts that velocity first and restores it afterward. No dt
conversion is required. It leaves normal recovery and normal-to-tangent
crossresponse intact and restores old targets by exact assignment. The
schedule assigns a unique owner to each column, avoiding parallel duplicate
saves. Execution must still validate lazy buffer allocation under CUDAgraph
capture and actual phase/work behavior. This is a bounded trial, not a proven
whole-system correction.
