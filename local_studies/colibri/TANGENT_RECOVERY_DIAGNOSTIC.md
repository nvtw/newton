# Tangential joint-recovery trial diagnostic

## Result

The local candidate is **not a creep solution**. Corrected native direct two-body
600-frame controls, 30 substeps per 120 Hz collision interval, one biased and one
final relaxation iteration, SOR 1, normalized FP32 material reference:

| Control | Base origin horizontal drift, 5–10 s |
|---|---:|
| Corrected owned callbacks, unchanged recovery treatment | 2.445929 µm/s |
| Tangential trial includes restored hard-joint recovery | 3.013147 µm/s |

Candidate passed existing geometry/joint/source guards; depth 46.20 µm, anchor
75.34 nm, axis 4.37e-7 rad. Synchronized public-step timing was 48.584 ms
(20.583 FPS). It adds two small kernels per biased contact call. No 60-second
extension or production promotion is justified.

Artifacts: /tmp/colibri_native_tangent_recovery600.*;
comparison /tmp/colibri_tangent_recovery600_comparison.json.
The actual owned callback module paths, solver kernel object identity, metric
helper identity, source semantics, and CPU break-state probe are checked by the
corrected native wrapper. Older native sweep results before that wrapper repair
did not execute all intended owned friction changes and cannot establish their
effect.

## Source mechanism and scope

PhoenX integrates biased velocities before its final relaxation. The direct-owned
contact path computes the hard-joint recovery velocity, strips it before contact
solves, then restores it before position integration. Final relaxed velocity does
not undo motion already integrated. PhysX TGS also integrates positional-iteration
motion and writes accumulated displacement separately from its final velocity;
the existence of this ordering alone is not a PhoenX defect.

The strip protects near-blocked normal contacts from enormous recovery impulses.
Removing it globally is unsafe. This bounded candidate leaves normal trials,
normal response, normal load, joint recovery, drive rows, and joint-nullspace
contact response unchanged. It temporarily adds C_t * v_recovery to tangent
targets, preserves the existing normal-to-tangent cross response, then restores
the exact old targets. It applies no second recovery impulse.

This is a mixed-channel diagnostic: normal constraints see stripped velocity,
friction sees restored velocity. It is not a proof of a complete stabilized
Coulomb formulation for arbitrary coupled normal/tangent geometry. A live
per-phase impulse/work acceptance was not established; the live check establishes
geometry bounds and motion only. Its worse creep rejects it for the current goal.

## Regression evidence

check_joint_recovery_friction.py invokes the actual FP32 native metric projector
on an analytical two-mass, collinear relative-recovery fixture with an independent
ground normal. Legacy tangent trial leaves base velocity -50 µm/s despite
available friction; candidate leaves -1.26e-12 m/s, preserves relative recovery
1e-4 m/s and paired P/L. Contact work -2.5e-9 J plus recovery work +5e-9 J equals
kinetic change +2.5e-9 J. --legacy fails the stationarity assertion; default passes.
This counterexample proves only its stated geometry/channel behavior.

check_tangent_recovery_trial.py CPU-compiles the actual shift kernel and checks
off-center velocity signs, untouched normal and unused-tail bytes, exact target
restoration, and zero friction with zero cone radii.

check_tangent_recovery_near_blocked.py runs the existing native near-blocked
mu=0 recovery regression at its original 1-substep/8-iteration budget with the
direct solver. Baseline and candidate pass and produce byte-identical q/qd.

## Reproduction

Use the repository Python environment through uv:

    CUDA_VISIBLE_DEVICES='' uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.check_joint_recovery_friction --legacy
    CUDA_VISIBLE_DEVICES='' uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.check_joint_recovery_friction
    CUDA_VISIBLE_DEVICES='' uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.check_tangent_recovery_trial
    uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.check_tangent_recovery_near_blocked

Local live control (not a default):

    COLIBRI_DIFFERENTIAL_REFERENCE=1 COLIBRI_DIFFERENTIAL_NORMALIZED=1 COLIBRI_STAGE_RUNNER=local_studies.colibri.tangent_recovery_trial uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.native_conditioned_two_body --native-path direct --body-count 2 --frames 600 --substeps 30 --save-history --output /tmp/colibri_native_tangent_recovery600.json

Final contact readout has 11 loaded overlap points, three strictly inside the
friction cone, maximum interior slip 3.206 µm/s and loaded |vn| 6.754 µm/s.
It is not a work ledger or proof of stationary support.
