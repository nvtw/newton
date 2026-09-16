# Native drive impulse accounting audit

## Conclusion

The saved crank constitutive residual is real finite-iteration error. There is no missing copy-count factor or stale accumulated-impulse reset in the examined path. This is a frozen final-relaxation audit at frame 330, not a trajectory-wide accuracy proof.

## Physical equation and lifetime

For a dynamic joint row, the solver uses
`r = J v + lambda / m_dynamic - v_reference`.
Preparation constructs `m_dynamic = armature + dt*damping + dt^2*stiffness` (including the applicable passive terms), and the reference from the corresponding old momentum and drive targets. For these angular rows, lambda is angular impulse in N m s; `lambda/dt` is the last substep's mean torque, not a full-frame average torque.

The snapshot has dt = 1/3600 s and unbounded drives. In
`articulations/direct_equality.py`, `begin_substep` invokes the snapshot kernel that resets accumulated impulse once. Subsequent `refresh_geometry` updates geometry without resetting impulse, dynamic mass or reference. The final relaxation adds to the same accumulation.

In `constraints/bilateral_joint.py`, each joint row adds its solved impulse exactly once. Its endpoint copy receives `c_b W_b J_b^T delta_lambda`, because `constraint_joint._ms_load_body_pair` scales inverse physical mass/inertia by that body's copy count. The final copy average divides by c_b, so the physical response is `W_b J_b^T delta_lambda`. Thus lambda is already physical; dividing it by copy count would introduce an error. Broadcast/average kernels alter velocities, not accumulated joint impulse.

## Independent scheduling and copy audit

Joint initialization uses cid offset zero. The deterministic greedy builder in `mass_splitting/color_groups.py` visits constraint IDs in ascending order, so contacts cannot recolor earlier joints. Joint 0 connects bodies 1/2 and receives color 0; joint 1 connects 2/3 and receives color 1. The grouped sweep passes `slab=color//4` as the copy partition ID. Both drives therefore use partition 0.

CPU script `audit_drive_copy_accounting.py` independently reconstructs this schedule and locates each body's partition-0 slot in the captured CSR copy table. Results reproduce the parent's separate audit:

| Joint | Final group-copy residual (rad/s) | Physical averaged residual (rad/s) | Averaging increment |
|---|---:|---:|---:|
| Base-Frame | 0.036732305 | 0.132004888 | 0.095272582 |
| Crank | 0.225291421 | 0.490757978 | 0.265466557 |

There is substantial error before averaging, consistent with later coupled rows reopening an earlier drive equation. Averaging further increases it. These endpoint snapshots do not isolate every individual intervening row's contribution.

## Crank quantitative budget

Physical Jv after relaxation is -2.999159034 rad/s against reference -3.490658522 rad/s. Accumulated impulse is -1.180149411e-6 N m s and inverse dynamic mass is 628.3185083. The intended compliant load lag is only 0.000741510 rad/s; the remaining constitutive residual is 0.490757978 rad/s.

The final relaxation reduces residual from 0.639747639 to 0.490757978 rad/s. Its own drive impulse change predicts a physical Jv change of -0.233770540 rad/s, while all rows together change Jv by -0.148596470 rad/s. Other row responses therefore contribute +0.085174070 rad/s along this coordinate. This is coupling/convergence evidence, not justification for altering drive gains, masses or compliance.

## Reproduction and artifacts

CPU-only:
`uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.audit_drive_copy_accounting`

Inputs:
- /tmp/colibri_support_relax330.npz
- /tmp/colibri_ground_pre_average330_copies.npz

Outputs/evidence:
- /tmp/colibri_drive_copy_accounting_independent.json
- /tmp/colibri_drive_accounting_audit.json
- /tmp/colibri_native_drive_lag.json

No canonical solver changes, GPU kernel launches, or tolerance changes were made for this audit.
