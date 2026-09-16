# Fixed joint row selection across actual poses

Current-source captures at frames1,30,120 passed source hashes and byte-exact trajectory-prefix checks independently. assemble_joint_capture.py reconstructs J/W/rhs/A/initial/compliance; it reproduces all original330 operator arrays bit-exact. reform_joint_pivots.py is now parameterized by snapshot/operator/output.

## Exact rank and all-row checks

The SAME184 independent row IDs certified at330 remain independent and span the complete reformed operator at all three poses. Exact rational full rank and selected rank are184, nullity4. No constraints were silently dropped and no numerical rank cutoff was used.

| Frame | All188 equation residual, FP64 | Smallest selected factor SV | Max impulse coefficient |
|---|---:|---:|---:|
| 1 | 5.39e-10 | 5.97e-10 | 9266 |
| 30 | 2.34e-10 | 1.10e-9 | 12595 |
| 120 | 3.59e-11 | 6.84e-10 | 5647 |

Exact normalized null-RHS inconsistency stays below1.28e-19. The geometry reform gives exactly zero total torque coefficient at all poses.

## Mobility and gauge stress

All eligible contacts (49/38/37 respectively) were checked. Gram mobilities are exactly symmetric and PSD by construction, retaining resolved modes. Physical virtual-work relative errors are below2.95e-15. Maximum joint-response residuals per unit contact impulse are1.85e-9,5.48e-11,3.65e-10.

An alternative reverse-order exact-independent row selection is also mathematically valid but numerically less accurate: compared to the fixed330 basis, its contact mobility differs relatively by8.21e-6,4.47e-7,1.44e-6. Corresponding physical velocity differences are1.60e-5,7.28e-6,2.91e-6. Both gauges still give small all-row equation residuals, so residual alone does not certify forward accuracy in this ill-conditioned problem.

A45-digit cold-frame1 reference resolves which is better:
- Fixed330 FP64 mobility relative error1.40e-8; maximum coefficient error2.21e-6.
- Fixed330 FP64 velocity correction error2.32e-8.
- Reverse-gauge FP64 mobility relative error8.20e-6.

## Conclusion

The fixed184-row structural selection generalizes across these actual poses, including the cold pose. Generic FP64 gauge independence is NOT established. Accuracy worsens near small physical modes and depends strongly on basis conditioning. A practical method needs a well-conditioned certified basis and forward-response/invariant checks, not merely a small linear residual. No physical mode was truncated, no live correction was accepted, and no production source was edited.

Artifacts:
- /tmp/colibri_joint_pose{1,30,120}_{operator,reformed,rank}.*
- /tmp/colibri_joint_pose_gauge_mobility.json
- /tmp/colibri_joint_pose1_mp.{npz,json}
- /tmp/colibri_joint_pose1_mp_mobility.npz

Scripts: assemble_joint_capture.py, reform_joint_pivots.py, audit_joint_pose_rank.py, audit_joint_pose_mobility.py, probe_joint_unsquared_mp.py.
