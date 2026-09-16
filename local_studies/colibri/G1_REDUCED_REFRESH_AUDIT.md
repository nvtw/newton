# Reduced G1 performance audit

Compared current worktree with f0073be53, using source and the existing
/tmp/g1_current_kernels.csv and /tmp/g1_head_kernels.csv. No runtime changes.

## Measured added work

- Impulse response: 160 new launches, 10.34ms total; HEAD has none.
- Reduced depth factor: 4480 vs2240 launches, 14.12 vs8.02ms total.
- Packed row build: 216 vs56 launches, 6.22 vs1.51ms.
- Contact gather: 216 vs56 launches, 3.11 vs0.86ms.
- Local kinematics:168 launches current; previously only initial import updates.
- Generalized contact tile: same320 launches,22.73 vs17.96ms total.

These are short Nsight totals, not standalone throughput claims.

## Source causes

ReducedPhoenXArticulation.solve_constraints(relax=True) now refreshes factor,
local kinematics and per-body impulse response unconditionally before its
contact-active early return. Refreshing pose-dependent contact response after
integration is a physical correctness fix; removing it wholesale is invalid.

The beginning of each substep already skips the dense per-link impulse response
when contact_block_system.requires_impulse_response is false. The relaxation
path should obey the same predicate: the packed rigid partitioned route builds
responses from ABA and does not consume that extra response cache.

end_substep marks _kinematics_current=True and publishes articulation_origin,
body_q_local, body_q_com and joint_anchor_local. Relaxation currently ignores
this flag and reconstructs those same buffers. A frozen-state comparison should
verify that reusing them matches the intended arithmetic before acceptance.

The full implicit mass factor cannot automatically be reused between relax and
next begin: predicted joint limits depend on q+dt*qd, and relaxation changes qd.
The factor diagonal can therefore change even at the same pose.

## Independent contact solve cost

The packed contact tile computes three cross mobilities from unchanged
generalized Jacobian/response rows for every point on every iteration.
These could be cached once per point per invocation (8to1 for eight sweeps).
Normal/tangent velocity dot products must remain per iteration. Preserve the
correct coupled friction disk minimizer and all momentum/energy tests.

The extra cross_mobility array parameter is in fallback/deferred kernels,
not the packed tile kernel. It does not explain this dominant G1 packed cost.
