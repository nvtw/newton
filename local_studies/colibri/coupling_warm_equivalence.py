import numpy as np
import warp as wp
from newton.viewer import ViewerNull
from newton._src.solvers.phoenx.examples.example_rigid_cloth_rigid_drop import Example
from newton._src.solvers.phoenx.articulations.direct_contact_response import DirectContactResponseData
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer, cc_get_normal, cc_get_tangent1, cc_get_normal_lambda, cc_get_tangent1_lambda, cc_get_tangent2_lambda, cc_get_r0, cc_get_r1
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.articulations.direct_contact_gs import _apply_raw_contact_impulse


@wp.kernel
def raw_warm(data: DirectContactResponseData, cc: ContactContainer, bodies: BodyContainer):
    # One thread keeps the diagnostic scatter deterministic.
    for k in range(data.contact_mechanism.shape[0]):
        if data.contact_mechanism[k] >= 0:
            n = cc_get_normal(cc, k)
            t = cc_get_tangent1(cc, k)
            lam = cc_get_normal_lambda(cc, k)
            if data.mobility[0, k] <= 1.0e-12:
                lam = 0.0
            impulse = lam*n + cc_get_tangent1_lambda(cc, k)*t + cc_get_tangent2_lambda(cc, k)*wp.cross(n,t)
            _apply_raw_contact_impulse(bodies, data.contact_body0[k], cc_get_r0(cc,k), -impulse)
            _apply_raw_contact_impulse(bodies, data.contact_body1[k], cc_get_r1(cc,k), impulse)


e = Example(ViewerNull(), width=6, height=6)
e.graph = None
for _ in range(30):
    e.step()
w = e.solver.world
d = e.solver._direct_equality_system
original = wp.launch_tiled


def instrument(*args, **kwargs):
    kernel = kwargs.get('kernel', args[0] if args else None)
    if getattr(kernel, 'key', '') != 'warm_start_direct_contact_runs_kernel':
        return original(*args, **kwargs)
    arrays = [w.bodies.velocity, w.bodies.angular_velocity, d.accumulated_impulse, d.rhs, d.delta, d.solve_active]
    saved = [wp.clone(a) for a in arrays]
    before = np.column_stack((arrays[0].numpy(), arrays[1].numpy()))
    d.solve(use_bias=False)
    baseline = np.column_stack((arrays[0].numpy(), arrays[1].numpy()))
    for a, s in zip(arrays, saved):
        wp.copy(a,s)
    wp.launch(raw_warm, dim=1, inputs=[w._direct_contact_response.data,w._contact_container,w.bodies],device=w.device)
    d.solve(use_bias=False)
    reference = np.column_stack((arrays[0].numpy(), arrays[1].numpy())) - baseline
    for a, s in zip(arrays, saved):
        wp.copy(a,s)
    original(*args, **kwargs)
    projected = np.column_stack((arrays[0].numpy(), arrays[1].numpy())) - before
    error = projected-reference
    print('WARM_EQUIVALENCE', np.max(abs(error)), 'reference',np.max(abs(reference)), 'projected',np.max(abs(projected)),flush=True)
    print('WORST', sorted(enumerate(np.max(abs(error),axis=1)),key=lambda x:x[1])[-8:],flush=True)
    print('CUBE', reference[e.cube_body+1],projected[e.cube_body+1],flush=True)
    np.savez('/tmp/colibri_cube_warm_equivalence.npz',before=before,reference=reference,projected=projected,error=error)
    raise SystemExit


wp.launch_tiled = instrument
e.step()
