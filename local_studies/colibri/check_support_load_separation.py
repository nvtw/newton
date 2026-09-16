"""Expose coupled-support load subtraction using the production CPU function."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import sys
from pathlib import Path
import numpy as np
import warp as wp
wp.config.enable_cuda = False
from newton._src.solvers.phoenx.constraints.contact_projection import _friction_normal_lambda
from newton._src.solvers.phoenx.constraints.constraint_container import soft_constraint_coefficients

@wp.kernel
def evaluate(lambdas: wp.array(dtype=wp.float32), diagonal: wp.array(dtype=wp.float32),
             biases: wp.array(dtype=wp.float32), result: wp.array(dtype=wp.float32),
             coefficients: wp.array(dtype=wp.float32)):
    i = wp.tid()
    rate, mc, ic = soft_constraint_coefficients(wp.float32(1.0e9), wp.float32(1.0), wp.float32(1.0 / 1800.0))
    result[i] = _friction_normal_lambda(lambdas[i], wp.float32(1.0) / diagonal[i], biases[i], mc, wp.float32(1.0))
    if i == 0:
        coefficients[0] = mc
        coefficients[1] = ic
        coefficients[2] = rate

def main():
    """Check current defect and independently defined physical acceptance controls."""
    wp.init()
    wp.set_device("cpu")
    a2 = np.pi * (2 + np.pi)
    mc, ic = a2 / (1+a2), 1/(1+a2)
    A = np.array([[1.01, .99], [.99, 1.01]])
    K = A + np.diag(ic/mc*np.diag(A))
    G, b = .001, .02
    physical = np.linalg.solve(K, np.full(2,G))
    recovery = np.linalg.solve(K, np.full(2,b))
    biased = np.linalg.solve(K, np.full(2,G+b))
    assert np.linalg.eigvalsh(A).min() > .019
    np.testing.assert_allclose(biased, physical+recovery, rtol=1e-14)
    single_physical = mc*G/1.01
    inputs = np.r_[biased,recovery,mc*(G+b)/1.01,mc*b/1.01]
    coeff = wp.zeros(3,dtype=wp.float32,device="cpu")
    output = wp.zeros(6,dtype=wp.float32,device="cpu")
    wp.launch(evaluate,dim=6,inputs=[wp.array(inputs,dtype=wp.float32,device="cpu"),
        wp.array(np.full(6,1.01),dtype=wp.float32,device="cpu"),
        wp.array(np.full(6,-b),dtype=wp.float32,device="cpu"),output,coeff],device="cpu")
    native = output.numpy()
    np.testing.assert_allclose(coeff.numpy()[:2],[mc,ic],rtol=2e-7)
    assert np.all(native[:4] == 0), native
    assert abs(native[4]-single_physical) < 3e-9
    assert abs(native[5]) < 3e-9
    assert np.max(abs(native[:2]-physical)) > .00048
    # Full sagittal physical solve: actual separated points, no duplicate normals.
    # generalized velocity=(vx,vz,wy), m=Iyy=1; leverarms=(+/-0.1,0,-0.1).
    J = np.array([[0.,1.,.1],[0.,1.,-.1],[1.,0.,-.1]])
    v_free = np.array([.0001,-G,0.])
    audits={}
    for name,gamma in [("rigid",0.),("native_soft",ic/mc*1.01)]:
        op=J@J.T+np.diag([gamma,gamma,0.])
        impulses=np.linalg.solve(op,-J@v_free)
        normals=impulses[:2]
        tangents=np.full(2,impulses[2]/2)
        assert np.all(normals>0)
        assert np.all(abs(tangents)<.5*normals)
        dv=J.T@impulses
        after=v_free+dv
        assert abs((J@after)[2])<1e-15
        np.testing.assert_allclose(J@after+np.diag([gamma,gamma,0.])@impulses,0,atol=1e-15)
        points=np.array([[-.1,0,-.1],[.1,0,-.1]])
        impulse_vectors=np.column_stack([tangents,np.zeros(2),normals])
        body_P=np.array([dv[0],0,dv[1]])
        body_L=np.array([0,dv[2],0])
        ground_P=-impulse_vectors.sum(0)
        ground_L=-np.cross(points,impulse_vectors).sum(0)
        np.testing.assert_allclose(body_P+ground_P,0,atol=1e-15)
        np.testing.assert_allclose(body_L+ground_L,0,atol=1e-15)
        delta_ke=.5*(after@after-v_free@v_free)
        midpoint_work=.5*(after+v_free)@dv
        assert delta_ke<=0
        assert abs(delta_ke-midpoint_work)<1e-20
        audits[name]=dict(normals=normals.tolist(),tangents=tangents.tolist(),
            final_velocity=after.tolist(),delta_ke_J=float(delta_ke),
            midpoint_work_J=float(midpoint_work),linear_reaction_error=float(np.max(abs(body_P+ground_P))),
            angular_reaction_error=float(np.max(abs(body_L+ground_L))))
    report=dict(status="CURRENT_DEFECT_EXPOSED",production_function="_friction_normal_lambda",
        actual_native_coefficients=coeff.numpy().tolist(),normal_eigenvalues=np.linalg.eigvalsh(A).tolist(),
        coupled_physical=physical.tolist(),coupled_recovery=recovery.tolist(),coupled_biased=biased.tolist(),
        production_loads=native.tolist(),physical_audits=audits,
        future_acceptance="Positive physical load for coupled gravity; exact single-row limit; zero recovery-only friction; actual support reactions and dissipative physical work.",
        scope="Actual production load function and coefficients on CPU. Independent physical reference, not a full production solver test; no recovery replacement implemented.")
    Path("/tmp/colibri_support_load_regression.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if "--require-correct-load" in sys.argv:
        np.testing.assert_allclose(native[:2], physical, rtol=2e-5, atol=3e-9,
                                   err_msg="Coupled physical gravity support was erased")
if __name__=="__main__":
    main()
