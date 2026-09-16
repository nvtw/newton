"""Frozen physical six-contact Base/ground reference, without GPU initialization."""
import json
from pathlib import Path
import numpy as np
from local_studies.colibri.coulomb_semismooth import solve_coulomb, natural_map_evaluator

def main():
    a=np.load("/tmp/colibri_support_phases330.npz")
    def ph(p):return {k.split(".",1)[1]:a[k] for k in a.files if k.startswith(p+".")}
    before,solved,after=map(ph,["biased_before","biased_solved","biased_averaged"])
    h=solved["headers"].view(np.int32);d=solved["derived"];l=solved["lambdas"]
    W=np.zeros((6,6));W[:3,:3]=np.eye(3)*solved["inverse_mass"][1]
    xx,yy,zz,xy,xz,yz=solved["inverse_inertia"][1].astype(float)
    W[3:,3:]=[[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]]
    M=np.linalg.inv(W)
    points={};selected=[]
    for col in range(int(solved["column_count"][0])):
        b0,b1=h[1:3,col]
        if {b0,b1}!={0,1}:continue
        for k in range(h[5,col],h[5,col]+h[6,col]):
            n=l[:3,k].astype(float);t=l[3:6,k].astype(float)
            axes=np.array([n,t,np.cross(n,t)])
            r=d[9:12,k].astype(float) if b0==1 else d[12:15,k].astype(float)
            J=np.c_[axes,np.cross(r,axes)]*(-1 if b0==1 else 1)
            lam=solved["impulses"][:,k].astype(float)
            rg=d[9:12,k].astype(float) if b0==0 else d[12:15,k].astype(float)
            Jg=np.c_[axes,np.cross(rg,axes)]*(-1 if b0==0 else 1)
            points[k]=(J,r,lam,float(d[3,k]),float(solved["headers"][3,col]),Jg)
            if d[3,k]<=0 and lam[0]>1e-10:selected.append(k)
    assert len(selected)==6,selected
    J=np.vstack([points[k][0] for k in selected])
    initial=np.concatenate([points[k][2] for k in selected])
    mu=np.array([points[k][4] for k in selected])
    def base_twist(state):
        start=int(state["copy_section_end"][0]);end=int(state["copy_section_end"][1])
        if end>start:
            return np.r_[state["copy_velocity"][start:end].astype(float).mean(0),state["copy_angular_velocity"][start:end].astype(float).mean(0)]
        return np.r_[state["velocity"][1],state["angular_velocity"][1]].astype(float)
    vafter=base_twist(after)
    # Remove ONLY selected total lambda; all other captured reactions stay fixed.
    free=vafter-W@J.T@initial
    # Independent reconstruction from pre-phase physical state and all lambda deltas.
    vb=base_twist(before)
    joint=np.zeros(6);jd=solved["joint_data"].view(np.int32)
    for j in range(int(a["num_joints"][0])):
        for side,key in [(0,"joint_wrench0"),(1,"joint_wrench1")]:
            if jd[1+side,j]!=1:continue
            st=solved["joint_structural_index"][j]
            for i in range(solved["joint_row_count"][j]):
                row=solved["joint_row_indices"][j,i];local=solved["joint_row_local"][row]
                joint+=solved[key][st,local].astype(float)*(float(solved["joint_accumulated"][row])-float(before["joint_accumulated"][row]))
    contact_delta=np.zeros(6)
    # Include all incident contact pairs, not just ground, in reconstruction.
    for col in range(int(solved["column_count"][0])):
        b0,b1=h[1:3,col]
        if 1 not in (b0,b1):continue
        for k in range(h[5,col],h[5,col]+h[6,col]):
            n=l[:3,k].astype(float);t=l[3:6,k].astype(float);axes=np.array([n,t,np.cross(n,t)])
            r=d[9:12,k] if b0==1 else d[12:15,k]
            jrow=np.c_[axes,np.cross(r,axes)]*(-1 if b0==1 else 1)
            contact_delta+=jrow.T@(solved["impulses"][:,k].astype(float)-before["impulses"][:,k].astype(float))
    reconstruction=vb+W@(joint+contact_delta)
    error=float(np.max(abs(reconstruction-vafter)))
    assert error<2e-6,error
    A=J@W@J.T;rhs=J@free
    mc=.9417003989219666;ic=.05829954519867897
    reports={}
    for name,regularization in [("rigid_unbiased",np.zeros(6)),("native_soft_unbiased",ic/mc*np.diag(A)[::3])]:
        value,tries=solve_coulomb(A,rhs,initial,regularization,mu,residual_scale=1e4)
        if name=="rigid_unbiased":
            from scipy.optimize import linprog
            rays=np.zeros((18,6*64))
            for i in range(6):
                for j in range(64):
                    angle=2*np.pi*j/64
                    rays[3*i:3*i+3,i*64+j]=[1,mu[i]*np.cos(angle),mu[i]*np.sin(angle)]
            B=W@J.T@rays*1e-4
            scales=np.maximum(np.linalg.norm(B,axis=1),abs(free))
            lp=linprog(np.ones(6*64),A_eq=B/scales[:,None],b_eq=-free/scales,
                       bounds=(0,None),method="highs",
                       options={"primal_feasibility_tolerance":1e-10,"dual_feasibility_tolerance":1e-10})
            tries.append(dict(method="inscribed_cone_sticking_feasibility_64_rays",success=lp.success,
                              scope="Existence certificate only; exact circular cone and original velocity equations audited afterward"))
            if lp.success:
                np.savez("/tmp/colibri_frozen_base_rigid_naturalmap_rejected.npz",solution=value.copy())
                value=rays@lp.x*1e-4
        evaluate,_,op,_=natural_map_evaluator(A,rhs,regularization,mu)
        residual=float(np.max(abs(evaluate(value)[0])))
        v=free+W@J.T@value
        triples=value.reshape(-1,3);vel=(J@v).reshape(-1,3)
        impulse=J.T@value
        ground_impulse=sum((points[k][5].T@triples[i] for i,k in enumerate(selected)),np.zeros(6))
        ground_linear=ground_impulse[:3]
        ground_angular=ground_impulse[3:]+np.cross(solved["position"][0],ground_linear)
        body_angular=impulse[3:]+np.cross(solved["position"][1],impulse[:3])
        ke=.5*(v@M@v-free@M@free);work=.5*(v+free)@impulse
        omitted={str(k):(points[k][0]@v).tolist() for k in points if k not in selected and points[k][3]<=0}
        reports[name]=dict(residual=residual,tries=tries,impulses=triples.tolist(),velocity=v.tolist(),
            max_tangent_speed=float(np.linalg.norm(vel[:,1:],axis=1).max()),
            normal_total_Ns=float(triples[:,0].sum()),friction_capacity_sum_Ns=float(mu@triples[:,0]),
            cone_violation=float(np.max(np.linalg.norm(triples[:,1:],axis=1)-mu*triples[:,0])),
            energy_change_J=float(ke),midpoint_work_J=float(work),work_error_J=float(abs(ke-work)),
            P_error=float(np.max(abs(impulse[:3]+ground_linear))),L_error=float(np.max(abs(body_angular+ground_angular))),
            omitted_non_spec_velocities=omitted,min_omitted_normal=float(min(x[0] for x in omitted.values())))
        np.savez("/tmp/colibri_frozen_base_"+name+".npz",J=J,W=W,rhs=rhs,regularization=regularization,initial=initial,solution=value,free=free,velocity=v,points=np.array(selected))
    result=dict(source="/tmp/colibri_support_phases330.npz",selected_points=selected,
        eligibility="Exactly six native-loaded contacts with bias<=0. All other52-6 ground impulses remain fixed, including speculative rows.",
        external_joint_impulse=joint.tolist(),native_after=vafter.tolist(),free_twist=free.tolist(),
        phase_reconstruction_max_error=error,results=reports,
        scope="Frozen contact correction with captured joint loading held fixed. Not whole-assembly equilibrium or live stabilization proof. Audit omitted eligible rows before any full-manifold acceptance.")
    Path("/tmp/colibri_frozen_base_support_reference.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
