"""Bounded CPU physical-coordinate Coulomb face solve; no live installation."""
# ruff: noqa: TID253 -- standalone optional-dependency numerical reference.
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from local_studies.colibri.coulomb_semismooth import natural_map_evaluator
from local_studies.colibri.coupled_support_online import assemble_snapshot


def factor(a):
    """Retain all seven physical modes after exactly five hard joint equations."""
    l = np.linalg.cholesky(a["W"])
    hard = a["diagonal"] == 0
    _, singular, vt = np.linalg.svd(a["B"][hard] @ l, full_matrices=True)
    assert singular[-1] > 1e-6
    g = l @ vt[len(singular):].T
    dynamic = a["B"][~hard] @ g
    reduced = np.eye(g.shape[1]) + dynamic.T @ (dynamic / a["diagonal"][~hard, None])
    g = np.linalg.solve(np.linalg.cholesky(reduced), g.T).T
    f = a["C"] @ g
    assert np.max(abs(f @ f.T - a["A"])) < 1e-9
    return f, g


def seed_sweeps(a, seed, sweeps):
    """Original-order exact local metric Coulomb updates, for bounded seeding."""
    lam = seed.copy()
    for _ in range(sweeps):
        for p in range(len(a["mu"])):
            row = 3*p
            gradient = a["A"] @ lam + a["rhs"]
            lam[row] = max(0., lam[row] - (gradient[row] + a["gamma"][p]*lam[row]) /
                           (a["A"][row,row] + a["gamma"][p]))
            ids = slice(row+1,row+3)
            gradient = a["A"] @ lam + a["rhs"]
            matrix = a["A"][ids,ids]
            force = matrix @ lam[ids] - gradient[ids]
            radius = a["mu"][p]*lam[row]
            if radius == 0:
                lam[ids] = 0
                continue
            tangent = np.linalg.solve(matrix, force)
            if np.linalg.norm(tangent) > radius:
                def error(shift):
                    return np.linalg.norm(np.linalg.solve(matrix+shift*np.eye(2),force))-radius
                upper = max(float(np.linalg.norm(matrix)), 1.)
                for _ in range(60):
                    if error(upper) <= 0:
                        break
                    upper *= 2
                shift = brentq(error,0.,upper,xtol=1e-12)
                tangent = np.linalg.solve(matrix+shift*np.eye(2),force)
            lam[ids] = tangent
    return lam


def modes(a, seed):
    """Classify the current natural-map face, without using a solution oracle."""
    _, scale, operator, _ = natural_map_evaluator(a["A"],a["rhs"],a["gamma"],a["mu"])
    trial = (seed-(operator@seed+a["rhs"])/scale).reshape(-1,3)
    active = trial[:,0] > 0
    stick = active & (a["mu"]>0) & (np.linalg.norm(trial[:,1:],axis=1) <= a["mu"]*np.maximum(seed[::3],0))
    return active, stick


def solve_face(a, f, seed, active, stick):
    """Eliminate compliant normals and slipping tangents; retain hard multipliers."""
    dimension = f.shape[1]
    normal_slot, tangent_slot = {}, {}
    cursor = dimension
    for p in np.flatnonzero(active):
        if a["gamma"][p] == 0:
            normal_slot[p] = cursor
            cursor += 1
        if stick[p]:
            tangent_slot[p] = cursor
            cursor += 2
    x = np.zeros(cursor)
    x[:dimension] = f.T @ seed
    for p, slot in normal_slot.items():
        x[slot] = seed[3*p]
    for p, slot in tangent_slot.items():
        x[slot:slot+2] = seed[3*p+1:3*p+3]

    def evaluate(x):
        gradient = (f@x[:dimension]+a["rhs"]).reshape(-1,3)
        impulse = np.zeros(3*len(active))
        derivative = np.zeros((len(impulse),cursor))
        equations, jacobian = [], []
        for p in np.flatnonzero(active):
            row = 3*p
            if p in normal_slot:
                slot = normal_slot[p]
                impulse[row] = x[slot]
                derivative[row,slot] = 1
                eq = np.zeros(cursor); eq[:dimension] = f[row]
                equations.append(gradient[p,0]); jacobian.append(eq)
            else:
                impulse[row] = -gradient[p,0]/a["gamma"][p]
                derivative[row,:dimension] = -f[row]/a["gamma"][p]
            if stick[p]:
                slot = tangent_slot[p]
                impulse[row+1:row+3] = x[slot:slot+2]
                derivative[row+1:row+3,slot:slot+2] = np.eye(2)
                for j in (1,2):
                    eq = np.zeros(cursor); eq[:dimension] = f[row+j]
                    equations.append(gradient[p,j]); jacobian.append(eq)
            elif a["mu"][p] > 0:
                length = np.linalg.norm(gradient[p,1:])
                if length < 1e-15:
                    raise ValueError("Sliding direction undefined on this face")
                direction = gradient[p,1:]/length
                impulse[row+1:row+3] = -a["mu"][p]*impulse[row]*direction
                derivative[row+1:row+3] = -a["mu"][p]*np.outer(direction,derivative[row])
                derivative[row+1:row+3,:dimension] -= (
                    a["mu"][p]*impulse[row]/length*(np.eye(2)-np.outer(direction,direction))@f[row+1:row+3]
                )
        residual = np.r_[x[:dimension]-f.T@impulse,equations]
        top = -f.T@derivative; top[:,:dimension] += np.eye(dimension)
        jac = np.vstack([top,np.asarray(jacobian).reshape(-1,cursor)])
        return residual,jac,impulse

    history = []
    for iteration in range(12):
        residual, jac, lam = evaluate(x)
        error = float(np.max(abs(residual)))
        if error < 1e-12:
            break
        if cursor > 32:
            return lam,dict(accepted_face=False,reason="More than32unknowns; stop before dense redundant dual solve",unknowns=cursor)
        try:
            delta = np.linalg.solve(jac,-residual)
        except np.linalg.LinAlgError:
            return lam,dict(accepted_face=False,reason="Singular face Jacobian; no physical rank truncation",unknowns=cursor)
        linear = float(np.max(abs(jac@delta+residual)))
        if linear > 1e-9:
            return lam,dict(accepted_face=False,reason="Linear backward error",linear_error=linear,unknowns=cursor)
        accepted = False
        for backtrack in range(20):
            alpha = 2.**(-backtrack)
            trial = x+alpha*delta
            rr,_,_ = evaluate(trial)
            if rr@rr <= (1-1e-4*alpha)*(residual@residual):
                x = trial; accepted = True; break
        history.append(dict(iteration=iteration,error=error,linear_error=linear,alpha=alpha,accepted=accepted))
        if not accepted:
            break
    residual,jac,lam = evaluate(x)
    return lam,dict(accepted_face=bool(np.max(abs(residual))<1e-10),face_error=float(np.max(abs(residual))),
                    unknowns=cursor,history=history,jacobian_condition=float(np.linalg.cond(jac)))


def audit(a,lam):
    evaluate,*_ = natural_map_evaluator(a["A"],a["rhs"],a["gamma"],a["mu"])
    joint = np.linalg.solve(a["K"],a["targets"]-a["B"]@a["free"]-a["B"]@a["W"]@a["C"].T@lam)
    impulse = a["C"].T@lam+a["B"].T@joint
    velocity = a["free"]+a["W"]@impulse
    gradient = (a["A"]@lam+a["rhs"]).reshape(-1,3)
    triples = lam.reshape(-1,3)
    normal = gradient[:,0]+a["gamma"]*triples[:,0]
    report = dict(natural_error=float(np.max(abs(evaluate(lam)[0]))),minimum_normal=float(triples[:,0].min()),
        minimum_normal_gradient=float(normal.min()),normal_complementarity=float(np.max(abs(normal*triples[:,0]))),
        cone_violation=float(np.max(np.linalg.norm(triples[:,1:],axis=1)-a["mu"]*triples[:,0])),
        joint_error=float(np.max(abs(a["B"]@velocity+a["diagonal"]*joint-a["targets"]))),
        physical_scatter_error=float(np.max(abs(np.linalg.solve(a["W"],velocity-a["free"])-impulse))),
        impulse_work=float(.5*(velocity+a["free"])@impulse),
        kinetic_change=float(.5*(velocity@np.linalg.solve(a["W"],velocity)-a["free"]@np.linalg.solve(a["W"],a["free"]))),
    )
    report["accepted"] = bool(report["natural_error"]<1e-8 and report["minimum_normal"]>=-1e-10 and report["cone_violation"]<1e-8)
    return report


def main():
    z = np.load("/tmp/colibri_base_frame_totalnormal_phases330.npz")
    records=[]
    for phase in ("biased","relax"):
        d={k.split(".",1)[1]:z[k] for k in z.files if k.startswith(phase+"_solved.")}
        a=assemble_snapshot(d,phase,float(z["dt"][0]),int(z["num_joints"][0]))
        f,g=factor(a)
        reference=np.load(f"/tmp/colibri_two_body_condensed_gpu.{phase}32.npz")["lam"]
        for label,seed in (("native",a["old"]),("two_sweep",seed_sweeps(a,a["old"],2)),("reference_face_only",reference)):
            active,stick=modes(a,seed)
            start = a["old"] if label=="reference_face_only" else seed
            try:
                lam,report=solve_face(a,f,start,active,stick)
                checked=audit(a,lam)
            except (ValueError,np.linalg.LinAlgError) as exc:
                report=dict(accepted_face=False,reason=str(exc));checked={"accepted":False}
            record=dict(phase=phase,seed=label,active=np.flatnonzero(active).tolist(),sticking=np.flatnonzero(stick).tolist(),
                        solve=report,audit=checked,oracle_face=label=="reference_face_only")
            records.append(record)
            if checked["accepted"]:
                np.savez(f"/tmp/colibri_physical_face_{phase}_{label}.npz",lam=lam,F=f,G=g)
    Path("/tmp/colibri_physical_face_support.json").write_text(json.dumps(records,indent=2))
    print(json.dumps(records,indent=2))


if __name__=="__main__":
    main()
