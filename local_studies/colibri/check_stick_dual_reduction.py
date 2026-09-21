"""Certified redundant sticking equations and cone-feasible impulse redistribution."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize

from local_studies.colibri.physical_face_support import audit, factor, modes


def main():
    """Retain five resolved sticking modes and certify every original equation."""
    a = dict(np.load("/tmp/colibri_analytical_coupled10.rejected.npz"))
    f, g = factor(a)
    _, stick = modes(a, a["solution"])
    ids = np.flatnonzero(stick)
    rows = np.array([3 * p + j for p in ids for j in (1, 2)])
    e = f[rows]
    q = a["rhs"][rows]
    u, s, vt = np.linalg.svd(e, full_matrices=True)
    # Five is the physical tangent-response dimension for this captured subset;
    # both small resolved modes are retained. Certify all omitted equations.
    basis = vt[:5]
    null = vt[5:].T
    yp = -basis.T @ ((u[:, :5].T @ q) / s[:5])
    assert np.max(abs(e @ yp + q)) < 1e-12
    assert np.max(abs(e @ null)) < 1e-12
    normal = np.arange(0, len(a["rhs"]), 3)

    def response(x):
        y = yp + null @ x
        gradient = (f @ y + a["rhs"]).reshape(-1, 3)
        lam = np.zeros_like(gradient)
        for p in range(len(lam)):
            if a["gamma"][p] > 0:
                lam[p, 0] = max(0.0, -gradient[p, 0] / a["gamma"][p])
            else:
                assert gradient[p, 0] > 0
            if not stick[p]:
                length = np.linalg.norm(gradient[p, 1:])
                if length > 0:
                    lam[p, 1:] = -a["mu"][p] * lam[p, 0] * gradient[p, 1:] / length
        return null.T @ (y - f.T @ lam.ravel()), y, lam

    initial = null.T @ (f.T @ a["solution"] - yp)
    result = least_squares(
        lambda x: response(x)[0] * 1e6, initial, jac="3-point", xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=100
    )
    residual, y, lam = response(result.x)
    needed = y - f.T @ lam.ravel()
    tangent_map = f[rows].T
    # Orthonormal physical coordinates remove duplicate equality equations;
    # the final full response residual is checked independently.
    eq = basis @ tangent_map
    target = basis @ needed
    scale = max(np.linalg.norm(target), 1e-8)
    radii = a["mu"][ids] * lam[ids, 0] / scale
    trial0 = np.linalg.lstsq(eq, target / scale, rcond=None)[0]
    constraints = [
        dict(type="eq", fun=lambda x: eq @ x - target / scale, jac=lambda x: eq),
        dict(type="ineq", fun=lambda x: radii * radii - np.sum(x.reshape(-1, 2) ** 2, axis=1)),
    ]
    cone = minimize(
        lambda x: 0.5 * np.dot(x, x),
        trial0,
        jac=lambda x: x,
        method="SLSQP",
        constraints=constraints,
        options=dict(maxiter=200, ftol=1e-14),
    )
    lam[ids, 1:] = cone.x.reshape(-1, 2) * scale
    report = dict(
        stick=ids.tolist(),
        singular=s.tolist(),
        compatibility=float(np.max(abs(e @ yp + q))),
        null_response=float(np.max(abs(e @ null))),
        physical_residual=float(np.max(abs(residual))),
        evaluations=int(result.nfev),
        cone_status=str(cone.message),
        full_response=float(np.max(abs(y - f.T @ lam.ravel()))),
        audit=audit(a, lam.ravel()),
    )
    certificate = []
    radii_actual = a["mu"][ids] * lam[ids, 0]
    for direction in basis:
        target = float(direction @ needed)
        bound = float(np.sum(radii_actual * np.linalg.norm((e @ direction).reshape(-1, 2), axis=1)))
        certificate.append(dict(required=target, cone_support_bound=bound, violates=bool(abs(target) > bound)))
    report["cone_support_certificates"] = certificate
    Path("/tmp/colibri_stick_dual_reduction.json").write_text(json.dumps(report, indent=2))
    np.savez("/tmp/colibri_stick_dual_reduction.npz", solution=lam.ravel(), y=y, f=f, stick=stick)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
