"""Higher precision unsquared joint diagnostic with a certified pure-dual gauge."""

import argparse
import json
import time
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp

parser = argparse.ArgumentParser()
parser.add_argument("--keep-from")
parser.add_argument("--save-mobility", action="store_true")
parser.add_argument("--input", default="/tmp/colibri_global_joint_probe.npz")
parser.add_argument("--dps", type=int, default=45)
parser.add_argument("--gauge", type=int, default=15)
parser.add_argument("--output", default="/tmp/colibri_joint_unsquared_mp")
args = parser.parse_args()
mp.mp.dps = args.dps
d = np.load(args.input)
active = np.flatnonzero(np.any(d["W"] != 0, axis=(1, 2)))
F = np.concatenate((d["J"].reshape(188, -1).T, np.eye(188)[d["compliance"] > 0]), axis=0)
exact = sp.polys.matrices.DomainMatrix.from_Matrix(sp.Matrix([[sp.Rational(float(x)) for x in row] for row in F]))
null = exact.nullspace().to_Matrix()
if null.rows == 1:
    assert null[0, args.gauge] != 0
    keep = [i for i in range(188) if i != args.gauge]
else:
    keep = list(exact.rref()[1])
if args.keep_from:
    keep = json.loads(Path(args.keep_from).read_text())["reference_rows"]
    assert len(exact.extract(range(exact.shape[0]), keep).rref()[1]) == len(keep)
    assert len(keep) == 188 - null.rows
n = len(keep)
J = mp.matrix(d["J"][:, active, :].reshape(188, -1).tolist())
W = mp.zeros(len(active) * 6)
L = mp.zeros(len(active) * 6)
for k, body in enumerate(active):
    block = mp.matrix(d["W"][body].tolist())
    chol = mp.cholesky(block)
    for i in range(6):
        for j in range(6):
            W[6 * k + i, 6 * k + j] = block[i, j]
            L[6 * k + i, 6 * k + j] = chol[i, j]
B = J * L
C = mp.zeros(n, len(active) * 6 + 2)
drive_rows = np.flatnonzero(d["compliance"])
for i, row in enumerate(keep):
    for j in range(B.cols):
        C[i, j] = B[row, j]
    for j, dr in enumerate(drive_rows):
        if row == dr:
            C[i, B.cols + j] = mp.sqrt(float(d["compliance"][row]))
scale = [1 / mp.sqrt(mp.fsum(C[i, j] ** 2 for j in range(C.cols))) for i in range(C.rows)]
for i in range(C.rows):
    for j in range(C.cols):
        C[i, j] *= scale[i]
rhs = mp.matrix([-mp.mpf(float(d["rhs"][row])) * scale[i] for i, row in enumerate(keep)])
start = time.perf_counter()
print("QR start", flush=True)
Q, T = mp.qr(C.T, mode="full" if args.save_mobility else "skinny")
print("QR done", time.perf_counter() - start, flush=True)
a = mp.zeros(n, 1)
for i in range(n):
    a[i] = (rhs[i] - mp.fsum(T[j, i] * a[j] for j in range(i))) / T[i, i]
z = Q[:, :n] * a
ell = mp.zeros(n, 1)
for i in reversed(range(n)):
    ell[i] = (a[i] - mp.fsum(T[i, j] * ell[j] for j in range(i + 1, n))) / T[i, i]
lam = mp.zeros(188, 1)
for i, row in enumerate(keep):
    lam[row] = ell[i] * scale[i]
dv = L * z[: B.cols, :]
response = W * J.T * lam
original = (
    J * dv
    + mp.matrix([mp.mpf(float(x)) * lam[i] for i, x in enumerate(d["compliance"])])
    + mp.matrix(d["rhs"].tolist())
)
out = {
    "precision_decimal_digits": args.dps,
    "exact_gauge_row": args.gauge if null.rows == 1 else None,
    "exact_nullity": null.rows,
    "rhs_convention": "saved rhs is positive residual; correction solves A delta = -rhs",
    "elapsed_seconds": time.perf_counter() - start,
    "original_equation_max_residual": float(max(abs(x) for x in original)),
    "response_max_residual": float(max(abs(x) for x in response - dv)),
    "max_impulse": float(max(abs(x) for x in lam)),
}
v = np.array(list(dv), dtype=float).reshape(-1, 6)
out.update(
    max_dv=float(np.max(np.linalg.norm(v[:, :3], axis=1))), max_dw=float(np.max(np.linalg.norm(v[:, 3:], axis=1)))
)
np.savez(args.output + ".npz", delta=np.array(list(lam), float), dv=v, active=active)
if args.save_mobility:
    complement = Q[:, n:]
    G = L * complement[: B.cols, :]
    slack = complement[B.cols :, :]
    nullcheck = B * complement[: B.cols, :]
    for row in range(188):
        for k, drive_row in enumerate(drive_rows):
            if row == drive_row:
                for column in range(nullcheck.cols):
                    nullcheck[row, column] += mp.sqrt(float(d["compliance"][row])) * slack[k, column]
    out["complement_constraint_residual"] = float(max(abs(x) for x in nullcheck))
    out["complement_columns"] = complement.cols
    np.savez(
        args.output + "_mobility.npz",
        G=np.array(G.tolist(), float),
        slack=np.array(slack.tolist(), float),
        active=active,
        drive_rows=drive_rows,
        R=d["compliance"],
    )
Path(args.output + ".json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2), flush=True)
