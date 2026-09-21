"""Reconstruct a frozen Kamino Schur operator in float64."""

import numpy as np
from scipy.linalg import block_diag

d = np.load("/tmp/colibri_zero_bias_frozen_audit.npz")
n = int(d["problem_dim"][0])
b = int(d["problem_njc"][0])
k = n - b
J = np.zeros((n, len(d["bodies_inv_m"]) * 6))
for (r, c), value in zip(d["jacobian_nzb_coords"][: int(d["jacobian_num_nzb"][0])], d["jacobian_nzb_values"]):
    J[r, c : c + 6] += value
M = block_diag(
    *[block_diag(float(m) * np.eye(3), I.astype(float)) for m, I in zip(d["bodies_inv_m"], d["bodies_inv_I"])]
)
A = J @ M @ J.T
stride = int(d["dvi_bilateral_response_stride"][0])
C = d["dvi_bilateral_coupling"].reshape(b, stride)[:, :k].astype(float)
R = d["dvi_bilateral_response"].reshape(b, stride)[:, :k].astype(float)
print("dims", n, b, k, "coupling_error", np.max(abs(C - A[:b, b:])), flush=True)
# Infer diagonal-only drive/armature regularization from cached B^-1 C.
error = C - A[:b, :b] @ R
regularization = np.sum(error * R, axis=1) / np.maximum(np.sum(R * R, axis=1), 1e-30)
print("inferred_regularization", regularization, flush=True)
B = A[:b, :b] + np.diag(regularization)
response = np.linalg.solve(B, A[:b, b:])
S = A[b:, b:] - A[b:, :b] @ response
f = d["problem_v_f"][:n].astype(float)
q = f[b:] - A[b:, :b] @ np.linalg.solve(B, f[:b])
lam = d["solution_lambdas"][:n].astype(float)
velocity = A @ lam + f
velocity[:b] += regularization * lam[:b]
cached_diag = d["dvi_inequality_projected_diagonal"][b:n]
print("response_relative_error", np.linalg.norm(R - response) / np.linalg.norm(response), flush=True)
print("diagonal_error", np.max(abs(cached_diag - np.diag(S))), flush=True)
print(
    "full_velocity_error",
    np.max(abs(velocity - d["solution_v_plus"][:n])),
    "eq_residual",
    np.max(abs(velocity[:b])),
    flush=True,
)
print(
    "rhs_error", np.max(abs(J @ d["problem_u_f"].ravel() + d["problem_v_b"][:n] + d["problem_v_i"][:n] - f)), flush=True
)
np.savez(
    "/tmp/colibri_zero_bias_schur_reference.npz",
    J=J,
    inverse_mass=M,
    A=A,
    B=B,
    S=S,
    q=q,
    f=f,
    lambdas=lam,
    velocity=velocity,
    regularization=regularization,
    mu=d["problem_mu"][: k // 3],
    cached_response=R,
    cached_coupling=C,
)
