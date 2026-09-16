# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Benchmark vendor GPU QR on the frozen joint matrix, not a live solver."""

import json
from pathlib import Path

import numpy as np


def main():
    """Measure complete QR and independently check its constrained subspace."""
    import scipy.linalg as la
    import torch

    torch.set_num_threads(1)
    p = dict(np.load("/tmp/colibri_joint_reformed_input.npz"))
    m = dict(np.load("/tmp/colibri_joint_fp64_mobility.npz"))
    keep = json.loads(Path("/tmp/colibri_joint_pose1_rank.json").read_text())["reference_rows"]
    active = m["active"]
    drives = m["drive_rows"]
    J = p["J"][:, active].reshape(len(p["rhs"]), -1)
    L = la.block_diag(*(la.cholesky(p["W"][body], lower=True) for body in active))
    C = np.column_stack((J @ L, np.diag(np.sqrt(p["compliance"]))[:, drives]))
    scale = 1 / np.linalg.norm(C[keep], axis=1)
    matrix = (C[keep] * scale[:, None]).T
    a = torch.tensor(matrix, dtype=torch.float64, device="cuda")
    q = torch.empty((len(matrix), len(matrix)), dtype=a.dtype, device=a.device)
    r = torch.empty_like(a)
    for _ in range(10):
        torch.linalg.qr(a, mode="complete", out=(q, r))
    samples = []
    for _ in range(7):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(50):
            torch.linalg.qr(a, mode="complete", out=(q, r))
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 50)
    graph_result = {}
    try:
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            torch.linalg.qr(a, mode="complete", out=(q, r))
        for _ in range(10):
            graph.replay()
        graph_samples = []
        for _ in range(7):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(50):
                graph.replay()
            end.record()
            end.synchronize()
            graph_samples.append(start.elapsed_time(end) / 50)
        graph_result = {"captured": True, "batch_mean_ms": graph_samples, "median_ms": float(np.median(graph_samples))}
    except RuntimeError as error:
        graph_result = {"captured": False, "error": str(error)}
    Q, T = q.cpu().numpy(), r.cpu().numpy()
    n = len(keep)
    Z = Q[:, n:]
    G = L @ Z[: len(L)]
    H = G @ G.T
    Href = m["G"] @ m["G"].T
    reconstruction = np.linalg.norm(Q @ T - matrix) / np.linalg.norm(matrix)
    orthogonal = np.max(np.abs(Q.T @ Q - np.eye(len(Q))))
    subspace_residual = np.max(np.abs(C @ Z))
    y = la.solve_triangular(T[:n, :n].T, -p["rhs"][keep] * scale, lower=True)
    z = Q[:, :n] @ y
    delta = np.zeros(len(J))
    delta[keep] = scale * la.solve_triangular(T[:n, :n], y)
    dv = L @ z[: len(L)]
    equation = J @ dv + p["compliance"] * delta + p["rhs"]
    result = {
        "scope": "Frozen 218x184 complete QR only; excludes geometry, rank certification, mass assembly, contact solve and state scatter",
        "backend": "torch.linalg.qr CUDA vendor backend",
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "matrix_shape": list(matrix.shape),
        "cuda_graph": graph_result,
        "batch_mean_ms": samples,
        "median_ms": float(np.median(samples)),
        "two_factorizations_per_60Hz_frame_ms": float(2 * np.median(samples)),
        "sixty_factorizations_per_60Hz_frame_ms": float(60 * np.median(samples)),
        "relative_reconstruction_error": float(reconstruction),
        "max_orthogonality_error": float(orthogonal),
        "all_row_subspace_residual": float(subspace_residual),
        "relative_physical_mobility_error_vs_cpu": float(np.linalg.norm(H - Href) / np.linalg.norm(Href)),
        "original_equation_max_residual": float(np.max(np.abs(equation))),
    }
    assert result["original_equation_max_residual"] < 1e-8, result
    assert reconstruction < 1e-12 and orthogonal < 1e-12, result
    Path("/tmp/colibri_joint_qr_gpu_benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
