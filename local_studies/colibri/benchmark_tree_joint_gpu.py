"""Frozen actual tree factor benchmark; no live solver performance claim."""

import json
from pathlib import Path

import numpy as np


def main():
    import torch

    torch.set_num_threads(1)
    data = dict(np.load("/tmp/colibri_joint_tree_mobility.npz"))
    mass, rows, drive = [
        torch.tensor(data[key], dtype=torch.float64, device="cuda")
        for key in ("reduced_mass", "loop_velocity_rows", "drive_sqrt_rows")
    ]

    def factor():
        chol, info = torch.linalg.cholesky_ex(mass, check_errors=False)
        velocity = torch.linalg.solve_triangular(chol, rows.T, upper=False).T
        matrix = torch.cat((velocity, drive), dim=1)
        matrix = (matrix / torch.linalg.vector_norm(matrix, dim=1)[:, None]).T
        q, r = torch.linalg.qr(matrix, mode="complete")
        return matrix, q, r, info

    breakdown = {}
    chol0 = torch.linalg.cholesky_ex(mass, check_errors=False)[0]
    qr0 = torch.tensor(data["qr_input"], dtype=torch.float64, device="cuda")
    for name, operation in (
        ("cholesky", lambda: torch.linalg.cholesky_ex(mass, check_errors=False)),
        ("triangular_solve", lambda: torch.linalg.solve_triangular(chol0, rows.T, upper=False)),
        ("complete_qr", lambda: torch.linalg.qr(qr0, mode="complete")),
        ("reduced_qr", lambda: torch.linalg.qr(qr0, mode="reduced")),
    ):
        for _ in range(10):
            operation()
        piece_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(piece_graph):
            operation()
        piece_samples = []
        for _ in range(7):
            begin, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(100):
                piece_graph.replay()
            finish.record()
            finish.synchronize()
            piece_samples.append(begin.elapsed_time(finish) / 100)
        breakdown[name] = {"median_ms": float(np.median(piece_samples)), "samples_ms": piece_samples}

    for _ in range(10):
        output = factor()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = factor()
    samples = []
    for _ in range(7):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 100)
    matrix, q, r, info = [x.cpu().numpy() for x in output]
    error = np.linalg.norm(matrix - data["qr_input"]) / np.linalg.norm(data["qr_input"])
    reconstruction = np.linalg.norm(q @ r - matrix) / np.linalg.norm(matrix)
    orthogonality = np.max(np.abs(q.T @ q - np.eye(len(q))))
    assert info == 0 and error < 1e-10
    assert reconstruction < 1e-12 and orthogonality < 1e-12
    result = {
        "scope": "Frozen actual 46x46 mass Cholesky, triangular solve, assembly/scaling and complete 48x14 QR. Excludes tree construction, mass assembly, reactions, contacts and scatter.",
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "breakdown": breakdown,
        "graph_batch_mean_ms": samples,
        "median_ms": float(np.median(samples)),
        "sixty_factors_ms": float(60 * np.median(samples)),
        "input_relative_error": float(error),
        "reconstruction_error": float(reconstruction),
        "orthogonality_error": float(orthogonality),
    }
    Path("/tmp/colibri_tree_gpu_breakdown.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
