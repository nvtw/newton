# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Dedicated frozen tree Cholesky/solve/QR benchmark; no production changes."""

import json
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri.benchmark_small_tree_qr import small_qr


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    __shared__ double lower[46*47], rhs[46*14], diag;
    __shared__ int valid;
    int lane=threadIdx.x;
    if(lane==0) valid=1;
    for(int i=lane;i<46*46;i+=256) lower[(i/46)*47+i%46]=mass.data[i];
    for(int i=lane;i<46*14;i+=256) rhs[i]=rows.data[(i%14)*46+i/14];
    __syncthreads();
    for(int k=0;k<46;++k) {
        if(lane==0) {
            double pivot=lower[k*47+k];
            if(!(pivot>0.0) || !isfinite(pivot)) valid=0;
            diag=sqrt(pivot);
            lower[k*47+k]=diag;
        }
        __syncthreads();
        if(lane>k && lane<46) lower[lane*47+k]/=diag;
        __syncthreads();
        for(int i=lane;i<46*46;i+=256) {
            int r=i/46,c=i%46;
            if(r>=c && c>k) lower[r*47+c]-=lower[r*47+k]*lower[c*47+k];
        }
        if(lane<14) rhs[k*14+lane]/=diag;
        __syncthreads();
        for(int i=lane;i<46*14;i+=256) {
            int r=i/14,c=i%14;
            if(r>k) rhs[i]-=lower[r*47+k]*rhs[k*14+c];
        }
        __syncthreads();
    }
    int warp=lane/32, local=lane%32;
    for(int col=warp;col<14;col+=8) {
        double norm=0.0;
        for(int row=local;row<48;row+=32) {
            double value=(row<46)?rhs[row*14+col]:drive.data[col*2+row-46];
            norm+=value*value;
        }
        for(int offset=16;offset>0;offset/=2) norm+=__shfl_down_sync(0xffffffffu,norm,offset);
        norm=sqrt(__shfl_sync(0xffffffffu,norm,0));
        if(local==0 && (!(norm>0.0) || !isfinite(norm))) atomicExch(&valid,0);
        for(int row=local;row<48;row+=32) {
            double value=(row<46)?rhs[row*14+col]:drive.data[col*2+row-46];
            output.data[row*14+col]=value/norm;
        }
    }
    for(int i=lane;i<46*46;i+=256) cholout.data[i]=lower[(i/46)*47+i%46];
    __syncthreads();
    if(lane==0) status.data[0]=valid;
    __syncthreads();
#endif
""")
def prepare(
    mass: wp.array(dtype=wp.float64),
    rows: wp.array(dtype=wp.float64),
    drive: wp.array(dtype=wp.float64),
    output: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    cholout: wp.array(dtype=wp.float64),
): ...


@wp.kernel
def prepare_and_factor(
    mass: wp.array(dtype=wp.float64),
    rows: wp.array(dtype=wp.float64),
    drive: wp.array(dtype=wp.float64),
    output: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    cholout: wp.array(dtype=wp.float64),
    q: wp.array(dtype=wp.float64),
    r: wp.array(dtype=wp.float64),
):
    prepare(mass, rows, drive, output, status, cholout)
    if status[0] == 1:
        small_qr(output, q, r)


def main():
    """Check all four physical poses and isolate captured factor setup cost."""
    import torch

    wp.init()
    output = wp.empty(48 * 14, dtype=wp.float64, device="cuda:0")
    status = wp.empty(1, dtype=wp.int32, device="cuda:0")
    q = wp.empty(48 * 48, dtype=wp.float64, device="cuda:0")
    r = wp.empty(48 * 14, dtype=wp.float64, device="cuda:0")
    cholout = wp.empty(46 * 46, dtype=wp.float64, device="cuda:0")
    checks = []
    for suffix in ("_pose1", "_pose30", "_pose120", ""):
        path = f"/tmp/colibri_joint_tree{suffix}_mobility.npz"
        data = dict(np.load(path))
        inputs = [
            wp.array(data[key].ravel(), dtype=wp.float64, device="cuda:0")
            for key in ("reduced_mass", "loop_velocity_rows", "drive_sqrt_rows")
        ]
        args = [*inputs, output, status, cholout, q, r]
        wp.launch(prepare_and_factor, dim=256, inputs=args, block_dim=256, device="cuda:0")
        assert status.numpy()[0] == 1, path
        actual = output.numpy().reshape(48, 14)
        Q, R = q.numpy().reshape(48, 48), r.numpy().reshape(48, 14)
        relative = np.linalg.norm(actual - data["qr_input"]) / np.linalg.norm(data["qr_input"])
        reconstruction = np.linalg.norm(Q @ R - actual) / np.linalg.norm(actual)
        orthogonality = np.max(np.abs(Q.T @ Q - np.eye(48)))
        complement = Q[:, 14:]
        actual_chol = np.tril(cholout.numpy().reshape(46, 46))
        actual_basis = data["N"] @ np.linalg.solve(actual_chol.T, np.eye(46))
        G = actual_basis @ complement[:46]
        mass_error = np.linalg.norm(actual_chol @ actual_chol.T - data["reduced_mass"]) / np.linalg.norm(
            data["reduced_mass"]
        )
        assert mass_error < 1e-12
        reference = data["G"] @ data["G"].T
        mobility_error = np.linalg.norm(G @ G.T - reference) / np.linalg.norm(reference)
        all_rows = np.max(np.abs(data["K_all"] @ complement))
        assert relative < 1e-10 and reconstruction < 1e-12 and orthogonality < 1e-12, path
        assert mobility_error < 1e-7 and all_rows < 1e-8, path
        checks.append(
            {
                "input": path,
                "qr_input_error": float(relative),
                "reconstruction": float(reconstruction),
                "orthogonality": float(orthogonality),
                "mass_reconstruction_error": float(mass_error),
                "mobility_error": float(mobility_error),
                "all_row_residual": float(all_rows),
            }
        )
    with wp.ScopedCapture(device="cuda:0") as capture:
        for _ in range(100):
            wp.launch(prepare_and_factor, dim=256, inputs=args, block_dim=256, device="cuda:0")
    stream = torch.cuda.ExternalStream(wp.get_stream("cuda:0").cuda_stream)
    samples = []
    for _ in range(7):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(stream)
        wp.capture_launch(capture.graph)
        end.record(stream)
        end.synchronize()
        samples.append(start.elapsed_time(end) / 100)
    assert np.array_equal(Q, q.numpy().reshape(48, 48))
    report = {
        "scope": "Frozen mass Cholesky + solve + scaling + full QR; excludes geometry/mass assembly/contacts/scatter",
        "pose_checks": checks,
        "samples_ms": samples,
        "median_ms": float(np.median(samples)),
        "repeat_byte_identical": True,
    }
    Path("/tmp/colibri_tree_prepare_gpu.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
