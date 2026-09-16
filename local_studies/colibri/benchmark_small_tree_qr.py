# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Frozen 48x14 FP64 Householder QR prototype, launched with exactly 256 lanes."""

import json
from pathlib import Path

import numpy as np
import warp as wp


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    __shared__ double a[48*14], q[48*49], v[48], beta, alpha;
    int lane = threadIdx.x;
    for (int i=lane; i<48*14; i+=256) a[i]=input.data[i];
    for (int i=lane; i<48*48; i+=256) q[(i/48)*49+i%48]=(i/48 == i%48)?1.0:0.0;
    __syncthreads();
    for (int k=0; k<14; ++k) {
        if(lane<32) {
            double norm=0.0;
            for(int i=lane; i<48; i+=32) if(i>=k) norm+=a[i*14+k]*a[i*14+k];
            for(int offset=16; offset>0; offset/=2)
                norm+=__shfl_down_sync(0xffffffffu,norm,offset);
            if(lane==0) alpha=-copysign(sqrt(norm),a[k*14+k]);
        }
        __syncthreads();
        if(lane<48) {
            v[lane]=(lane<k)?0.0:a[lane*14+k];
            if(lane==k) v[lane]-=alpha;
        }
        __syncthreads();
        if(lane<32) {
            double vv=0.0;
            for(int i=lane; i<48; i+=32) vv+=v[i]*v[i];
            for(int offset=16; offset>0; offset/=2)
                vv+=__shfl_down_sync(0xffffffffu,vv,offset);
            if(lane==0) beta=(vv==0.0)?0.0:2.0/vv;
        }
        __syncthreads();
        int warp=lane/32, local=lane%32;
        for(int col=warp; col<14; col+=8) {
            if(col>=k) {
                double dot=0.0;
                for(int i=local; i<48; i+=32) dot+=v[i]*a[i*14+col];
                for(int offset=16; offset>0; offset/=2)
                    dot+=__shfl_down_sync(0xffffffffu,dot,offset);
                dot=__shfl_sync(0xffffffffu,dot,0);
                for(int i=local; i<48; i+=32) a[i*14+col]-=beta*v[i]*dot;
            }
        }
        for(int row=warp; row<48; row+=8) {
            double dot=0.0;
            for(int i=local; i<48; i+=32) dot+=q[row*49+i]*v[i];
            for(int offset=16; offset>0; offset/=2)
                dot+=__shfl_down_sync(0xffffffffu,dot,offset);
            dot=__shfl_sync(0xffffffffu,dot,0);
            for(int i=local; i<48; i+=32) q[row*49+i]-=beta*dot*v[i];
        }
        __syncthreads();
    }
    for(int i=lane; i<48*14; i+=256) rout.data[i]=a[i];
    for(int i=lane; i<48*48; i+=256) qout.data[i]=q[(i/48)*49+i%48];
#endif
""")
def small_qr(input: wp.array(dtype=wp.float64), qout: wp.array(dtype=wp.float64), rout: wp.array(dtype=wp.float64)): ...


@wp.kernel
def factor(input: wp.array(dtype=wp.float64), qout: wp.array(dtype=wp.float64), rout: wp.array(dtype=wp.float64)):
    small_qr(input, qout, rout)


def main():
    """Validate the actual matrix and measure captured repeat factorization."""
    import torch

    wp.init()
    data = dict(np.load("/tmp/colibri_joint_tree_mobility.npz"))
    matrix = data["qr_input"]
    a = wp.array(matrix.ravel(), dtype=wp.float64, device="cuda:0")
    q = wp.empty(48 * 48, dtype=wp.float64, device="cuda:0")
    r = wp.empty(48 * 14, dtype=wp.float64, device="cuda:0")
    wp.launch(factor, dim=256, inputs=[a, q, r], block_dim=256, device="cuda:0")
    wp.synchronize()
    Q, R = q.numpy().reshape(48, 48), r.numpy().reshape(48, 14)
    reconstruction = np.linalg.norm(Q @ R - matrix) / np.linalg.norm(matrix)
    orthogonality = np.max(np.abs(Q.T @ Q - np.eye(48)))
    ref, _ = np.linalg.qr(matrix, mode="complete")
    projector = Q[:, 14:] @ Q[:, 14:].T
    ref_projector = ref[:, 14:] @ ref[:, 14:].T
    projector_error = np.linalg.norm(projector - ref_projector)
    assert reconstruction < 1e-12 and orthogonality < 1e-12
    assert projector_error < 1e-7
    with wp.ScopedCapture(device="cuda:0") as capture:
        for _ in range(100):
            wp.launch(factor, dim=256, inputs=[a, q, r], block_dim=256, device="cuda:0")
    samples = []
    stream = torch.cuda.ExternalStream(wp.get_stream("cuda:0").cuda_stream)
    for _ in range(7):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(stream)
        wp.capture_launch(capture.graph)
        end.record(stream)
        end.synchronize()
        samples.append(start.elapsed_time(end) / 100)
    assert np.array_equal(Q, q.numpy().reshape(48, 48)), "Repeated QR is not deterministic"
    pose_checks = []
    for suffix in ("", "_pose1", "_pose30", "_pose120"):
        path = f"/tmp/colibri_joint_tree{suffix}_mobility.npz"
        pose = dict(np.load(path))
        pose_input = wp.array(pose["qr_input"].ravel(), dtype=wp.float64, device="cuda:0")
        wp.launch(factor, dim=256, inputs=[pose_input, q, r], block_dim=256, device="cuda:0")
        pose_q = q.numpy().reshape(48, 48)
        complement = pose_q[:, 14:]
        physical = pose["Tbody"] @ complement[:46]
        mobility = physical @ physical.T
        reference = pose["G"] @ pose["G"].T
        relative = np.linalg.norm(mobility - reference) / np.linalg.norm(reference)
        all_rows = np.max(np.abs(pose["K_all"] @ complement))
        assert relative < 1e-7 and all_rows < 1e-8, (path, relative, all_rows)
        pose_checks.append(
            {
                "input": path,
                "relative_physical_mobility_error": float(relative),
                "all_remaining_joint_row_residual": float(all_rows),
            }
        )
    result = {
        "scope": "Frozen actual 48x14 QR only; no production changes",
        "pose_checks": pose_checks,
        "repeat_byte_identical": True,
        "samples_ms": samples,
        "median_ms": float(np.median(samples)),
        "reconstruction": float(reconstruction),
        "orthogonality": float(orthogonality),
        "complement_projector_error": float(projector_error),
    }
    Path("/tmp/colibri_small_tree_qr_gpu.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
