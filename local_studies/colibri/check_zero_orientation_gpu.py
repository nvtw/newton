"""GPU normalization stationarity and local exact-zero native integration control."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp
from newton._src.solvers.phoenx.solver_phoenx_kernels import (
    MOTION_DYNAMIC,
    _integrate_velocities_kernel,
    rotate_inertia,
    sym6_from_mat33,
)

from local_studies.colibri.check_quaternion_stationarity import repeat
from local_studies.colibri.exact_zero_orientation import exact_zero_integrate
from newton._src.solvers.phoenx.body import BodyContainer, body_container_zeros


@wp.kernel(enable_backward=False)
def initialize(b: BodyContainer, q: wp.array[wp.quatf], omega: wp.float32):
    """Use finite anisotropic inertia and off-origin resting dynamic bodies."""
    i = wp.tid()
    b.orientation[i] = q[i]
    b.position[i] = wp.vec3f(0.1, 0.2, 0.3)
    b.motion_type[i] = MOTION_DYNAMIC
    b.inverse_mass[i] = wp.float32(1.0)
    local = wp.mat33f(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0)
    b.inverse_inertia[i] = local
    b.inverse_inertia_world[i] = sym6_from_mat33(rotate_inertia(wp.quat_to_matrix(q[i]), local))
    b.angular_velocity[i] = wp.vec3f(0.0, 0.0, omega)


def main():
    """Require zero-motion preservation and unchanged nonzero-omega behavior."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    old = np.load("/tmp/colibri_quaternion_stationarity_0.0.npz")["q"]
    ids = np.flatnonzero(
        np.all(old[:, 1].view(np.uint32) == old[:, 3].view(np.uint32), axis=1)
        & np.any(old[:, 1].view(np.uint32) != old[:, 2].view(np.uint32), axis=1)
    )
    q = old[ids, 0]
    qa = wp.array(q, dtype=wp.quatf, device=args.device)
    out = wp.zeros((len(q), 6), dtype=wp.quatf, device=args.device)
    wp.launch(repeat, dim=len(q), inputs=[qa, out, wp.float32(0.0)], device=args.device)
    values = out.numpy()
    cycles = np.all(values[:, 1].view(np.uint32) == values[:, 3].view(np.uint32), axis=1) & np.any(
        values[:, 1].view(np.uint32) != values[:, 2].view(np.uint32), axis=1
    )
    records = []
    for omega in (0.0, 1e-8, 1e-6, 1e-3):
        outputs = []
        for kernel in (_integrate_velocities_kernel, exact_zero_integrate):
            b = body_container_zeros(len(q), device=args.device)
            wp.launch(initialize, dim=len(q), inputs=[b, qa, wp.float32(omega)], device=args.device)
            states = []
            for _step in range(4):
                wp.launch(kernel, dim=len(q), inputs=[b, wp.float32(1.0 / 3600.0)], device=args.device)
                states.append(b.orientation.numpy())
            outputs.append(
                {
                    "q": np.asarray(states),
                    "v": b.velocity.numpy(),
                    "w": b.angular_velocity.numpy(),
                    "x": b.position.numpy(),
                    "inertia": b.inverse_inertia_world.numpy(),
                }
            )
        baseline, candidate = outputs
        if omega == 0:
            assert candidate["q"].tobytes() == np.broadcast_to(q, (4, *q.shape)).copy().tobytes()
            assert np.count_nonzero(candidate["v"]) == 0 and np.count_nonzero(candidate["w"]) == 0
            assert candidate["x"].tobytes() == baseline["x"].tobytes()
        else:
            assert all(candidate[k].tobytes() == baseline[k].tobytes() for k in candidate)
        records.append(
            {
                "omega": omega,
                "baseline_changed": int(np.sum(np.any(baseline["q"][0] != q, axis=1))),
                "candidate_changed": int(np.sum(np.any(candidate["q"][0] != q, axis=1))),
                "nonzero_bytes_equal": bool(omega != 0),
            }
        )
    report = {
        "device": args.device,
        "input_cpu_cycles": len(q),
        "device_helper_cycles": int(cycles.sum()),
        "cases": records,
        "scope": "Exact-zero guard only; all nonzero angular velocities retain original native midpoint integration and its existing1e-9 angle deadzone. Resting mass/inertia/off-origin bodies have exactlyzero P,L,KE before/after. No production edits.",
    }
    Path("/tmp/colibri_zero_orientation_" + args.device.replace(":", "_") + ".json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
