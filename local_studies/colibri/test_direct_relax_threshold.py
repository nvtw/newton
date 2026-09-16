"""CPU native-kernel regression for scaled direct relaxation skipping.

Default mode records the current skipped residual and the exactly solved
one-row control. --require-correction is a fail-before acceptance gate.
--force-correction exercises the same native RHS/scatter with its gate bypassed
only in this local fixture; it does not modify the production threshold.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.direct_equality import (
    _DIRECT_RELAX_RESIDUAL_TOLERANCE,
    _accumulate_direct_impulse_kernel,
    _apply_direct_equality_delta_kernel,
    _build_direct_equality_rhs_kernel,
    _prepare_direct_equality_diagonal_kernel,
)
from newton._src.solvers.phoenx.body import body_container_zeros


def run_case(mass_scale=1.0, force_correction=False, use_bias=False):
    """Run the actual diagonal/RHS/impulse kernels for one hard translational row."""
    device = "cpu"

    def ints(values):
        return wp.array(values, dtype=wp.int32, device=device)

    def floats(values):
        return wp.array(values, dtype=wp.float32, device=device)

    bodies = body_container_zeros(3, device=device)
    masses = np.array([1.0, 4.0]) * mass_scale
    bodies.inverse_mass.assign(np.r_[0.0, 1 / masses].astype(np.float32))
    before = np.zeros((3, 3), np.float32)
    before[2, 0] = np.float32(7.8e-5)
    bodies.velocity.assign(before)
    joint = ints([0])
    local = ints([0])
    structural = ints([0])
    parent = ints([0])
    child = ints([1])
    dynamic = wp.array([False], dtype=wp.bool, device=device)
    w0 = np.zeros((1, 6, 6), np.float32)
    w0[0, 0, 0] = -1
    w1 = -w0
    wrench0 = wp.array(w0, dtype=wp.spatial_vector, device=device)
    wrench1 = wp.array(w1, dtype=wp.spatial_vector, device=device)
    row_arrays = [wp.zeros((1, 6), dtype=wp.float32, device=device) for _ in range(4)]
    error, stiffness, damping, bias = row_arrays
    dynamic_mass = floats([1.0])
    matrix = floats([0.0])
    scale = floats([0.0])
    wp.launch(
        _prepare_direct_equality_diagonal_kernel,
        dim=1,
        inputs=[
            ints([0]),
            joint,
            local,
            dynamic,
            structural,
            parent,
            child,
            wrench0,
            wrench1,
            bodies,
            floats([0.0]),
            wp.float32(3600.0),
            error,
            stiffness,
            damping,
            dynamic_mass,
            bias,
            matrix,
            scale,
        ],
        device=device,
    )
    rhs = floats([0.0])
    active = ints([int(use_bias)])
    accumulated = floats([0.0])
    wp.launch(
        _build_direct_equality_rhs_kernel,
        dim=1,
        inputs=[
            joint,
            local,
            dynamic,
            structural,
            parent,
            child,
            wrench0,
            wrench1,
            bias,
            floats([0.0]),
            accumulated,
            dynamic_mass,
            bodies,
            wp.bool(use_bias),
            scale,
            rhs,
            active,
        ],
        device=device,
    )
    native_active = int(active.numpy()[0])
    # One row: the native equilibrated factor solve is exactly division by its
    # only diagonal. Use that analytical solution, then actual native scatter.
    if native_active or force_correction:
        delta = floats(rhs.numpy() / matrix.numpy())
        wp.launch(_accumulate_direct_impulse_kernel, dim=1, inputs=[delta, scale, accumulated], device=device)
        wp.launch(
            _apply_direct_equality_delta_kernel,
            dim=3,
            inputs=[
                ints([0, 0, 1, 2]),
                ints([0, 0]),
                joint,
                local,
                structural,
                parent,
                child,
                wrench0,
                wrench1,
                delta,
                scale,
                bodies,
            ],
            device=device,
        )
    after = bodies.velocity.numpy().astype(float)
    residual = float(after[2, 0] - after[1, 0])
    dp = np.sum(masses[:, None] * (after[1:] - before[1:]), axis=0)
    impulse = masses[:, None] * (after[1:] - before[1:])
    work = float(np.sum(0.5 * impulse * (before[1:] + after[1:])))
    energy = float(np.sum(0.5 * masses[:, None] * (after[1:] ** 2 - before[1:].astype(float) ** 2)))
    storage_bound = 4 * np.finfo(np.float32).eps * float(np.sum(masses[:, None] * (abs(before[1:]) + abs(after[1:]))))
    np.testing.assert_allclose(dp, 0, atol=storage_bound)
    np.testing.assert_allclose(work, energy, atol=1e-15)
    return {
        "mass_scale": mass_scale,
        "use_bias": use_bias,
        "forced_local_control": force_correction,
        "native_solve_active": native_active,
        "scaled_rhs": float(rhs.numpy()[0]),
        "row_scale": float(scale.numpy()[0]),
        "threshold": float(_DIRECT_RELAX_RESIDUAL_TOLERANCE),
        "initial_physical_residual_m_s": float(before[2, 0]),
        "final_physical_residual_m_s": residual,
        "physical_impulse_Ns": float(accumulated.numpy()[0]),
        "linear_momentum_error_Ns": dp.tolist(),
        "FP32_storage_momentum_bound_Ns": float(storage_bound),
        "work_J": work,
        "energy_change_J": energy,
    }


def main():
    """Expose native skip and compare the same row's physically paired correction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-correction", action="store_true")
    parser.add_argument("--force-correction", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/direct_relax_threshold.json"))
    args = parser.parse_args()
    actual = run_case(force_correction=args.force_correction)
    corrected = run_case(force_correction=True)
    primary = run_case(use_bias=True)
    scales = [run_case(mass_scale=value) for value in (1e-3, 1.0, 1e3)]
    report = {"actual": actual, "forced_control": corrected, "primary": primary, "mass_scaling": scales}
    args.output.write_text(json.dumps(report, indent=2))
    assert abs(corrected["final_physical_residual_m_s"]) < 1e-10
    assert abs(primary["final_physical_residual_m_s"]) < 1e-10
    assert corrected["work_J"] < 0
    if args.require_correction:
        assert abs(actual["final_physical_residual_m_s"]) < 1e-8, "Nonzero hard-row residual skipped"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
