"""Compatibility installer for canonical inactive axial-row preparation."""

from newton._src.solvers.phoenx.constraints.constraint_joint import joint_constraint_prepare_inequality as prepare


def install():
    """Bind the canonical helper for existing local comparison scripts."""
    from newton._src.solvers.phoenx import solver_phoenx_kernels

    solver_phoenx_kernels.joint_constraint_prepare_inequality = prepare
