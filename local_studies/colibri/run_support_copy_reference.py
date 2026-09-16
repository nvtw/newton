"""Reproduce the copy-scaled frozen solve with numerical residual scaling."""

import numpy as np

from local_studies.colibri import coupled_support_reference as reference


def main():
    """Scale optimization residual units only; keep physical acceptance unchanged."""
    import scipy.optimize as opt

    original = opt.least_squares
    original_solve = reference.solve_coulomb
    seed = np.load("/tmp/colibri_support_copy_reference.npz")["solution"]

    def scaled(fun, x, jac, **kwargs):
        return original(lambda value: 1e4 * fun(value), x, jac=lambda value: 1e4 * jac(value), **kwargs)

    opt.least_squares = scaled
    reference.solve_coulomb = lambda A, rhs, initial, gamma, friction: original_solve(A, rhs, seed, gamma, friction)
    try:
        reference.main(
            copy_counts=np.load("/tmp/colibri_support_bridge_counts.npz")["star"],
            output="/tmp/colibri_support_copy_reference_scaled_residual",
        )
    finally:
        opt.least_squares = original
        reference.solve_coulomb = original_solve


if __name__ == "__main__":
    main()
