# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""KAMINO: Linear Algebra: Matrix factorization implementations (kernels and launchers)"""

from .llt_blocked import (
    llt_blocked_factorize,
    llt_blocked_solve,
    llt_blocked_solve_inplace,
    make_llt_blocked_factorize_kernel,
    make_llt_blocked_solve_inplace_kernel,
    make_llt_blocked_solve_kernel,
)
from .llt_blocked_nd import (
    llt_blocked_nd_build_inv_P,
    llt_blocked_nd_build_tile_pattern,
    llt_blocked_nd_factorize,
    llt_blocked_nd_permute_matrix,
    llt_blocked_nd_permute_vector,
    llt_blocked_nd_solve,
    llt_blocked_nd_solve_inplace,
    llt_blocked_nd_symbolic_fill_in,
    make_llt_blocked_nd_build_inv_P_kernel,
    make_llt_blocked_nd_build_tile_pattern_kernel,
    make_llt_blocked_nd_factorize_kernel,
    make_llt_blocked_nd_permute_matrix_kernel,
    make_llt_blocked_nd_permute_vector_kernel,
    make_llt_blocked_nd_solve_inplace_kernel,
    make_llt_blocked_nd_solve_kernel,
    make_llt_blocked_nd_symbolic_fill_in_kernel,
)
from .llt_sequential import (
    _llt_sequential_factorize,
    _llt_sequential_solve,
    _llt_sequential_solve_inplace,
    llt_sequential_factorize,
    llt_sequential_solve,
    llt_sequential_solve_inplace,
)

###
# Module API
###

__all__ = [
    "_llt_sequential_factorize",
    "_llt_sequential_solve",
    "_llt_sequential_solve_inplace",
    "llt_blocked_factorize",
    "llt_blocked_nd_build_inv_P",
    "llt_blocked_nd_build_tile_pattern",
    "llt_blocked_nd_factorize",
    "llt_blocked_nd_permute_matrix",
    "llt_blocked_nd_permute_vector",
    "llt_blocked_nd_solve",
    "llt_blocked_nd_solve_inplace",
    "llt_blocked_nd_symbolic_fill_in",
    "llt_blocked_solve",
    "llt_blocked_solve_inplace",
    "llt_sequential_factorize",
    "llt_sequential_solve",
    "llt_sequential_solve_inplace",
    "make_llt_blocked_factorize_kernel",
    "make_llt_blocked_nd_build_inv_P_kernel",
    "make_llt_blocked_nd_build_tile_pattern_kernel",
    "make_llt_blocked_nd_factorize_kernel",
    "make_llt_blocked_nd_permute_matrix_kernel",
    "make_llt_blocked_nd_permute_vector_kernel",
    "make_llt_blocked_nd_solve_inplace_kernel",
    "make_llt_blocked_nd_solve_kernel",
    "make_llt_blocked_nd_symbolic_fill_in_kernel",
    "make_llt_blocked_solve_inplace_kernel",
    "make_llt_blocked_solve_kernel",
]
