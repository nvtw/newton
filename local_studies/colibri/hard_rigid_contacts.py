# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic hard normal rows; preserve geometry, bias and friction."""

import warp as wp

from newton._src.solvers.phoenx.constraints import constraint_contact_cloth


@wp.func
def _hard_coefficients(idt: wp.float32):
    return wp.float32(1.0), wp.float32(0.0)


def install():
    """Select before constructing or capturing the solver."""
    constraint_contact_cloth._default_contact_solver_coefficients = _hard_coefficients
