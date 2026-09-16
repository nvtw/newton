# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compatibility installer for studies created before the constructor option."""


def install(solver):
    world = solver.world
    if (
        world.step_layout != "single_world"
        or not world.mass_splitting_enabled
        or world._colored_contact_headers
        or world._colored_contact_rows
        or world._contact_patch_enabled
        or world.num_particles
        or world.num_cloth_triangles
        or world.num_cloth_bending
        or world.num_soft_tetrahedra
        or world.num_soft_hexahedra
    ):
        raise ValueError("Split parallel preparation requires uncompressed rigid single-world point contacts.")
    # New callers should pass parallel_contact_prepare=True to SolverPhoenX.
    # Existing studies install their joint/chunk route after constructing it.
    world.parallel_contact_prepare = True
