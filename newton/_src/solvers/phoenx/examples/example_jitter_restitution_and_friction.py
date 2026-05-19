# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Friction sweep: ten boxes sliding on a frictionless plane.
#
# Each box gets a different friction coefficient (linear ramp from
# 0.0 to 0.9). All ten start at the same height with the same initial
# +X linear velocity and decelerate at exactly ``mu * g`` -- the box
# with ``mu = 0.0`` glides forever, the ``mu = 0.9`` box stops first.
#
# The ground material has ``mu = 0`` and *every* material uses the
# ``MAX`` friction combine mode. PhysX-style combine resolution then
# gives ``effective_mu = max(mu_box, mu_ground) = mu_box`` per pair,
# so each cube's deceleration is determined purely by its own
# coefficient. (See the docstring on :class:`Example` for a full
# walk-through of how friction is combined.)
#
# Run: python -m newton._src.solvers.phoenx.examples.example_jitter_restitution_and_friction
###########################################################################

from __future__ import annotations

import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    run_ported_example,
)
from newton._src.solvers.phoenx.materials import (
    CombineMode,
    Material,
    material_table_from_list,
)

#: Half-extents of every cube (m).
HE = 0.4
#: Number of cubes in the sweep.
N_CUBES = 10
#: Cube center-to-center spacing along Y (m). Must comfortably exceed
#: ``2 * HE`` so the cubes never collide with each other.
SPACING = 2.0
#: Spawn height above the plane (m). Small clearance lets the
#: speculative-contact path grab the floor on the first step without
#: any initial overlap.
SPAWN_HEIGHT = HE + 0.01
#: Initial horizontal velocity along +X (m/s). Picked so the slowest
#: deceleration (``mu = 0.1`` -> ``a = 0.981 m/s^2``) still stops
#: within a handful of seconds.
INITIAL_VX = 8.0
#: Friction value for the ground material. With ``MAX`` combine mode
#: the effective per-pair friction collapses to the cube's own
#: coefficient; the ground value is irrelevant as long as it sits
#: below the cubes' values, so ``0.0`` is the cleanest choice.
GROUND_MU = 0.0


class Example(PortedExample):
    """Ten boxes sliding on a (combine-mode-`MAX`) frictionless plane.

    Friction combine in PhoenX (PhysX-style)
    ----------------------------------------
    Every shape carries a *material*: a record of
    ``(static_friction, dynamic_friction, restitution,
    friction_combine_mode, restitution_combine_mode)`` (see
    :class:`newton._src.solvers.phoenx.materials.Material`).

    For every contact pair ``(shape_a, shape_b)`` PhoenX resolves an
    effective friction in two steps -- *exactly* PhysX's rule:

    1. **Pick the stricter combine mode.** The combine modes are
       integer-valued (``AVERAGE = 0``, ``MIN = 1``, ``MULTIPLY = 2``,
       ``MAX = 3``). The mode used for the pair is
       ``max(mode_a, mode_b)``: the surface that asked for the
       stricter / more specific rule wins.
    2. **Apply that mode to the two ``dynamic_friction`` values.**

       * ``AVERAGE``    -> ``(mu_a + mu_b) / 2``
       * ``MIN``        -> ``min(mu_a, mu_b)`` (slippiest wins)
       * ``MULTIPLY``   -> ``mu_a * mu_b``
       * ``MAX``        -> ``max(mu_a, mu_b)`` (grippiest wins)

    A symmetric resolution runs over ``static_friction`` and is used
    by the solver's stick / slip transition.

    What this demo uses
    -------------------
    All ten cubes and the ground use ``friction_combine_mode = MAX``,
    so the effective per-pair friction is
    ``max(mu_cube, mu_ground) = max(mu_cube, 0) = mu_cube``. With the
    ground perfectly frictionless on its own, each cube's
    deceleration ``a = mu_cube * g`` is determined purely by its own
    material. Walking down the row from low to high ``mu``, the cubes
    visibly fan out -- the ``mu = 0.0`` cube glides forever, the
    ``mu = 0.9`` cube stops almost immediately.

    To verify the rule, switch the combine mode to ``AVERAGE``: the
    effective friction becomes ``mu_cube / 2`` (since the ground is
    0) and every cube travels twice as far. ``MULTIPLY`` would zero
    out the friction for every pair (any factor times 0 is 0), so
    none of the cubes would stop -- another useful sanity check.
    """

    sim_substeps = 12
    solver_iterations = 16

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()
        extents: list = []
        # Cubes line up along Y so they never collide with each other;
        # they slide along +X without interaction.
        y0 = -((N_CUBES - 1) * SPACING) * 0.5
        for i in range(N_CUBES):
            mu = i / (N_CUBES - 1) * 0.9  # 0.0, 0.1, ..., 0.9
            cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=float(mu))
            body = builder.add_body(
                xform=wp.transform(
                    p=wp.vec3(0.0, y0 + i * SPACING, SPAWN_HEIGHT),
                    q=wp.quat_identity(),
                ),
                linear_velocity=(INITIAL_VX, 0.0, 0.0),
            )
            builder.add_shape_box(body, hx=HE, hy=HE, hz=HE, cfg=cfg)
            extents.append(default_box_half_extents(HE, HE, HE))
        return extents

    def post_build(self) -> None:
        """Install per-shape friction materials using ``MAX`` combine.

        :class:`PortedExample` doesn't wire ``model.shape_material_mu``
        through to PhoenX (it skips the ``solver.py`` adapter and
        constructs :class:`PhoenXWorld` directly), so the per-shape
        ``cfg.mu`` set on the cubes above would otherwise be ignored
        and every contact would use ``world.default_friction``. We
        build a small material table mirroring the same ``mu`` values
        and bind it to the world before the simulate graph is
        captured.
        """
        # Newton ordering: each ``add_shape_box`` produced one shape;
        # the ground plane added a single shape first. So shape index
        # 0 is the ground, shapes 1..N are the cubes.
        materials: list[Material] = []
        # Index 0: ground. ``MAX`` combine + mu = 0 means every
        # cube-ground pair resolves to mu = max(mu_cube, 0) = mu_cube.
        materials.append(
            Material(
                static_friction=GROUND_MU,
                dynamic_friction=GROUND_MU,
                restitution=0.0,
                friction_combine_mode=CombineMode.MAX,
                restitution_combine_mode=CombineMode.MAX,
            )
        )
        # Indices 1..N: the cubes, mu = 0.0 .. 0.9.
        for i in range(N_CUBES):
            mu = float(i / (N_CUBES - 1) * 0.9)
            materials.append(
                Material(
                    static_friction=mu,
                    dynamic_friction=mu,
                    restitution=0.0,
                    friction_combine_mode=CombineMode.MAX,
                    restitution_combine_mode=CombineMode.MAX,
                )
            )

        material_table = material_table_from_list(materials, device=self.device)
        # Per-shape -> material index map. Identity here: shape 0 ->
        # material 0 (ground), shape i+1 -> material i+1 (cube i).
        shape_count = int(self.model.shape_count)
        shape_material = wp.array(
            list(range(shape_count)),
            dtype=wp.int32,
            device=self.device,
        )
        self.world.set_materials(material_table, shape_material)

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(8.0, -16.0, 5.0), pitch=-20.0, yaw=120.0)


if __name__ == "__main__":
    run_ported_example(Example)
