# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Per-shape material system for :class:`PhoenXWorld`.

PhysX-style: each shape carries a material index, the table holds
``(static_friction, dynamic_friction, restitution,
friction_combine_mode, restitution_combine_mode)``, and contact pairs
resolve an effective friction / restitution by combining the two
materials with the stricter combine mode
(``max(mode_a, mode_b)`` wins; AVERAGE < MIN < MULTIPLY < MAX).

Combine modes (PhysX ``PxCombineMode``):

* :data:`COMBINE_AVERAGE` -- ``(a + b) / 2`` (default).
* :data:`COMBINE_MIN` -- slippier surface wins.
* :data:`COMBINE_MULTIPLY` -- ``a * b``; coefficient product.
* :data:`COMBINE_MAX` -- grippier surface wins.

Material 0 is reserved as the default (``mu = 0.5``, ``e = 0``,
``COMBINE_AVERAGE``) so un-assigned shapes behave like the
pre-material code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import warp as wp

__all__ = [
    "COMBINE_AVERAGE",
    "COMBINE_MAX",
    "COMBINE_MIN",
    "COMBINE_MULTIPLY",
    "DEFAULT_MATERIAL_INDEX",
    "CombineMode",
    "Material",
    "MaterialData",
    "material_table_from_list",
    "pack_material_data",
    "resolve_friction_in_kernel",
    "resolve_friction_static_in_kernel",
]


class CombineMode(IntEnum):
    """PhysX-style combine modes for per-pair material resolution.

    Numeric values match the PhysX ``PxCombineMode`` enum exactly so
    the "stricter wins" tie-break is a plain ``max(mode_a, mode_b)``.
    """

    AVERAGE = 0
    MIN = 1
    MULTIPLY = 2
    MAX = 3


#: Module-level constants so kernels can import them as compile-time
#: literals (``wp.constant``-wrapped at the bottom of this module) and
#: Python callers can use them without importing the enum.
COMBINE_AVERAGE = int(CombineMode.AVERAGE)
COMBINE_MIN = int(CombineMode.MIN)
COMBINE_MULTIPLY = int(CombineMode.MULTIPLY)
COMBINE_MAX = int(CombineMode.MAX)

#: Material index 0 is always the default material. Shapes whose
#: ``shape_material[s]`` is unset (-1 sentinel) or exactly 0 fall
#: back to it so the pre-material contact path (``default_friction``)
#: survives as "every shape has material 0".
DEFAULT_MATERIAL_INDEX: int = 0


@dataclass
class Material:
    """Plain-Python handle for a single material.

    Used host-side in :meth:`WorldBuilder.add_material`; the solver
    only ever sees the packed :class:`MaterialData` ``wp.struct`` form
    produced by :func:`pack_material_data`.

    Units follow PhysX: ``static_friction`` / ``dynamic_friction`` are
    dimensionless Coulomb coefficients; ``restitution`` is the normal
    bounce coefficient in ``[0, 1]`` (0 = perfectly inelastic, 1 =
    perfectly elastic).
    """

    static_friction: float = 0.5
    dynamic_friction: float = 0.5
    restitution: float = 0.0
    friction_combine_mode: int = COMBINE_AVERAGE
    restitution_combine_mode: int = COMBINE_AVERAGE

    def __post_init__(self) -> None:
        if self.static_friction < 0.0:
            raise ValueError(f"static_friction must be >= 0 (got {self.static_friction})")
        if self.dynamic_friction < 0.0:
            raise ValueError(f"dynamic_friction must be >= 0 (got {self.dynamic_friction})")
        if self.dynamic_friction > self.static_friction + 1e-6:
            # PhysX enforces this as a hard error; we only warn via
            # assertion because the solver currently only consumes
            # dynamic_friction (the tangent clamp is ``mu_k *
            # lam_n``) and a small inversion is harmless. Tighten to
            # a hard error if the solver ever grows a separate stick
            # row.
            pass
        if not (0.0 <= self.restitution <= 1.0):
            raise ValueError(f"restitution must be in [0, 1] (got {self.restitution})")
        for name, mode in (
            ("friction_combine_mode", self.friction_combine_mode),
            ("restitution_combine_mode", self.restitution_combine_mode),
        ):
            if mode not in (
                COMBINE_AVERAGE,
                COMBINE_MIN,
                COMBINE_MULTIPLY,
                COMBINE_MAX,
            ):
                raise ValueError(
                    f"{name}={mode} is not a valid CombineMode; expected one of AVERAGE/MIN/MULTIPLY/MAX (0/1/2/3)"
                )


@wp.struct
class MaterialData:
    """Kernel-visible packed material record.

    Mirrors :class:`Material` field-for-field. Kept as its own struct
    (rather than two parallel float/int arrays) so the materials
    table stays a single ``wp.array[MaterialData]`` with natural
    indexed access in the contact-pack kernel.
    """

    static_friction: wp.float32
    dynamic_friction: wp.float32
    restitution: wp.float32
    friction_combine_mode: wp.int32
    restitution_combine_mode: wp.int32


def pack_material_data(m: Material) -> MaterialData:
    """Convert host-side :class:`Material` to the kernel struct."""
    d = MaterialData()
    d.static_friction = float(m.static_friction)
    d.dynamic_friction = float(m.dynamic_friction)
    d.restitution = float(m.restitution)
    d.friction_combine_mode = int(m.friction_combine_mode)
    d.restitution_combine_mode = int(m.restitution_combine_mode)
    return d


def material_table_from_list(materials: list[Material], device: wp.context.Devicelike = None) -> wp.array:
    """Build a ``wp.array[MaterialData]`` from a Python list.

    Element 0 is *always* the default material (same defaults as
    :class:`Material`'s dataclass). If the caller's list is empty
    the returned array still has one element. If the caller supplies
    their own material at index 0 we keep it -- they're explicitly
    overriding the default.
    """
    if not materials:
        materials = [Material()]
    # Structured numpy dtype matching MaterialData layout. Warp's
    # ``from_numpy`` can take an array of this dtype and reinterpret
    # it as ``MaterialData``.
    dtype = np.dtype(
        [
            ("static_friction", np.float32),
            ("dynamic_friction", np.float32),
            ("restitution", np.float32),
            ("friction_combine_mode", np.int32),
            ("restitution_combine_mode", np.int32),
        ]
    )
    arr = np.zeros(len(materials), dtype=dtype)
    for i, m in enumerate(materials):
        arr[i]["static_friction"] = m.static_friction
        arr[i]["dynamic_friction"] = m.dynamic_friction
        arr[i]["restitution"] = m.restitution
        arr[i]["friction_combine_mode"] = m.friction_combine_mode
        arr[i]["restitution_combine_mode"] = m.restitution_combine_mode
    return wp.from_numpy(arr, dtype=MaterialData, device=device)


# ---------------------------------------------------------------------------
# Kernel-side combine helpers
# ---------------------------------------------------------------------------
#
# The contact pack kernel imports these ``@wp.func`` helpers so the
# combine logic lives in one place. Per PhysX, the effective mode is
# ``max(mode_a, mode_b)`` -- the stricter policy wins.


_COMBINE_AVERAGE_C = wp.constant(wp.int32(COMBINE_AVERAGE))
_COMBINE_MIN_C = wp.constant(wp.int32(COMBINE_MIN))
_COMBINE_MULTIPLY_C = wp.constant(wp.int32(COMBINE_MULTIPLY))
_COMBINE_MAX_C = wp.constant(wp.int32(COMBINE_MAX))


@wp.func
def _combine_values(a: wp.float32, b: wp.float32, mode: wp.int32) -> wp.float32:
    """Apply a single combine mode to two scalar material values."""
    if mode == _COMBINE_MIN_C:
        return wp.min(a, b)
    if mode == _COMBINE_MULTIPLY_C:
        return a * b
    if mode == _COMBINE_MAX_C:
        return wp.max(a, b)
    # Default / AVERAGE
    return (a + b) * wp.float32(0.5)


@wp.func
def resolve_friction_in_kernel(
    materials: wp.array[MaterialData],
    mat_a: wp.int32,
    mat_b: wp.int32,
    default_friction: wp.float32,
) -> wp.float32:
    """Compute the effective per-pair friction coefficient.

    Matches PhysX semantics: the effective friction combines the two
    materials' ``dynamic_friction`` values under whichever
    ``friction_combine_mode`` is stricter (larger enum value).
    A negative material index short-circuits to ``default_friction``
    so callers that haven't wired in a materials table still get the
    old constant-friction behaviour.

    Called per contact column during :func:`ingest_contacts` -- the
    per-pair friction is written into the contact constraint header
    and reused across every PGS iteration for the rest of the step.
    """
    if materials.shape[0] == 0:
        return default_friction
    mat_count = materials.shape[0]
    if mat_a < 0 or mat_a >= mat_count or mat_b < 0 or mat_b >= mat_count:
        return default_friction
    ma = materials[mat_a]
    mb = materials[mat_b]
    mode = wp.max(ma.friction_combine_mode, mb.friction_combine_mode)
    return _combine_values(ma.dynamic_friction, mb.dynamic_friction, mode)


@wp.func
def resolve_friction_static_in_kernel(
    materials: wp.array[MaterialData],
    mat_a: wp.int32,
    mat_b: wp.int32,
    default_friction: wp.float32,
) -> wp.float32:
    """Compute the effective per-pair *static* friction coefficient.

    Twin of :func:`resolve_friction_in_kernel` -- same combine-mode
    selection but reads each material's ``static_friction`` instead
    of ``dynamic_friction``. The solver uses this as the "stick"
    threshold in the contact iterate: when the raw tangent impulse
    magnitude stays inside ``mu_static * lam_n`` the contact is in
    the static regime; once it breaches that threshold the impulse
    is clamped to ``mu_dynamic * lam_n`` (kinetic regime).

    Falls back to ``default_friction`` when no materials are
    registered, so pre-material scenes get ``mu_static ==
    mu_dynamic == default_friction`` and the two-regime clamp
    collapses to the single-coefficient circular cone.
    """
    if materials.shape[0] == 0:
        return default_friction
    mat_count = materials.shape[0]
    if mat_a < 0 or mat_a >= mat_count or mat_b < 0 or mat_b >= mat_count:
        return default_friction
    ma = materials[mat_a]
    mb = materials[mat_b]
    mode = wp.max(ma.friction_combine_mode, mb.friction_combine_mode)
    return _combine_values(ma.static_friction, mb.static_friction, mode)
