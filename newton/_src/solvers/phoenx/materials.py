# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Per-shape material system for :class:`PhoenXWorld`.

PhysX-style: each shape carries a material index. Contact pairs combine the two
materials with the stricter combine mode (max(mode_a, mode_b) wins;
AVERAGE < MIN < MULTIPLY < MAX). Material 0 is the default fallback.
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
    """PhysX ``PxCombineMode``. Numeric order = "stricter wins" via ``max``."""

    AVERAGE = 0
    MIN = 1
    MULTIPLY = 2
    MAX = 3


COMBINE_AVERAGE = int(CombineMode.AVERAGE)
COMBINE_MIN = int(CombineMode.MIN)
COMBINE_MULTIPLY = int(CombineMode.MULTIPLY)
COMBINE_MAX = int(CombineMode.MAX)

DEFAULT_MATERIAL_INDEX: int = 0


@dataclass
class Material:
    """Host-side material handle. Frictions are dimensionless Coulomb coefficients;
    restitution is the normal bounce coefficient in [0, 1]."""

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
    """Kernel-visible packed material record. Mirrors :class:`Material`."""

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
    """Build a ``wp.array[MaterialData]`` from a Python list. Element 0 is always
    populated (default :class:`Material` if list is empty)."""
    if not materials:
        materials = [Material()]
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


# Kernel-side combine helpers. Per PhysX, the effective mode is max(mode_a, mode_b).
# AVERAGE is the fallthrough default in _combine_values, so it needs no constant.

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
    return (a + b) * wp.float32(0.5)


@wp.func
def resolve_friction_in_kernel(
    materials: wp.array[MaterialData],
    mat_a: wp.int32,
    mat_b: wp.int32,
    default_friction: wp.float32,
) -> wp.float32:
    """Effective per-pair dynamic friction. Falls back to ``default_friction`` if
    materials table is empty or indices are out of range."""
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
    """Effective per-pair static friction (stick threshold). Same combine logic as
    :func:`resolve_friction_in_kernel` but reads ``static_friction``."""
    if materials.shape[0] == 0:
        return default_friction
    mat_count = materials.shape[0]
    if mat_a < 0 or mat_a >= mat_count or mat_b < 0 or mat_b >= mat_count:
        return default_friction
    ma = materials[mat_a]
    mb = materials[mat_b]
    mode = wp.max(ma.friction_combine_mode, mb.friction_combine_mode)
    return _combine_values(ma.static_friction, mb.static_friction, mode)
