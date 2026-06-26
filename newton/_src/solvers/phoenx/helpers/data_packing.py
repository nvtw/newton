# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""C-style ``offsetof`` / ``sizeof`` for Warp structs.

Warp ``@wp.struct`` types are backed by a ``ctypes.Structure`` whose layout
matches the device-side C struct, so host ctypes offsets are device offsets.
Used for the column-major dword layout of constraint containers.
"""

from __future__ import annotations

import ctypes

import warp as wp
from warp._src.codegen import Struct as _WarpStruct

__all__ = [
    "dword_offset_of",
    "field_offsets",
    "num_dwords",
    "offset_of",
    "reinterpret_float_as_int",
    "reinterpret_int_as_float",
    "size_of",
]


def _as_warp_struct(struct_type: object) -> _WarpStruct:
    """Accept either the user-facing ``@wp.struct`` class or its ``Struct``."""
    if isinstance(struct_type, _WarpStruct):
        return struct_type
    inner = getattr(struct_type, "_wp_struct_", None) or getattr(struct_type, "cls", None)
    if isinstance(inner, _WarpStruct):
        return inner
    raise TypeError(f"{struct_type!r} is not a @wp.struct type")


def offset_of(struct_type: object, field: str) -> int:
    """Byte offset of ``field`` inside a ``@wp.struct`` type."""
    s = _as_warp_struct(struct_type)
    if field not in s.vars:
        raise AttributeError(f"{s.key!r} has no field {field!r}; available: {list(s.vars)}")
    return getattr(s.ctype, field).offset


def size_of(struct_type: object) -> int:
    """Size in bytes of a ``@wp.struct`` type (C/C++ ``sizeof``)."""
    return ctypes.sizeof(_as_warp_struct(struct_type).ctype)


def field_offsets(struct_type: object) -> dict[str, int]:
    """Return ``{field_name: byte_offset}`` for every field of ``struct_type``."""
    s = _as_warp_struct(struct_type)
    return {name: getattr(s.ctype, name).offset for name in s.vars}


def dword_offset_of(struct_type: object, field: str) -> int:
    """Field offset in 4-byte dwords. Asserts dword alignment (the column-major
    layout indexes a flat ``wp.array2d[wp.float32]`` and can't address sub-dword
    fields)."""
    byte_off = offset_of(struct_type, field)
    if byte_off % 4 != 0:
        raise ValueError(
            f"{_as_warp_struct(struct_type).key}.{field} byte offset {byte_off} is not "
            "dword-aligned; the column-major dword layout cannot address it."
        )
    return byte_off // 4


def num_dwords(struct_type: object) -> int:
    """Total dword count of a ``@wp.struct`` (``size_of(...) // 4``).
    Asserts dword alignment; mixed widths may need explicit padding."""
    sz = size_of(struct_type)
    if sz % 4 != 0:
        raise ValueError(
            f"{_as_warp_struct(struct_type).key} size {sz} is not dword-aligned; "
            "rearrange fields to 4-byte alignment for the column-major dword layout."
        )
    return sz // 4


# Reinterpret-cast helpers. Non-differentiable; do not feed autodiff tape.


@wp.func_native("""
return reinterpret_cast<int32_t&>(value);
""")
def reinterpret_float_as_int(value: wp.float32) -> wp.int32:
    """Bit-cast a ``float32`` to ``int32`` (IEEE-754 bit pattern)."""
    ...


@wp.func_native("""
return reinterpret_cast<float&>(value);
""")
def reinterpret_int_as_float(value: wp.int32) -> wp.float32:
    """Bit-cast an ``int32`` back to ``float32``."""
    ...
