"""C-style ``offsetof`` / ``sizeof`` for Warp structs.

Warp doesn't expose pointer arithmetic or an ``offsetof`` operator inside
``@wp.func`` / ``@wp.kernel`` — structs are values there, not addressable
memory. But every ``@wp.struct`` is backed on the host by a
``ctypes.Structure`` (``MyStruct.ctype``) whose layout is what the runtime
serializes to the device, and the device-side C struct that Warp's codegen
emits has its fields in the same declaration order. So host ``ctypes``
offsets are the device offsets.

This module exposes those offsets as plain Python ints so they can be
baked into ``wp.constant``s, used to interpret a ``wp.array(dtype=wp.uint8)``
view of a typed buffer, etc.

Example:

    >>> from newton._src.solvers.phoenx.constraints.constraint_ball_socket import BallSocketData
    >>> from newton._src.solvers.phoenx.helpers.data_packing import offset_of, dword_offset_of, num_dwords
    >>> offset_of(BallSocketData, "body1")
    0
    >>> dword_offset_of(BallSocketData, "local_anchor1")
    2
    >>> num_dwords(BallSocketData)  # 31 dwords for a stripped-down BallSocket
    31
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
    # ``@wp.struct`` rebinds the class name to a ``Struct`` instance, so
    # ``BallSocketData`` is already a ``Struct``. Accept both for safety.
    if isinstance(struct_type, _WarpStruct):
        return struct_type
    inner = getattr(struct_type, "_wp_struct_", None) or getattr(struct_type, "cls", None)
    if isinstance(inner, _WarpStruct):
        return inner
    raise TypeError(f"{struct_type!r} is not a @wp.struct type")


def offset_of(struct_type: object, field: str) -> int:
    """Byte offset of ``field`` inside a ``@wp.struct`` type.

    Mirrors C/C++ ``offsetof(struct_type, field)``.

    Args:
        struct_type: A type decorated with ``@wp.struct``.
        field: The field name as declared in the struct's annotations.

    Returns:
        The field's byte offset within the struct's host/device layout.

    Raises:
        TypeError: If ``struct_type`` is not a Warp struct.
        AttributeError: If ``field`` is not a member of the struct.
    """
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
    """Field offset measured in 4-byte dwords instead of bytes.

    Used by the column-major constraint container, which stores all
    constraint state as a flat ``wp.array2d[wp.float32]`` indexed by
    ``[dword_offset, constraint_index]``. Asserts the byte offset is
    dword-aligned -- if it isn't, the struct has a non-float-sized field
    that the column-major layout can't address (introduce a u32 padding
    field or split the struct).
    """
    byte_off = offset_of(struct_type, field)
    if byte_off % 4 != 0:
        raise ValueError(
            f"{_as_warp_struct(struct_type).key}.{field} byte offset {byte_off} is not "
            "dword-aligned; the column-major dword layout cannot address it."
        )
    return byte_off // 4


def num_dwords(struct_type: object) -> int:
    """Total dword count of a ``@wp.struct`` (``size_of(...) // 4``).

    Asserts the struct size is dword-aligned -- which it is for any
    struct made of ``int32`` / ``float32`` / ``vec3f`` / ``quatf`` /
    ``mat33f`` fields. Mixed widths (``int64`` next to ``int32`` etc.)
    can introduce padding that breaks the dword count; in that case
    rearrange / pad fields so the struct is purely 4-byte words.
    """
    sz = size_of(struct_type)
    if sz % 4 != 0:
        raise ValueError(
            f"{_as_warp_struct(struct_type).key} size {sz} is not dword-aligned; "
            "rearrange fields to 4-byte alignment for the column-major dword layout."
        )
    return sz // 4


# ---------------------------------------------------------------------------
# Reinterpret-cast helpers (C++ ``reinterpret_cast<U&>(t)``)
# ---------------------------------------------------------------------------
#
# Warp has no built-in bit-cast operator. ``wp.int32(value)`` performs a
# *value* conversion (truncates toward zero); to get the IEEE-754 bit
# pattern of a float as an int (or vice versa) we need a real
# ``reinterpret_cast``. ``@wp.func_native`` lets us drop straight into
# C++/CUDA: the body string is spliced into the generated kernel
# verbatim, so we get the exact codegen the compiler would produce for a
# C++ ``reinterpret_cast<U&>(t)`` -- no function-call overhead and no
# memory traffic.
#
# Pattern mirrors ``float_flip`` in
# :mod:`newton._src.geometry.contact_reduction` (Stereopsis radix sort).
# Differentiability: these are intentionally non-differentiable bit
# tricks; do not feed their outputs through the autodiff tape.


@wp.func_native("""
return reinterpret_cast<int32_t&>(value);
""")
def reinterpret_float_as_int(value: wp.float32) -> wp.int32:
    """Bit-cast a ``float32`` to ``int32`` (IEEE-754 bit pattern).

    Equivalent to C++ ``reinterpret_cast<int32_t&>(value)``. Total order
    on positive floats is preserved; for a sortable key across signs see
    ``float_flip`` in :mod:`newton._src.geometry.contact_reduction`.
    """
    ...


@wp.func_native("""
return reinterpret_cast<float&>(value);
""")
def reinterpret_int_as_float(value: wp.int32) -> wp.float32:
    """Bit-cast an ``int32`` back to ``float32`` (inverse of :func:`reinterpret_float_as_int`)."""
    ...


