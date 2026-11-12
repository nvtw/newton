# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tree-sparse dense mass factors for small reduced-coordinate trees.

The dense backend of :class:`ReducedPhoenXArticulation` replaces the
articulated-body (ABA) recurrences by an explicit factor of each
articulation's generalized mass. One warp per articulation assembles ``M``
about the articulation origin with the composite-rigid-body algorithm (CRBA),
adds the implicit joint diagonal (armature, ``dt kd``, ``dt^2 ke``), and
factors ``M = L^T L``. Eliminating leaves first, ``L`` inherits the ancestor
sparsity of ``M`` without fill-in (Featherstone, *Rigid Body Dynamics
Algorithms*, Sec. 6.5): row ``k`` is nonzero only at the ancestor DOFs of
``k`` and at ``k``. Rows are stored in path layout, their entries following
the DOF's root-to-self path, so the entry of ancestor ``j`` sits at the depth
of ``j``.

Row ``k`` of the factor is ``M_kj - sum_i L_ik L_ij`` over the descendants
``i`` of ``k``, written in pull form so DOFs of equal height are independent.
The shared-memory triangular solves hold the vector in lane registers and
eliminate one column per step: ``L^T z = b`` leaves first, each ``z_k``
updating its ancestors, and ``L x = z`` root first, each ``x_j`` updating its
descendants; the global :func:`dense_tree_solve` keeps the pull form by levels.

The velocity advance applies ``M^-1 = L^-1 L^-T`` on each articulation's lane
group, and the contact rows store ``J`` and ``W = M^-1 J^T`` for the projected
Gauss-Seidel contact solve, with ``J M^-1 J^T = |L^-T J^T|^2``.

Topology always comes from arrays: identical articulation topologies share one
template of path tables. Only the shared-memory capacities are compile-time
buckets.
"""

from __future__ import annotations

import dataclasses
import functools

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.reduced_contact_block import (
    _CACHED_PAGE_COUNT,
    _MAX_ROWS,
    _store_point_row_effective_mass,
    _sync_contact_group,
    _sync_contact_warp,
)
from newton._src.solvers.phoenx.body import BodyContainer, DenseTreeData
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer

DENSE_DOF_BUCKETS = (16, 24, 32, 40, 48, 56, 64)
"""Compile-time DOF capacities of the dense backend; larger trees keep the articulated-body path."""

_LANES = 32
# Words of the static shared memory one block may declare.
_SHARED_WORDS = 48 * 1024 // 4
_SYM66 = wp.types.vector(length=21, dtype=wp.float32)
# Rounds of 32 entries that one staging chunk keeps in flight.
_STAGE_ROUNDS = 8
_INERTIA_STRIDE = 24
_MOTION_STRIDE = 8
_vec24 = wp.types.vector(length=_INERTIA_STRIDE, dtype=wp.float32)
# Relative pivot floor of the float32 factorization: a few ulps of the assembled
# diagonal keep a round-off-cancelled pivot positive without changing
# well-conditioned trees.
_PIVOT_FLOOR = wp.constant(wp.float32(2.0**-20))


def dense_dof_bucket(dof_count: int) -> int:
    """Return the smallest DOF bucket that holds ``dof_count`` DOFs, or 0."""
    for bucket in DENSE_DOF_BUCKETS:
        if dof_count <= bucket:
            return bucket
    return 0


_shared_handle = wp.types.vector(length=2, dtype=wp.uint64)
"""A block's workspace: its 32-bit shared-window address and the global address of its factor tables."""


@functools.cache
def _make_shared_base(length: int, tables: int | None = None):
    """Return the handle of one block's static shared workspace of ``length`` words.

    With ``tables``, the handle also carries the address that maps the halfword
    offsets of a template's factor-only tables, which count from shared word 0
    with the template image at word ``tables``, into the image at
    ``image[source]`` in global memory; see :class:`DenseSharedLayout`.

    Warp tiles synchronize the block after every element write; the dense
    kernels order their phases with ``__syncwarp`` instead and address raw,
    16-byte aligned shared storage through 32-bit ``ld.shared``/``st.shared``
    addresses. Generic pointers into shared memory would make the compiler
    rebuild the block's shared window from ``SR_CgaCtaId`` at most accesses.
    """
    table_address = "0ull" if tables is None else f"(unsigned long long)(image.data + source) - {4 * int(tables)}ull"
    snippet = f"""
#if defined(__CUDA_ARCH__)
    __shared__ __align__(16) float dense_shared[{int(length)}];
    unsigned address = (unsigned)__cvta_generic_to_shared(dense_shared);
    // An opaque copy keeps the address in a register: rebuilt, it would read SR_CgaCtaId
    // on the dependency chain of every shared access.
    asm volatile("mov.u32 %0, %0;" : "+r"(address));
    return wp::vec_t<2, wp::uint64>(address, {table_address});
#else
    return wp::vec_t<2, wp::uint64>();
#endif
"""

    if tables is None:

        @wp.func_native(snippet)
        def shared_base() -> _shared_handle: ...

    else:

        @wp.func_native(snippet)
        def shared_base(image: wp.array[wp.uint32], source: wp.int32) -> _shared_handle: ...

    return shared_base


# Shared accesses are volatile so they keep their program order among each other and
# around ``__syncwarp``; the address is the 32-bit shared-window offset in ``base``.
_ADDRESS = "(unsigned)base[0] + {scale}u * (unsigned)index"


def _device_only(snippet: str, fallback: str = "") -> str:
    """Guard a shared-memory snippet; the dense kernels only run on CUDA devices."""
    return f"\n#if defined(__CUDA_ARCH__)\n{snippet}\n#else\n{fallback}\n#endif\n"


@wp.func_native(
    _device_only(
        f"""
    float value;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(value) : "r"({_ADDRESS.format(scale=4)}));
    return value;
""",
        "return {};",
    )
)
def _load(base: _shared_handle, index: wp.int32) -> wp.float32: ...


@wp.func_native(
    _device_only(
        f"""asm volatile("st.shared.f32 [%0], %1;" :: "r"({_ADDRESS.format(scale=4)}), "f"(value));""",
        "",
    )
)
def _store(base: _shared_handle, index: wp.int32, value: wp.float32): ...


@wp.func_native(
    _device_only(
        f"""
    int value;
    asm volatile("ld.shared.s32 %0, [%1];" : "=r"(value) : "r"({_ADDRESS.format(scale=4)}));
    return value;
""",
        "return {};",
    )
)
def _load_int(base: _shared_handle, index: wp.int32) -> wp.int32: ...


@wp.func_native(
    _device_only(
        f"""asm volatile("st.shared.s32 [%0], %1;" :: "r"({_ADDRESS.format(scale=4)}), "r"(value));""",
        "",
    )
)
def _store_int(base: _shared_handle, index: wp.int32, value: wp.int32): ...


@wp.func_native(
    _device_only(
        f"""
    const unsigned address = {_ADDRESS.format(scale=4)};
    wp::vec_t<24, float> result;
    for (int word = 0; word < 6; ++word) {{
        asm volatile("ld.shared.v4.f32 {{%0, %1, %2, %3}}, [%4];"
                     : "=f"(result[4 * word]), "=f"(result[4 * word + 1]), "=f"(result[4 * word + 2]),
                       "=f"(result[4 * word + 3])
                     : "r"(address + 16u * word));
    }}
    return result;
""",
        "return {};",
    )
)
def _load24(base: _shared_handle, index: wp.int32) -> _vec24:
    """Load a padded spatial inertia from 16-byte aligned shared storage."""
    ...


@wp.func_native(
    _device_only(
        f"""
    const unsigned address = {_ADDRESS.format(scale=4)};
    wp::spatial_vector result;
    float pad0;
    float pad1;
    asm volatile("ld.shared.v4.f32 {{%0, %1, %2, %3}}, [%4];"
                 : "=f"(result[0]), "=f"(result[1]), "=f"(result[2]), "=f"(result[3])
                 : "r"(address));
    asm volatile("ld.shared.v4.f32 {{%0, %1, %2, %3}}, [%4];"
                 : "=f"(result[4]), "=f"(result[5]), "=f"(pad0), "=f"(pad1)
                 : "r"(address + 16u));
    return result;
""",
        "return {};",
    )
)
def _load_motion(base: _shared_handle, index: wp.int32) -> wp.spatial_vector:
    """Load a padded motion subspace from 16-byte aligned shared storage."""
    ...


def _unrolled_module(name: str, max_unroll: int):
    """Return a module whose fixed-count loops up to ``max_unroll`` unroll into registers."""
    module = wp.get_module(name)
    module.options["max_unroll"] = max(int(module.options.get("max_unroll", 16)), int(max_unroll))
    return module


@wp.func
def _descendant_solve_sum(
    shared: _shared_handle,
    desc_row_base: wp.int32,
    desc_dof_base: wp.int32,
    begin: wp.int32,
    end: wp.int32,
    factor_offset: wp.int32,
    lane: wp.int32,
) -> wp.float32:
    """Return ``sum_i L_ik z_i`` over the descendants ``i`` of ``k`` for a lane's column ``z``."""
    sums = wp.vec4()
    index = begin
    while index + wp.int32(3) < end:
        sums += wp.vec4(
            _load(shared, factor_offset + _load_int(shared, desc_row_base + index))
            * _load(shared, wp.int32(_LANES) * _load_int(shared, desc_dof_base + index) + lane),
            _load(shared, factor_offset + _load_int(shared, desc_row_base + index + wp.int32(1)))
            * _load(shared, wp.int32(_LANES) * _load_int(shared, desc_dof_base + index + wp.int32(1)) + lane),
            _load(shared, factor_offset + _load_int(shared, desc_row_base + index + wp.int32(2)))
            * _load(shared, wp.int32(_LANES) * _load_int(shared, desc_dof_base + index + wp.int32(2)) + lane),
            _load(shared, factor_offset + _load_int(shared, desc_row_base + index + wp.int32(3)))
            * _load(shared, wp.int32(_LANES) * _load_int(shared, desc_dof_base + index + wp.int32(3)) + lane),
        )
        index += wp.int32(4)
    while index < end:
        sums[0] += _load(shared, factor_offset + _load_int(shared, desc_row_base + index)) * _load(
            shared, wp.int32(_LANES) * _load_int(shared, desc_dof_base + index) + lane
        )
        index += wp.int32(1)
    return (sums[0] + sums[1]) + (sums[2] + sums[3])


@wp.func
def _ancestor_solve_sum(
    shared: _shared_handle,
    path_base: wp.int32,
    factor_base: wp.int32,
    begin: wp.int32,
    end: wp.int32,
    lane: wp.int32,
) -> wp.float32:
    """Return ``sum_j L_ij x_j`` over the ancestor entries ``begin:end`` of row ``i`` for a lane's column."""
    sums = wp.vec4()
    entry = begin
    while entry + wp.int32(3) < end:
        sums += wp.vec4(
            _load(shared, factor_base + entry)
            * _load(shared, wp.int32(_LANES) * _load_int(shared, path_base + entry) + lane),
            _load(shared, factor_base + entry + wp.int32(1))
            * _load(shared, wp.int32(_LANES) * _load_int(shared, path_base + entry + wp.int32(1)) + lane),
            _load(shared, factor_base + entry + wp.int32(2))
            * _load(shared, wp.int32(_LANES) * _load_int(shared, path_base + entry + wp.int32(2)) + lane),
            _load(shared, factor_base + entry + wp.int32(3))
            * _load(shared, wp.int32(_LANES) * _load_int(shared, path_base + entry + wp.int32(3)) + lane),
        )
        entry += wp.int32(4)
    while entry < end:
        sums[0] += _load(shared, factor_base + entry) * _load(
            shared, wp.int32(_LANES) * _load_int(shared, path_base + entry) + lane
        )
        entry += wp.int32(1)
    return (sums[0] + sums[1]) + (sums[2] + sums[3])


@wp.func
def _sym66_times(packed: _vec24, vector: wp.spatial_vector) -> wp.spatial_vector:
    """Multiply a row-major upper-triangle packed symmetric 6x6 matrix by a vector."""
    result = wp.spatial_vector()
    index = wp.int32(0)
    for row in range(6):
        for column in range(row, 6):
            value = packed[index]
            result[row] += value * vector[column]
            if column != row:
                result[column] += value * vector[row]
            index += wp.int32(1)
    return result


@wp.func
def dense_tree_solve(
    dense: DenseTreeData,
    articulation: wp.int32,
    dof_start: wp.int32,
    lane: wp.int32,
    width: wp.int32,
    mask: wp.uint32,
    rhs: wp.array[wp.float32],
    solution: wp.array[wp.float32],
):
    """Write ``M^-1 rhs`` of one articulation into ``solution`` on its lane group.

    ``rhs`` must be complete for the group on entry; it receives ``L^-T rhs``.
    Lanes own the DOFs of one height level (``L^T z = rhs``) or depth level
    (``L x = z``) at a time.
    """
    base = dense.factor_start[articulation]
    template = dense.articulation_template[articulation]
    dof_base = dense.template_dof_start[template]
    for level in range(
        dense.template_height_start[template], dense.template_height_start[template + wp.int32(1)] - wp.int32(1)
    ):
        index = dense.height_start[level] + lane
        while index < dense.height_start[level + wp.int32(1)]:
            k = dense.height_dof[index]
            depth = dense.dof_entry[dof_base + k + wp.int32(1)] - wp.int32(1) - dense.dof_entry[dof_base + k]
            value = rhs[dof_start + k]
            for descendant in range(dense.desc_start[dof_base + k], dense.desc_start[dof_base + k + wp.int32(1)]):
                value -= (
                    dense.factor[base + dense.desc_row[descendant] + depth]
                    * rhs[dof_start + dense.desc_dof[descendant]]
                )
            rhs[dof_start + k] = value / dense.factor[base + dense.dof_entry[dof_base + k + wp.int32(1)] - wp.int32(1)]
            index += width
        _sync_contact_group(mask)
    path_base = dense.template_path_start[template]
    for level in range(
        dense.template_depth_start[template], dense.template_depth_start[template + wp.int32(1)] - wp.int32(1)
    ):
        index = dense.depth_start[level] + lane
        while index < dense.depth_start[level + wp.int32(1)]:
            i = dense.depth_dof[index]
            entry = dense.dof_entry[dof_base + i]
            diagonal_entry = dense.dof_entry[dof_base + i + wp.int32(1)] - wp.int32(1)
            value = rhs[dof_start + i]
            while entry < diagonal_entry:
                value -= dense.factor[base + entry] * solution[dof_start + dense.path[path_base + entry]]
                entry += wp.int32(1)
            solution[dof_start + i] = value / dense.factor[base + diagonal_entry]
            index += width
        _sync_contact_group(mask)


@wp.kernel(enable_backward=False)
def _dense_solve_kernel(
    articulation_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    dense: DenseTreeData,
    rhs: wp.array[wp.float32],
    solution: wp.array[wp.float32],
):
    thread = wp.tid()
    articulation = thread // wp.int32(_LANES)
    lane = thread - articulation * wp.int32(_LANES)
    dense_tree_solve(
        dense,
        articulation,
        joint_qd_start[articulation_start[articulation]],
        lane,
        wp.int32(_LANES),
        wp.uint32(0xFFFFFFFF),
        rhs,
        solution,
    )


@wp.kernel(enable_backward=False)
def _mask_kernel(values: wp.array[wp.int32], mask: wp.array[wp.int32], masked: wp.array[wp.int32]):
    index = wp.tid()
    masked[index] = values[index] * mask[index]


@wp.func_native(
    _device_only(
        f"""
    unsigned short value;
    asm volatile("ld.shared.u16 %0, [%1];" : "=h"(value) : "r"({_ADDRESS.format(scale=2)}));
    return value;
""",
        "return {};",
    )
)
def _load_u16(base: _shared_handle, index: wp.int32) -> wp.int32: ...


@wp.func_native(_device_only("    return ((const unsigned*)base[1])[index];", "return {};"))
def _image_word(base: _shared_handle, index: wp.int32) -> wp.uint32:
    """Load a word of the template image in global memory by its workspace word offset."""
    ...


@wp.func_native(
    _device_only(
        f"""asm volatile("st.shared.u16 [%0], %1;" :: "r"({_ADDRESS.format(scale=2)}), "h"((unsigned short)value));""",
        "",
    )
)
def _store_u16(base: _shared_handle, index: wp.int32, value: wp.int32): ...


@wp.func_native(
    _device_only(
        f"""
    unsigned value;
    asm volatile("ld.shared.u8 %0, [%1];" : "=r"(value) : "r"({_ADDRESS.format(scale=1)}));
    return value;
""",
        "return {};",
    )
)
def _load_u8(base: _shared_handle, index: wp.int32) -> wp.int32: ...


@wp.func_native(
    _device_only(
        f"""asm volatile("st.shared.u8 [%0], %1;" :: "r"({_ADDRESS.format(scale=1)}), "r"(value));""",
        "",
    )
)
def _store_u8(base: _shared_handle, index: wp.int32, value: wp.int32): ...


@wp.func_native(
    _device_only(
        f"""asm volatile("st.shared.u32 [%0], %1;" :: "r"({_ADDRESS.format(scale=4)}), "r"(value));""",
        "",
    )
)
def _store_word(base: _shared_handle, index: wp.int32, value: wp.uint32): ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    for (int offset = 16; offset > 0; offset >>= 1) {
        const int other = __shfl_xor_sync(0xffffffffu, value, offset);
        value = other > value ? other : value;
    }
#endif
    return value;
"""
)
def _warp_max_int(value: wp.int32) -> wp.int32:
    """Return the largest ``value`` among the lanes of the warp."""
    ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    return __shfl_sync(0xffffffffu, value, source);
#else
    return value;
#endif
"""
)
def _lane_float(value: wp.float32, source: wp.int32) -> wp.float32: ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    return __shfl_sync(0xffffffffu, value, source);
#else
    return value;
#endif
"""
)
def _lane_int(value: wp.int32, source: wp.int32) -> wp.int32: ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    return (int)__ballot_sync(0xffffffffu, value);
#else
    return value ? 1 : 0;
#endif
"""
)
def _warp_ballot(value: wp.bool) -> wp.int32:
    """Return the mask of the lanes whose ``value`` is true."""
    ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    return __ffs(mask) - 1;
#else
    for (int bit = 0; bit < 32; ++bit)
        if ((mask >> bit) & 1)
            return bit;
    return -1;
#endif
"""
)
def _lowest_lane(mask: wp.int32) -> wp.int32:
    """Return the lowest lane set in ``mask``, -1 for none."""
    ...


class DenseSharedLayout:
    """Offsets of one articulation's shared workspace for the dense factor.

    Float words hold the factor ``L``, the motion subspaces padded to float4
    words, and the assembled diagonal. The template image follows in words
    ``tables`` to ``tables_end`` as ancestor masks and 16-bit and 8-bit tables,
    addressed in halfwords and bytes; see :meth:`pack_template`. The workspace
    holds the image up to ``staged_end``: the masks and the path and subtree
    tables every pass reads. The factorization's own descendant and level
    tables stay in the template image in global memory, where the same
    offsets address them through the handle of :func:`_make_shared_base`;
    a block's shared memory bounds how many trees an SM runs at once. A
    scratch region holds the composite inertias while factoring and
    ``scratch_words`` words for the caller afterwards.
    """

    _HALFWORD_TABLES = ("desc_row", "level_pair", "dof_entry", "desc_start", "level_pair_start", "subtree_start")
    _BYTE_TABLES = ("path", "dof_joint", "subtree_joint", "joint_last_dof")

    def __init__(
        self,
        max_dofs: int,
        joint_capacity: int,
        entry_capacity: int,
        subtree_capacity: int,
        scratch_words: int = 0,
    ):
        self.factor = 0
        self.motion = entry_capacity
        self.diagonal = self.motion + _MOTION_STRIDE * max_dofs
        # Each DOF's ancestor DOFs as a 64-bit mask in two words, low word first.
        self.ancestors = self.diagonal + max_dofs
        half_word = self.ancestors + 2 * max_dofs
        self.tables = self.ancestors
        # Tables every pass reads follow the masks in the workspace.
        offset = 2 * half_word
        for name, count in (("dof_entry", max_dofs + 1), ("subtree_start", joint_capacity + 1)):
            setattr(self, name, offset)
            offset += count
        offset = 4 * ((offset + 1) // 2)
        for name, count in (
            ("path", entry_capacity),
            ("dof_joint", max_dofs),
            ("subtree_joint", subtree_capacity),
            ("joint_last_dof", joint_capacity),
        ):
            setattr(self, name, offset)
            offset += count
        self.staged_end = (offset + 3) // 4
        # The factorization's own tables stay in the template image.
        offset = 2 * self.staged_end
        for name, count in (
            ("desc_row", entry_capacity),
            ("level_pair", entry_capacity),
            ("desc_start", max_dofs + 1),
            ("level_pair_start", max_dofs + 2),
        ):
            setattr(self, name, offset)
            offset += count
        self.tables_end = (offset + 1) // 2
        # Spatial inertias are loaded as float4 words.
        self.scratch = 4 * ((self.staged_end + 3) // 4)
        # The scratch region also takes the factorization's tables once the inertias are spent.
        # One spare word takes the stores of lanes without a row.
        self.spare = self.scratch + max(
            _INERTIA_STRIDE * joint_capacity, self.tables_end - self.staged_end, scratch_words
        )
        self.size = self.spare + 1

    def pack_template(self, template: dict) -> np.ndarray:
        """Return the words ``tables`` to ``tables_end`` of a template's shared tables.

        Factor pairs ``k << 16 | t`` pack as ``k << 8 | t`` and joint path ends as
        ``joint_last_dof + 1``.
        """
        image = np.zeros(4 * (self.tables_end - self.tables), dtype=np.uint8)
        dof_count = len(template["dof_entry"]) - 1
        masks = np.asarray(template["ancestor_mask"], dtype=np.uint64)
        image[4 * (self.ancestors - self.tables) : 4 * (self.ancestors - self.tables) + masks.nbytes] = masks.view(
            np.uint8
        )
        values = {
            "desc_row": template["desc_row"],
            "level_pair": (template["level_pair"] >> 16) << 8 | (template["level_pair"] & 0xFF),
            "dof_entry": template["dof_entry"],
            "desc_start": template["desc_start"],
            "level_pair_start": template["level_pair_start"],
            "subtree_start": template["subtree_start"],
            "path": template["path"],
            "dof_joint": template["dof_joint"][:dof_count],
            "subtree_joint": template["subtree_joint"],
            "joint_last_dof": template["joint_last_dof"][:-1] + 1,
        }
        for names, dtype, scale in ((self._HALFWORD_TABLES, np.uint16, 2), (self._BYTE_TABLES, np.uint8, 1)):
            for name in names:
                table = np.asarray(values[name], dtype=dtype)
                start = scale * getattr(self, name) - 4 * self.tables
                image[start : start + table.nbytes] = table.view(np.uint8)
        return image.view(np.uint32)


class DenseTreeOps:
    """Device functions of one dense bucket on a :class:`DenseSharedLayout` workspace.

    Attributes:
        layout: The shared workspace layout.
        stage: Stage an articulation's template tables; returns its height level count.
        factor: Assemble and factor ``M = L^T L`` from the per-pose motion subspaces and inertias.
        restore: Reload a stored ``L`` and stage the motion subspaces it was factored with.
        solve_upper: Overwrite a shared vector ``b`` by ``L^-T b``, leaves first.
        solve_lower: Overwrite a shared vector ``z`` by ``L^-1 z``, root first.
        solve: Overwrite a shared vector ``b`` by ``M^-1 b``.
    """


@functools.cache
def make_dense_tree_ops(
    max_dofs: int, joint_capacity: int, entry_capacity: int, subtree_capacity: int, scratch_words: int = 0
) -> DenseTreeOps:
    """Build the device functions of one dense bucket.

    Kernels that call them compile with :func:`dense_unroll` loop unrolling.
    """
    layout = DenseSharedLayout(max_dofs, joint_capacity, entry_capacity, subtree_capacity, scratch_words)
    inertia_base = layout.scratch
    motion_base = layout.motion
    diagonal_base = layout.diagonal
    dof_entry = layout.dof_entry
    ancestors = layout.ancestors
    two_rows = max_dofs > _LANES
    desc_start = layout.desc_start
    desc_row = layout.desc_row
    level_pair_start = layout.level_pair_start
    level_pair = layout.level_pair
    subtree_start = layout.subtree_start
    path = layout.path
    spare = layout.spare
    dof_joint = layout.dof_joint
    subtree_joint = layout.subtree_joint
    table_base = layout.tables
    table_words = layout.staged_end - layout.tables
    table_rounds = (table_words + _LANES - 1) // _LANES
    # The factorization's own tables are copied over the composite inertias once those are
    # spent, so their halfword offsets shift from the image to the scratch region.
    cold_source = layout.staged_end
    cold_target = layout.scratch
    cold_words = layout.tables_end - layout.staged_end
    cold_rounds = (cold_words + _LANES - 1) // _LANES
    cold_shift = 2 * (layout.scratch - layout.staged_end)
    desc_start += cold_shift
    desc_row += cold_shift
    level_pair_start += cold_shift
    level_pair += cold_shift
    joint_rounds = (joint_capacity + _LANES - 1) // _LANES
    dof_rounds = (max_dofs + _LANES) // _LANES
    motion_rounds = (6 * max_dofs + _LANES - 1) // _LANES
    child_vector = wp.types.vector(length=joint_rounds, dtype=wp.int32)
    inertia_vector = wp.types.vector(length=21 * joint_rounds, dtype=wp.float32)
    motion_vector = wp.types.vector(length=motion_rounds, dtype=wp.float32)
    dof_values = wp.types.vector(length=dof_rounds, dtype=wp.float32)
    entry_rounds = (entry_capacity + _LANES - 1) // _LANES
    factor_vector = wp.types.vector(length=entry_rounds, dtype=wp.float32)
    stage_words = wp.types.vector(length=_STAGE_ROUNDS, dtype=wp.uint32)

    @wp.func
    def stage_motion(
        shared: _shared_handle, lane: wp.int32, dof_start: wp.int32, n: wp.int32, motion: wp.array[wp.float32]
    ):
        """Stage the per-pose motion subspaces as float4-padded shared rows."""
        motion_values = motion_vector()
        for r in range(motion_rounds):
            index = lane + wp.int32(r * _LANES)
            if index < wp.int32(6) * n:
                motion_values[r] = motion[wp.int32(6) * dof_start + index]
        for r in range(motion_rounds):
            index = lane + wp.int32(r * _LANES)
            if index < wp.int32(6) * n:
                dof = index // wp.int32(6)
                _store(
                    shared,
                    wp.int32(motion_base) + wp.int32(_MOTION_STRIDE) * dof + index - wp.int32(6) * dof,
                    motion_values[r],
                )

    @wp.func
    def copy_image_words(
        shared: _shared_handle,
        lane: wp.int32,
        image: wp.array[wp.uint32],
        source: wp.int32,
        target: wp.int32,
        count: wp.int32,
    ):
        """Copy ``count`` global words into the workspace, a chunk of loads in flight at a time.

        A store between two loads would make each global latency wait for the one before.
        """
        for chunk in range(0, table_rounds, _STAGE_ROUNDS):
            words = stage_words()
            for r in range(_STAGE_ROUNDS):
                word = lane + wp.int32((chunk + r) * _LANES)
                if word < count:
                    words[r] = image[source + word]
            for r in range(_STAGE_ROUNDS):
                word = lane + wp.int32((chunk + r) * _LANES)
                if word < count:
                    _store_word(shared, target + word, words[r])

    @wp.func
    def stage(
        shared: _shared_handle,
        articulation: wp.int32,
        lane: wp.int32,
        dense: DenseTreeData,
        template_image: wp.array[wp.uint32],
        template_image_start: wp.array[wp.int32],
    ) -> wp.int32:
        """Copy an articulation's template tables into the workspace; return its height level count."""
        template = dense.articulation_template[articulation]
        source = template_image_start[template]
        copy_image_words(shared, lane, template_image, source, wp.int32(table_base), wp.int32(table_words))
        heights = (
            dense.template_height_start[template + wp.int32(1)] - wp.int32(1) - dense.template_height_start[template]
        )
        _sync_contact_warp()
        return heights

    @wp.func
    def descendant_products(shared: _shared_handle, k: wp.int32, depth: wp.int32, column: wp.int32) -> wp.vec2:
        """Return ``sum_i L_ik L_i,column`` and ``sum_i L_ik^2`` over the descendants ``i`` of ``k``.

        ``L_ik`` sits at ``depth`` in row ``i``. Four independent partial sums
        overlap the shared-memory latencies.
        """
        products = wp.vec4()
        squares = wp.vec4()
        index = _load_u16(shared, wp.int32(desc_start) + k)
        end = _load_u16(shared, wp.int32(desc_start) + k + wp.int32(1))
        while index + wp.int32(3) < end:
            row0 = _load_u16(shared, wp.int32(desc_row) + index)
            row1 = _load_u16(shared, wp.int32(desc_row) + index + wp.int32(1))
            row2 = _load_u16(shared, wp.int32(desc_row) + index + wp.int32(2))
            row3 = _load_u16(shared, wp.int32(desc_row) + index + wp.int32(3))
            l0 = _load(shared, row0 + depth)
            l1 = _load(shared, row1 + depth)
            l2 = _load(shared, row2 + depth)
            l3 = _load(shared, row3 + depth)
            products += wp.vec4(
                l0 * _load(shared, row0 + column),
                l1 * _load(shared, row1 + column),
                l2 * _load(shared, row2 + column),
                l3 * _load(shared, row3 + column),
            )
            squares += wp.vec4(l0 * l0, l1 * l1, l2 * l2, l3 * l3)
            index += wp.int32(4)
        while index < end:
            row0 = _load_u16(shared, wp.int32(desc_row) + index)
            l0 = _load(shared, row0 + depth)
            products[0] += l0 * _load(shared, row0 + column)
            squares[0] += l0 * l0
            index += wp.int32(1)
        return wp.vec2(
            (products[0] + products[1]) + (products[2] + products[3]),
            (squares[0] + squares[1]) + (squares[2] + squares[3]),
        )

    @wp.func
    def factor(
        shared: _shared_handle,
        articulation: wp.int32,
        lane: wp.int32,
        joint_start: wp.int32,
        joint_count: wp.int32,
        dof_start: wp.int32,
        n: wp.int32,
        heights: wp.int32,
        joint_child: wp.array[wp.int32],
        motion: wp.array[wp.float32],
        body_inertia: wp.array[wp.float32],
        factor_diagonal: wp.array[wp.float32],
        dense: DenseTreeData,
        store: wp.bool,
    ):
        """CRBA plus leaves-first ``L^T L`` factorization of one staged articulation on one warp.

        Per-articulation inputs are staged into shared memory by fixed-count,
        unrolled loads, so their global latencies overlap instead of chaining.
        ``store`` also writes ``L`` to ``dense.factor``.
        """
        child = child_vector()
        for r in range(joint_rounds):
            joint = lane + wp.int32(r * _LANES)
            if joint < joint_count:
                child[r] = joint_child[joint_start + joint]
        inertia = inertia_vector()
        for r in range(joint_rounds):
            if lane + wp.int32(r * _LANES) < joint_count:
                for component in range(21):
                    inertia[r * 21 + component] = body_inertia[wp.int32(21) * child[r] + wp.int32(component)]
        diagonal = dof_values()
        for r in range(dof_rounds):
            dof = lane + wp.int32(r * _LANES)
            if dof < n:
                diagonal[r] = factor_diagonal[dof_start + dof]
        for r in range(joint_rounds):
            joint = lane + wp.int32(r * _LANES)
            if joint < joint_count:
                for component in range(21):
                    _store(
                        shared,
                        wp.int32(inertia_base) + wp.int32(_INERTIA_STRIDE) * joint + wp.int32(component),
                        inertia[r * 21 + component],
                    )
        stage_motion(shared, lane, dof_start, n, motion)
        for r in range(dof_rounds):
            dof = lane + wp.int32(r * _LANES)
            if dof < n:
                _store(shared, wp.int32(diagonal_base) + dof, diagonal[r])
        _sync_contact_warp()

        # Composite spatial inertias: each lane sums its joints' subtrees in registers.
        for r in range(joint_rounds):
            joint = lane + wp.int32(r * _LANES)
            if joint < joint_count:
                composite = _vec24()
                for index in range(
                    _load_u16(shared, wp.int32(subtree_start) + joint),
                    _load_u16(shared, wp.int32(subtree_start) + joint + wp.int32(1)),
                ):
                    composite += _load24(
                        shared,
                        wp.int32(inertia_base)
                        + wp.int32(_INERTIA_STRIDE) * _load_u8(shared, wp.int32(subtree_joint) + index),
                    )
                for component in range(21):
                    inertia[r * 21 + component] = composite[component]
        _sync_contact_warp()
        for r in range(joint_rounds):
            joint = lane + wp.int32(r * _LANES)
            if joint < joint_count:
                for component in range(21):
                    _store(
                        shared,
                        wp.int32(inertia_base) + wp.int32(_INERTIA_STRIDE) * joint + wp.int32(component),
                        inertia[r * 21 + component],
                    )
        _sync_contact_warp()

        # CRBA in path layout: M[i, k] = S_k . (I_c(joint(i)) S_i) for the ancestors k of i.
        row = lane
        while row < n:
            composite = _load24(
                shared,
                wp.int32(inertia_base) + wp.int32(_INERTIA_STRIDE) * _load_u8(shared, wp.int32(dof_joint) + row),
            )
            momentum = _sym66_times(
                composite, _load_motion(shared, wp.int32(motion_base) + wp.int32(_MOTION_STRIDE) * row)
            )
            entry = _load_u16(shared, wp.int32(dof_entry) + row)
            diagonal_entry = _load_u16(shared, wp.int32(dof_entry) + row + wp.int32(1)) - wp.int32(1)
            while entry < diagonal_entry:
                column = _load_u8(shared, wp.int32(path) + entry)
                _store(
                    shared,
                    entry,
                    wp.dot(_load_motion(shared, wp.int32(motion_base) + wp.int32(_MOTION_STRIDE) * column), momentum),
                )
                entry += wp.int32(1)
            # Keep the assembled diagonal for the relative pivot floor.
            value = wp.dot(_load_motion(shared, wp.int32(motion_base) + wp.int32(_MOTION_STRIDE) * row), momentum)
            value += _load(shared, wp.int32(diagonal_base) + row)
            _store(shared, wp.int32(diagonal_base) + row, value)
            _store(shared, diagonal_entry, value)
            row += wp.int32(_LANES)
        _sync_contact_warp()

        # The composite inertias are spent: stage the factorization's tables over them.
        for chunk in range(0, cold_rounds, _STAGE_ROUNDS):
            words = stage_words()
            for r in range(_STAGE_ROUNDS):
                word = lane + wp.int32((chunk + r) * _LANES)
                if word < wp.int32(cold_words):
                    words[r] = _image_word(shared, wp.int32(cold_source) + word)
            for r in range(_STAGE_ROUNDS):
                word = lane + wp.int32((chunk + r) * _LANES)
                if word < wp.int32(cold_words):
                    _store_word(shared, wp.int32(cold_target) + word, words[r])
        _sync_contact_warp()

        # M = L^T L by height levels: lane (k, t) forms entry t of row k from the scaled
        # rows of k's descendants, H_kt - sum_i L_ik L_it, and scales it by L_kk, which it
        # forms alike. Rows of equal height never read each other.
        for level in range(heights):
            pair = _load_u16(shared, wp.int32(level_pair_start) + level) + lane
            pair_end = _load_u16(shared, wp.int32(level_pair_start) + level + wp.int32(1))
            while pair < pair_end:
                packed = _load_u16(shared, wp.int32(level_pair) + pair)
                k = packed >> wp.int32(8)
                t = packed & wp.int32(0xFF)
                entry = _load_u16(shared, wp.int32(dof_entry) + k)
                depth = _load_u16(shared, wp.int32(dof_entry) + k + wp.int32(1)) - wp.int32(1) - entry
                # The assembled diagonal, as the diagonal lane may already store L_kk.
                assembled = _load(shared, wp.int32(diagonal_base) + k)
                sums = descendant_products(shared, k, depth, t)
                value = _load(shared, entry + t) - sums[0]
                pivot = wp.sqrt(wp.max(assembled - sums[1], _PIVOT_FLOOR * assembled))
                if t == depth:
                    value = pivot
                else:
                    value = value / pivot
                _store(shared, entry + t, value)
                pair += wp.int32(_LANES)
            _sync_contact_warp()

        if store:
            base = dense.factor_start[articulation]
            entry_count = _load_u16(shared, wp.int32(dof_entry) + n)
            entry = lane
            while entry < entry_count:
                dense.factor[base + entry] = _load(shared, entry)
                entry += wp.int32(_LANES)

    @wp.func
    def restore(
        shared: _shared_handle,
        articulation: wp.int32,
        lane: wp.int32,
        dof_start: wp.int32,
        n: wp.int32,
        motion: wp.array[wp.float32],
        dense: DenseTreeData,
    ):
        stage_motion(shared, lane, dof_start, n, motion)
        base = dense.factor_start[articulation]
        entry_count = _load_u16(shared, wp.int32(dof_entry) + n)
        values = factor_vector()
        for r in range(entry_rounds):
            entry = lane + wp.int32(r * _LANES)
            if entry < entry_count:
                values[r] = dense.factor[base + entry]
        for r in range(entry_rounds):
            entry = lane + wp.int32(r * _LANES)
            if entry < entry_count:
                _store(shared, entry, values[r])
        _sync_contact_warp()

    @wp.func
    def load_row(shared: _shared_handle, vector_base: wp.int32, n: wp.int32, row: wp.int32) -> wp.vec4:
        """Return a lane's row ``(b_row, 1 / L_row,row, row entry, depth of row)``, zero beyond ``n``."""
        result = wp.vec4(0.0)
        if row < n:
            entry = _load_u16(shared, wp.int32(dof_entry) + row)
            diagonal_entry = _load_u16(shared, wp.int32(dof_entry) + row + wp.int32(1)) - wp.int32(1)
            result = wp.vec4(
                _load(shared, vector_base + row),
                wp.float32(1.0) / _load(shared, diagonal_entry),
                wp.float32(entry),
                wp.float32(diagonal_entry - entry),
            )
        return result

    @wp.func
    def store_row(shared: _shared_handle, vector_base: wp.int32, n: wp.int32, row: wp.int32, value: wp.float32):
        if row < n:
            _store(shared, vector_base + row, value)

    @wp.func
    def solve_upper(shared: _shared_handle, vector_base: wp.int32, n: wp.int32, lane: wp.int32):
        """Overwrite a shared vector ``b`` by ``z = L^-T b`` one column per step, leaves first.

        Lane ``l`` holds ``b_l`` and ``b_{l+32}`` in registers. Step ``k`` broadcasts
        ``z_k = b_k / L_kk`` and every ancestor ``j`` of ``k`` subtracts ``L_kj z_k``;
        parent-first DOF numbering finishes each ``b_k`` before its step. Selects
        instead of branches keep the warp converged around each step's shuffle.
        """
        row0 = load_row(shared, vector_base, n, lane)
        value0 = row0[0]
        depth0 = wp.int32(row0[3])
        row1 = wp.vec4(0.0)
        value1 = wp.float32(0.0)
        depth1 = wp.int32(0)
        if wp.static(two_rows):
            row1 = load_row(shared, vector_base, n, lane + wp.int32(_LANES))
            value1 = row1[0]
            depth1 = wp.int32(row1[3])
        for step in range(n):
            k = n - wp.int32(1) - step
            entry_k = _load_u16(shared, wp.int32(dof_entry) + k)
            mask = _load_int(shared, wp.int32(ancestors) + wp.int32(2) * k)
            factor0 = _load(shared, entry_k + depth0)
            candidate = value0 * row0[1]
            if wp.static(two_rows):
                candidate = wp.where(k >= wp.int32(_LANES), value1 * row1[1], candidate)
            z = _lane_float(candidate, k & wp.int32(_LANES - 1))
            factor0 = wp.where(((mask >> lane) & wp.int32(1)) != wp.int32(0), factor0, wp.float32(0.0))
            value0 = wp.where(lane == k, z, value0 - factor0 * z)
            if wp.static(two_rows):
                mask = _load_int(shared, wp.int32(ancestors) + wp.int32(2) * k + wp.int32(1))
                factor1 = _load(shared, entry_k + depth1)
                factor1 = wp.where(((mask >> lane) & wp.int32(1)) != wp.int32(0), factor1, wp.float32(0.0))
                value1 = wp.where(lane + wp.int32(_LANES) == k, z, value1 - factor1 * z)
        store_row(shared, vector_base, n, lane, value0)
        if wp.static(two_rows):
            store_row(shared, vector_base, n, lane + wp.int32(_LANES), value1)
        _sync_contact_warp()

    @wp.func
    def solve_lower(shared: _shared_handle, vector_base: wp.int32, n: wp.int32, lane: wp.int32):
        """Overwrite a shared vector ``z`` by ``x = L^-1 z`` level by level, root first.

        The rows at path depth ``d`` finish ``x_i = z_i / L_ii``; every deeper row
        then subtracts ``L_ij x_j`` for its one ancestor ``j`` at that depth, the
        entry at position ``d`` of its row. A level costs a few shared accesses,
        where broadcasting one column per step cost a shuffle per DOF. Every lane
        stores its value at every level, so the warp never branches on a depth.
        """
        row0 = load_row(shared, vector_base, n, lane)
        value0 = row0[0]
        entry0 = wp.int32(row0[2])
        depth0 = wp.where(lane < n, wp.int32(row0[3]), wp.int32(-1))
        slot0 = wp.where(lane < n, vector_base + lane, wp.int32(spare))
        row1 = wp.vec4(0.0)
        value1 = wp.float32(0.0)
        entry1 = wp.int32(0)
        depth1 = wp.int32(-1)
        slot1 = wp.int32(spare)
        if wp.static(two_rows):
            row1 = load_row(shared, vector_base, n, lane + wp.int32(_LANES))
            value1 = row1[0]
            entry1 = wp.int32(row1[2])
            depth1 = wp.where(lane + wp.int32(_LANES) < n, wp.int32(row1[3]), wp.int32(-1))
            slot1 = wp.where(lane + wp.int32(_LANES) < n, vector_base + lane + wp.int32(_LANES), wp.int32(spare))
        for level in range(_warp_max_int(wp.max(depth0, depth1)) + wp.int32(1)):
            value0 = wp.where(depth0 == level, value0 * row0[1], value0)
            _store(shared, slot0, value0)
            if wp.static(two_rows):
                value1 = wp.where(depth1 == level, value1 * row1[1], value1)
                _store(shared, slot1, value1)
            _sync_contact_warp()
            # Rows at or above this level reread an entry of their own and leave their value.
            index0 = entry0 + wp.min(level, wp.max(depth0, wp.int32(0)))
            reduced0 = value0 - _load(shared, index0) * _load(
                shared, vector_base + _load_u8(shared, wp.int32(path) + index0)
            )
            value0 = wp.where(depth0 > level, reduced0, value0)
            if wp.static(two_rows):
                index1 = entry1 + wp.min(level, wp.max(depth1, wp.int32(0)))
                reduced1 = value1 - _load(shared, index1) * _load(
                    shared, vector_base + _load_u8(shared, wp.int32(path) + index1)
                )
                value1 = wp.where(depth1 > level, reduced1, value1)

    @wp.func
    def solve(shared: _shared_handle, vector_base: wp.int32, n: wp.int32, lane: wp.int32):
        solve_upper(shared, vector_base, n, lane)
        solve_lower(shared, vector_base, n, lane)

    ops = DenseTreeOps()
    ops.layout = layout
    ops.stage = stage
    ops.factor = factor
    ops.restore = restore
    ops.solve_upper = solve_upper
    ops.solve_lower = solve_lower
    ops.solve = solve
    return ops


def dense_unroll(layout: DenseSharedLayout, joint_capacity: int) -> int:
    """Return the loop unrolling the staged loads of :class:`DenseTreeOps` need."""
    table_rounds = (layout.staged_end - layout.tables + _LANES - 1) // _LANES
    entry_rounds = (layout.motion + _LANES - 1) // _LANES
    cold_rounds = (layout.tables_end - layout.staged_end + _LANES - 1) // _LANES
    return max(_STAGE_ROUNDS, table_rounds, cold_rounds, entry_rounds, 21 * ((joint_capacity + _LANES - 1) // _LANES))


@functools.cache
def _make_dense_factor_kernel(max_dofs: int, joint_capacity: int, entry_capacity: int, subtree_capacity: int):
    """Factor each articulation of one bucket into ``dense.factor``."""
    ops = make_dense_tree_ops(max_dofs, joint_capacity, entry_capacity, subtree_capacity)
    shared_base = _make_shared_base(ops.layout.size, ops.layout.tables)
    stage = ops.stage
    factor = ops.factor
    module = _unrolled_module(
        f"dense_tree_factor_{max_dofs}_{joint_capacity}_{entry_capacity}_{subtree_capacity}",
        dense_unroll(ops.layout, joint_capacity),
    )

    @wp.kernel(enable_backward=False, module=module)
    def dense_factor_kernel(
        bucket_articulation: wp.array[wp.int32],
        articulation_start: wp.array[wp.int32],
        articulation_end: wp.array[wp.int32],
        joint_child: wp.array[wp.int32],
        joint_qd_start: wp.array[wp.int32],
        motion: wp.array[wp.float32],
        body_inertia: wp.array[wp.float32],
        factor_diagonal: wp.array[wp.float32],
        template_image: wp.array[wp.uint32],
        template_image_start: wp.array[wp.int32],
        dense: DenseTreeData,
    ):
        thread = wp.tid()
        block = thread // wp.int32(_LANES)
        lane = thread - block * wp.int32(_LANES)
        articulation = bucket_articulation[block]
        joint_start = articulation_start[articulation]
        joint_count = articulation_end[articulation] - joint_start
        if joint_count <= wp.int32(0):
            # A runtime instance that is not in the scene: its whole block leaves.
            return
        dof_start = joint_qd_start[joint_start]
        n = joint_qd_start[articulation_end[articulation]] - dof_start
        shared = shared_base(template_image, template_image_start[dense.articulation_template[articulation]])
        heights = stage(shared, articulation, lane, dense, template_image, template_image_start)
        factor(
            shared,
            articulation,
            lane,
            joint_start,
            joint_count,
            dof_start,
            n,
            heights,
            joint_child,
            motion,
            body_inertia,
            factor_diagonal,
            dense,
            wp.bool(True),
        )

    return dense_factor_kernel


def dense_link_path_group(
    max_dofs: int, joint_capacity: int, entry_capacity: int, path_capacity: int, subtree_capacity: int
) -> int:
    """Return how many links of a bucket's trees form their paths at once in shared memory.

    See :func:`_make_dense_link_response_kernel`; a group of all the lanes, or of all a
    bucket's joints, takes every link at once.
    """
    scratch = DenseSharedLayout(max_dofs, joint_capacity, entry_capacity, subtree_capacity).scratch
    group = min(_LANES, joint_capacity, (_SHARED_WORDS - scratch - 1 - max_dofs) // (6 * path_capacity + 1))
    if group < 1:
        raise ValueError(f"the link paths of {max_dofs} DOFs on {path_capacity} positions exceed shared memory")
    return group


@functools.cache
def _make_dense_link_response_kernel(
    max_dofs: int, joint_capacity: int, entry_capacity: int, path_capacity: int, subtree_capacity: int
):
    """Factor the flagged trees of one bucket and form their links' responses from the factor.

    A tree is flagged when a column between trees touches it or its links hold deferred
    wrenches. Per link the kernel forms ``Z = L^-T J^T`` along the root path, where ``J`` is
    the link's Jacobian about the articulation origin and ``M = L^T L`` the factor, and the
    link's response to a wrench on itself, ``Z^T Z``. Two links of one tree couple through the
    positions their paths share. The tree's deferred rates then start from the wrenches its
    links already hold. Each lane owns one link and keeps its path in shared memory, next to
    the factor and its tables.
    """
    # Per link the six components of z at each path position; the odd stride keeps the links
    # of one position, and the positions of one link, on different banks. The links take
    # their paths in groups that fit the static shared memory with the rates.
    link_stride = 6 * path_capacity + 1
    group = dense_link_path_group(max_dofs, joint_capacity, entry_capacity, path_capacity, subtree_capacity)
    ops = make_dense_tree_ops(
        max_dofs, joint_capacity, entry_capacity, subtree_capacity, max_dofs + group * link_stride
    )
    layout = ops.layout
    shared_base = _make_shared_base(layout.size, layout.tables)
    stage = ops.stage
    factor = ops.factor
    dof_entry = layout.dof_entry
    path_table = layout.path
    joint_last_dof = layout.joint_last_dof
    motion_base = layout.motion
    # The factorization's tables are spent: its scratch region takes the rates and the paths.
    rate_base = layout.scratch
    z_base = rate_base + max_dofs
    module = _unrolled_module(
        f"dense_tree_link_response_{max_dofs}_{joint_capacity}_{entry_capacity}_{path_capacity}_{subtree_capacity}",
        dense_unroll(layout, joint_capacity),
    )

    @wp.func
    def load_z(shared: _shared_handle, slot: wp.int32) -> wp.spatial_vector:
        return wp.spatial_vector(
            _load(shared, slot),
            _load(shared, slot + 1),
            _load(shared, slot + 2),
            _load(shared, slot + 3),
            _load(shared, slot + 4),
            _load(shared, slot + 5),
        )

    @wp.kernel(enable_backward=False, module=module)
    def dense_link_response_kernel(
        bucket_articulation: wp.array[wp.int32],
        motion: wp.array[wp.float32],
        body_inertia: wp.array[wp.float32],
        factor_diagonal: wp.array[wp.float32],
        template_image: wp.array[wp.uint32],
        template_image_start: wp.array[wp.int32],
        cross_tree: wp.array[wp.int32],
        bodies: BodyContainer,
    ):
        thread = wp.tid()
        block = thread // wp.int32(_LANES)
        lane = thread - block * wp.int32(_LANES)
        articulation = bucket_articulation[block]
        data = bodies.reduced
        dense = data.dense
        joint_start = data.articulation_start[articulation]
        joint_count = data.articulation_end[articulation] - joint_start
        if joint_count <= wp.int32(0):
            # A runtime instance that is not in the scene: its whole block leaves.
            return
        if cross_tree[articulation] != wp.int32(0):
            # A tree in a column between trees holds deferred wrenches from here on.
            data.deferred_pending[articulation] = wp.int32(1)
        elif data.deferred_pending[articulation] == wp.int32(0):
            return
        dof_start = data.joint_qd_start[joint_start]
        n = data.joint_qd_start[data.articulation_end[articulation]] - dof_start
        shared = shared_base(template_image, template_image_start[dense.articulation_template[articulation]])
        heights = stage(shared, articulation, lane, dense, template_image, template_image_start)
        factor(
            shared,
            articulation,
            lane,
            joint_start,
            joint_count,
            dof_start,
            n,
            heights,
            data.joint_child,
            motion,
            body_inertia,
            factor_diagonal,
            dense,
            wp.bool(True),
        )

        # Deferred rates from the wrenches the links hold, link after link: few links hold
        # one. A DOF sits at the same path position, its depth, on every path through it, so
        # lane p sums the DOFs at depth p.
        k = lane
        while k < n:
            _store(shared, wp.int32(rate_base) + k, wp.float32(0.0))
            k += wp.int32(_LANES)
        wrench = wp.spatial_vector()
        path_entry = wp.int32(0)
        path_length = wp.int32(0)
        for group_start in range(0, joint_count, wp.static(group)):
            member = lane >= group_start and lane < wp.min(group_start + wp.int32(group), joint_count)
            z_link = wp.int32(z_base) + (lane - group_start) * wp.int32(link_stride)
            if member:
                link = data.joint_child[joint_start + lane]
                wrench = data.deferred_wrench[link]
                leaf = _load_u8(shared, wp.int32(joint_last_dof) + lane) - wp.int32(1)
                response0 = wp.spatial_vector()
                response1 = wp.spatial_vector()
                response2 = wp.spatial_vector()
                response3 = wp.spatial_vector()
                response4 = wp.spatial_vector()
                response5 = wp.spatial_vector()
                if leaf >= wp.int32(0):
                    path_entry = _load_u16(shared, wp.int32(dof_entry) + leaf)
                    path_length = _load_u16(shared, wp.int32(dof_entry) + leaf + wp.int32(1)) - path_entry
                    # L^T z = J^T, leaves first, for the six wrench axes at once.
                    for reverse in range(path_length):
                        position = path_length - wp.int32(1) - reverse
                        k = _load_u8(shared, wp.int32(path_table) + path_entry + position)
                        value = _load_motion(shared, wp.int32(motion_base) + wp.int32(_MOTION_STRIDE) * k)
                        for later in range(position + wp.int32(1), path_length):
                            row = _load_u8(shared, wp.int32(path_table) + path_entry + later)
                            value -= _load(shared, _load_u16(shared, wp.int32(dof_entry) + row) + position) * load_z(
                                shared, z_link + wp.int32(6) * later
                            )
                        entry = value / _load(shared, _load_u16(shared, wp.int32(dof_entry) + k + wp.int32(1)) - 1)
                        for component in range(6):
                            _store(shared, z_link + wp.int32(6) * position + wp.int32(component), entry[component])
                        data.link_path[link, position] = entry
                        data.link_path_dof[link, position] = dof_start + k
                        response0 += entry * entry[0]
                        response1 += entry * entry[1]
                        response2 += entry * entry[2]
                        response3 += entry * entry[3]
                        response4 += entry * entry[4]
                        response5 += entry * entry[5]
                if path_length < data.link_path.shape[1]:
                    data.link_path_dof[link, path_length] = wp.int32(-1)
                data.impulse_response[link, 0] = response0
                data.impulse_response[link, 1] = response1
                data.impulse_response[link, 2] = response2
                data.impulse_response[link, 3] = response3
                data.impulse_response[link, 4] = response4
                data.impulse_response[link, 5] = response5
            _sync_contact_warp()
            holding = _warp_ballot(member and path_length > wp.int32(0) and wp.length_sq(wrench) > wp.float32(0.0))
            while holding != wp.int32(0):
                joint = _lowest_lane(holding)
                holding = holding & (holding - wp.int32(1))
                w = wp.spatial_vector(
                    _lane_float(wrench[0], joint),
                    _lane_float(wrench[1], joint),
                    _lane_float(wrench[2], joint),
                    _lane_float(wrench[3], joint),
                    _lane_float(wrench[4], joint),
                    _lane_float(wrench[5], joint),
                )
                entry_start = _lane_int(path_entry, joint)
                length = _lane_int(path_length, joint)
                z_joint = wp.int32(z_base) + (joint - group_start) * wp.int32(link_stride)
                position = lane
                while position < length:
                    slot = wp.int32(rate_base) + _load_u8(shared, wp.int32(path_table) + entry_start + position)
                    _store(
                        shared, slot, _load(shared, slot) + wp.dot(load_z(shared, z_joint + wp.int32(6) * position), w)
                    )
                    position += wp.int32(_LANES)
            # The next group's paths take the same words.
            _sync_contact_warp()
        k = lane
        while k < n:
            data.deferred_rate[dof_start + k] = _load(shared, wp.int32(rate_base) + k)
            k += wp.int32(_LANES)

    return dense_link_response_kernel


@functools.cache
def _make_dense_contact_rows_kernel(max_dofs: int, entry_capacity: int, patch_rows: bool):
    """Contact rows ``J`` and responses ``W = M^-1 J^T`` from the dense factor.

    The factor and its tables are staged into shared memory. Each lane owns one
    row and solves it serially in its own shared-memory column, so all lanes
    walk the factor in lockstep and share its loads.
    """
    factor_base = _LANES * max_dofs
    path_base_shared = factor_base + entry_capacity
    desc_row_base = path_base_shared + entry_capacity
    desc_dof_base = desc_row_base + entry_capacity
    dof_entry_base = desc_dof_base + entry_capacity
    desc_start_base = dof_entry_base + max_dofs + 1
    motion_base = 4 * ((desc_start_base + max_dofs + 4) // 4)
    shared_base = _make_shared_base(motion_base + _MOTION_STRIDE * max_dofs)
    entry_chunks = (entry_capacity // _LANES + _STAGE_ROUNDS - 1) // _STAGE_ROUNDS
    dof_rounds = (max_dofs + _LANES) // _LANES
    motion_rounds = (6 * max_dofs + _LANES - 1) // _LANES
    chunk_values = wp.types.vector(length=_STAGE_ROUNDS, dtype=wp.float32)
    chunk_indices = wp.types.vector(length=_STAGE_ROUNDS, dtype=wp.int32)
    dof_indices = wp.types.vector(length=dof_rounds, dtype=wp.int32)
    motion_vector = wp.types.vector(length=motion_rounds, dtype=wp.float32)
    module = _unrolled_module(
        f"dense_tree_contact_rows_{max_dofs}_{entry_capacity}_{int(patch_rows)}", max(_STAGE_ROUNDS, motion_rounds)
    )

    @wp.func
    def add_path(
        bodies: BodyContainer,
        shared: _shared_handle,
        body: wp.int32,
        wrench: wp.spatial_vector,
        joint_start: wp.int32,
        joint_last_dof: wp.array[wp.int32],
        joint_base: wp.int32,
        lane: wp.int32,
    ) -> wp.int32:
        """Add ``S_k . wrench`` along the root path of ``body`` to the lane's column.

        Returns the last DOF of that path, or -1 when ``body`` moves with no DOF.
        """
        dof = joint_last_dof[joint_base + bodies.reduced.body_joint[body] - joint_start]
        if dof >= wp.int32(0):
            entry_end = _load_int(shared, wp.int32(dof_entry_base) + dof + wp.int32(1))
            for entry in range(_load_int(shared, wp.int32(dof_entry_base) + dof), entry_end):
                column = _load_int(shared, wp.int32(path_base_shared) + entry)
                slot = wp.int32(_LANES) * column + lane
                value = wp.dot(_load_motion(shared, wp.int32(motion_base) + wp.int32(_MOTION_STRIDE) * column), wrench)
                _store(shared, slot, _load(shared, slot) + value)
        return dof

    @wp.kernel(enable_backward=False, module=module)
    def dense_contact_rows_kernel(
        bucket_articulation: wp.array[wp.int32],
        bodies: BodyContainer,
        enabled: wp.array[wp.int32],
        point_count: wp.array[wp.int32],
        row_count_array: wp.array[wp.int32],
        row_body: wp.array2d[wp.int32],
        row_wrench: wp.array2d[wp.spatial_vector],
        row_body_pair: wp.array2d[wp.int32],
        row_wrench_pair: wp.array2d[wp.spatial_vector],
        point_contact: wp.array2d[wp.int32],
        cc: ContactContainer,
        max_page_count: wp.array[wp.int32],
        page_index: wp.array[wp.int32],
        prepare: wp.bool,
        joint_last_dof: wp.array[wp.int32],
        template_joint_start: wp.array[wp.int32],
        motion: wp.array[wp.float32],
        packed_jacobian: wp.array2d[wp.float32],
        packed_response: wp.array2d[wp.float32],
    ):
        thread = wp.tid()
        block = thread // wp.int32(_LANES)
        lane = thread - block * wp.int32(_LANES)
        articulation = bucket_articulation[block]
        if enabled[articulation] == wp.int32(0):
            return
        page = page_index[0]
        if not prepare and (
            page == wp.int32(0) or (page == wp.int32(1) and max_page_count[0] <= wp.int32(_CACHED_PAGE_COUNT))
        ):
            return
        data = bodies.reduced
        dense = data.dense
        joint_start = data.articulation_start[articulation]
        dof_start = data.joint_qd_start[joint_start]
        n = data.joint_qd_start[data.articulation_end[articulation]] - dof_start
        base = dense.factor_start[articulation]
        template = dense.articulation_template[articulation]
        dof_base = dense.template_dof_start[template]
        path_base = dense.template_path_start[template]
        joint_base = template_joint_start[template]
        storage_page = wp.min(page, wp.int32(_CACHED_PAGE_COUNT - 1))
        packed_articulation = articulation * wp.int32(_CACHED_PAGE_COUNT) + storage_page
        row_count = row_count_array[packed_articulation]
        shared = shared_base()

        # Stage the factor, its paths, the descendant lists, and the row starts.
        entry_count = dense.dof_entry[dof_base + n]
        desc_first = dense.desc_start[dof_base]
        desc_count = dense.desc_start[dof_base + n] - desc_first
        # Fixed-size chunks bound the registers that hold loads in flight.
        for chunk in range(entry_chunks):
            factor_values = chunk_values()
            path_values = chunk_indices()
            desc_rows = chunk_indices()
            desc_dofs = chunk_indices()
            for r in range(_STAGE_ROUNDS):
                entry = lane + (chunk * wp.int32(_STAGE_ROUNDS) + wp.int32(r)) * wp.int32(_LANES)
                if entry < entry_count:
                    factor_values[r] = dense.factor[base + entry]
                    path_values[r] = dense.path[path_base + entry]
                if entry < desc_count:
                    desc_rows[r] = dense.desc_row[desc_first + entry]
                    desc_dofs[r] = dense.desc_dof[desc_first + entry]
            for r in range(_STAGE_ROUNDS):
                entry = lane + (chunk * wp.int32(_STAGE_ROUNDS) + wp.int32(r)) * wp.int32(_LANES)
                if entry < entry_count:
                    _store(shared, wp.int32(factor_base) + entry, factor_values[r])
                    _store_int(shared, wp.int32(path_base_shared) + entry, path_values[r])
                if entry < desc_count:
                    _store_int(shared, wp.int32(desc_row_base) + entry, desc_rows[r])
                    _store_int(shared, wp.int32(desc_dof_base) + entry, desc_dofs[r])
        dof_entry = dof_indices()
        desc_start = dof_indices()
        for r in range(dof_rounds):
            dof = lane + wp.int32(r * _LANES)
            if dof <= n:
                dof_entry[r] = dense.dof_entry[dof_base + dof]
                desc_start[r] = dense.desc_start[dof_base + dof] - desc_first
        motion_values = motion_vector()
        for r in range(motion_rounds):
            index = lane + wp.int32(r * _LANES)
            if index < wp.int32(6) * n:
                motion_values[r] = motion[wp.int32(6) * dof_start + index]
        for r in range(motion_rounds):
            index = lane + wp.int32(r * _LANES)
            if index < wp.int32(6) * n:
                dof = index // wp.int32(6)
                _store(
                    shared,
                    wp.int32(motion_base) + wp.int32(_MOTION_STRIDE) * dof + index - wp.int32(6) * dof,
                    motion_values[r],
                )
        for r in range(dof_rounds):
            dof = lane + wp.int32(r * _LANES)
            if dof <= n:
                _store_int(shared, wp.int32(dof_entry_base) + dof, dof_entry[r])
                _store_int(shared, wp.int32(desc_start_base) + dof, desc_start[r])
        _sync_contact_warp()

        row = lane
        while row < row_count:
            packed_row = storage_page * wp.int32(_MAX_ROWS) + row
            for dof in range(n):
                _store(shared, wp.int32(_LANES) * dof + lane, 0.0)
            body = row_body[packed_articulation, row]
            wrench = row_wrench[packed_articulation, row]
            path_dof = add_path(bodies, shared, body, wrench, joint_start, joint_last_dof, joint_base, lane)
            body_pair = wp.int32(-1)
            wrench_pair = wp.spatial_vector()
            if wp.static(not patch_rows):
                body_pair = row_body_pair[packed_articulation, row]
                if body_pair >= wp.int32(0):
                    wrench_pair = row_wrench_pair[packed_articulation, row]
                    add_path(bodies, shared, body_pair, wrench_pair, joint_start, joint_last_dof, joint_base, lane)
            for dof in range(n):
                packed_jacobian[packed_row, dof_start + dof] = _load(shared, wp.int32(_LANES) * dof + lane)

            # z = L^-T J^T, leaves first; J M^-1 J^T = |z|^2. A one-body row is
            # supported on the root path of its body only, and so is z.
            inverse_mass = wp.float32(0.0)
            path_entry = wp.int32(0)
            path_length = wp.int32(0)
            if path_dof >= wp.int32(0):
                path_entry = _load_int(shared, wp.int32(dof_entry_base) + path_dof)
                path_length = _load_int(shared, wp.int32(dof_entry_base) + path_dof + wp.int32(1)) - path_entry
            if body_pair >= wp.int32(0):
                path_length = wp.int32(0)
            for reverse in range(path_length):
                position = path_length - wp.int32(1) - reverse
                k = _load_int(shared, wp.int32(path_base_shared) + path_entry + position)
                slot = wp.int32(_LANES) * k + lane
                value = _load(shared, slot)
                for later in range(position + wp.int32(1), path_length):
                    i = _load_int(shared, wp.int32(path_base_shared) + path_entry + later)
                    value -= _load(
                        shared, wp.int32(factor_base) + _load_int(shared, wp.int32(dof_entry_base) + i) + position
                    ) * _load(shared, wp.int32(_LANES) * i + lane)
                value /= _load(
                    shared, wp.int32(factor_base) + _load_int(shared, wp.int32(dof_entry_base) + k + wp.int32(1)) - 1
                )
                _store(shared, slot, value)
                inverse_mass += value * value
            pair_count = n
            if body_pair < wp.int32(0):
                pair_count = wp.int32(0)
            for reverse in range(pair_count):
                k = n - wp.int32(1) - reverse
                entry = _load_int(shared, wp.int32(dof_entry_base) + k)
                diagonal_entry = _load_int(shared, wp.int32(dof_entry_base) + k + wp.int32(1)) - wp.int32(1)
                depth = diagonal_entry - entry
                slot = wp.int32(_LANES) * k + lane
                value = _load(shared, slot) - _descendant_solve_sum(
                    shared,
                    wp.int32(desc_row_base),
                    wp.int32(desc_dof_base),
                    _load_int(shared, wp.int32(desc_start_base) + k),
                    _load_int(shared, wp.int32(desc_start_base) + k + wp.int32(1)),
                    wp.int32(factor_base) + depth,
                    lane,
                )
                value /= _load(shared, wp.int32(factor_base) + diagonal_entry)
                _store(shared, slot, value)
                inverse_mass += value * value
            # W = L^-1 z, root first.
            for i in range(n):
                entry = _load_int(shared, wp.int32(dof_entry_base) + i)
                diagonal_entry = _load_int(shared, wp.int32(dof_entry_base) + i + wp.int32(1)) - wp.int32(1)
                slot = wp.int32(_LANES) * i + lane
                value = _load(shared, slot) - _ancestor_solve_sum(
                    shared, wp.int32(path_base_shared), wp.int32(factor_base), entry, diagonal_entry, lane
                )
                value /= _load(shared, wp.int32(factor_base) + diagonal_entry)
                _store(shared, slot, value)
                packed_response[packed_row, dof_start + i] = value
            if wp.static(not patch_rows):
                _store_point_row_effective_mass(
                    bodies,
                    cc,
                    point_contact,
                    packed_articulation,
                    row,
                    body,
                    wrench,
                    body_pair,
                    wrench_pair,
                    inverse_mass,
                    prepare,
                )
            row += wp.int32(_LANES)

    return dense_contact_rows_kernel


def _articulation_template(joint_parent_local: list[int], joint_dof_count: list[int]) -> dict:
    """Build the path, descendant, and level tables of one local articulation topology."""
    joint_count = len(joint_parent_local)
    joint_dof_start = np.concatenate(([0], np.cumsum(joint_dof_count))).astype(int)
    dof_count = int(joint_dof_start[-1])
    # DOFs of one joint form a chain below the last DOF of the nearest ancestor joint with DOFs.
    joint_last_dof = [-1] * joint_count
    dof_parent = [-1] * dof_count
    dof_joint = [0] * dof_count
    for joint in range(joint_count):
        parent = joint_parent_local[joint]
        inherited = joint_last_dof[parent] if parent >= 0 else -1
        for dof in range(joint_dof_start[joint], joint_dof_start[joint + 1]):
            dof_parent[dof] = inherited
            dof_joint[dof] = joint
            inherited = dof
        joint_last_dof[joint] = inherited
    paths: list[list[int]] = []
    for dof in range(dof_count):
        parent = dof_parent[dof]
        paths.append([*(paths[parent] if parent >= 0 else []), dof])
    dof_entry = np.concatenate(([0], np.cumsum([len(path) for path in paths]))).astype(np.int32)
    descendants = [[i for i in range(dof_count) if i != k and k in paths[i]] for k in range(dof_count)]
    height = [0] * dof_count
    for dof in reversed(range(dof_count)):
        if dof_parent[dof] >= 0:
            height[dof_parent[dof]] = max(height[dof_parent[dof]], height[dof] + 1)
    height_levels = [[k for k in range(dof_count) if height[k] == h] for h in range(max(height, default=-1) + 1)]
    depth_levels = [
        [k for k in range(dof_count) if len(paths[k]) == d + 1] for d in range(max(map(len, paths), default=0))
    ]
    # Factor pairs (k, t) of each height level, packed as k << 16 | t.
    level_pairs = [[k << 16 | t for k in level for t in range(len(paths[k]))] for level in height_levels]
    joint_depth = [0] * joint_count
    for joint in range(joint_count):
        if joint_parent_local[joint] >= 0:
            joint_depth[joint] = joint_depth[joint_parent_local[joint]] + 1
    subtrees: list[list[int]] = [[joint] for joint in range(joint_count)]
    for joint in reversed(range(joint_count)):
        parent = joint_parent_local[joint]
        if parent >= 0:
            subtrees[parent].extend(subtrees[joint])

    def starts(lists) -> np.ndarray:
        return np.concatenate(([0], np.cumsum([len(item) for item in lists]))).astype(np.int32)

    def flat(lists) -> np.ndarray:
        return np.asarray([value for item in lists for value in item], dtype=np.int32)

    return {
        "dof_entry": dof_entry,
        "ancestor_mask": [sum(1 << ancestor for ancestor in path[:-1]) for path in paths],
        "path": flat(paths),
        "dof_joint": np.asarray([*dof_joint, 0], dtype=np.int32),
        "desc_start": starts(descendants),
        "desc_row": np.asarray([dof_entry[i] for item in descendants for i in item], dtype=np.int32),
        "desc_dof": flat(descendants),
        "height_start": starts(height_levels),
        "height_dof": flat(height_levels),
        "depth_start": starts(depth_levels),
        "depth_dof": flat(depth_levels),
        "level_pair_start": starts(level_pairs),
        "level_pair": flat(level_pairs),
        "subtree_start": starts(subtrees),
        "subtree_joint": flat(subtrees),
        "joint_last_dof": np.asarray([*joint_last_dof, -1], dtype=np.int32),
        "joint_parent": np.asarray([*joint_parent_local, -1], dtype=np.int32),
        "joint_depth": np.asarray([*joint_depth, 0], dtype=np.int32),
        "max_depth": max(joint_depth, default=0),
    }


@dataclasses.dataclass(frozen=True)
class DenseBucket:
    """Articulations that share one dense compile-time capacity."""

    max_dofs: int
    joint_capacity: int
    entry_capacity: int
    path_capacity: int
    subtree_capacity: int
    articulations: wp.array


class DenseTreeOperator:
    """Bucketed tree-sparse mass factors for the articulations of a reduced system.

    Args:
        model: Model whose articulations the reduced system owns.
        system: The :class:`ReducedArticulationSystem` providing per-pose
            motion subspaces, spatial inertias, and the implicit joint diagonal.
    """

    def __init__(self, model, system):
        self.model = model
        self.system = system
        self.device = model.device
        articulation_start = model.articulation_start.numpy()
        articulation_end = model.articulation_end.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        articulation_count = int(model.articulation_count)
        body_joint = np.full(int(model.body_count), -1, dtype=np.int32)
        for joint, child in enumerate(joint_child):
            body_joint[int(child)] = joint

        # Trees beyond the buckets fall back to the articulated-body recurrences.
        self.fallback_reasons: list[str] = []
        template_index: dict[tuple, int] = {}
        templates: list[dict] = []
        articulation_template = np.zeros(articulation_count, dtype=np.int32)
        factor_start = np.full(articulation_count, -1, dtype=np.int32)
        bucket_lists: dict[int, list[int]] = {}
        factor_size = 0
        for articulation in range(articulation_count):
            start = int(articulation_start[articulation])
            end = int(articulation_end[articulation])
            parents: list[int] = []
            for joint in range(start, end):
                parent = int(joint_parent[joint])
                parent_joint = int(body_joint[parent]) if parent >= 0 else -1
                # A root attached to a body outside the tree moves with no tree DOF.
                parents.append(parent_joint - start if start <= parent_joint < end else -1)
            # The path tables assume a parent precedes its children.
            if any(parent >= index for index, parent in enumerate(parents)):
                self.fallback_reasons.append(f"articulation {articulation} is not parent-first")
                continue
            dof_counts = [int(joint_qd_start[joint + 1] - joint_qd_start[joint]) for joint in range(start, end)]
            bucket = dense_dof_bucket(max(sum(dof_counts), end - start))
            if bucket == 0:
                self.fallback_reasons.append(
                    f"articulation {articulation} has more than {DENSE_DOF_BUCKETS[-1]} DOFs or joints"
                )
                continue
            key = (tuple(parents), tuple(dof_counts))
            if key not in template_index:
                template_index[key] = len(templates)
                templates.append(_articulation_template(parents, dof_counts))
            template = templates[template_index[key]]
            articulation_template[articulation] = template_index[key]
            bucket_lists.setdefault(bucket, []).append(articulation)
            factor_start[articulation] = factor_size
            factor_size += int(template["dof_entry"][-1])
        self.has_fallback = bool(self.fallback_reasons)
        if bucket_lists:
            # One launch for all trees, in the workspace of the largest. A launch ends with its
            # slowest block, so the smaller trees ride along instead of adding launches of their
            # own: a fleet of five robot types steps 1.1x to 1.6x faster than with one per size.
            bucket_lists = {max(bucket_lists): sorted(a for group in bucket_lists.values() for a in group)}

        def capacity(values) -> int:
            return _LANES * ((max(values) + _LANES - 1) // _LANES)

        # Each bucket's shared workspace holds its largest joint count, factor, root path, and subtree lists.
        self.buckets = [
            DenseBucket(
                max_dofs=bucket,
                joint_capacity=capacity(int(articulation_end[a] - articulation_start[a]) for a in articulations),
                entry_capacity=capacity(
                    int(templates[articulation_template[a]]["dof_entry"][-1]) for a in articulations
                ),
                path_capacity=max(
                    int(np.diff(templates[articulation_template[a]]["dof_entry"]).max(initial=0)) for a in articulations
                ),
                subtree_capacity=max(len(templates[articulation_template[a]]["subtree_joint"]) for a in articulations),
                articulations=wp.array(np.asarray(articulations, dtype=np.int32), device=self.device),
            )
            for bucket, articulations in sorted(bucket_lists.items())
        ]

        # Shared table images, packed with the layout of each template's bucket.
        images: list[np.ndarray] = [np.zeros(0, dtype=np.uint32)] * len(templates)
        for bucket in self.buckets:
            layout = DenseSharedLayout(
                bucket.max_dofs, bucket.joint_capacity, bucket.entry_capacity, bucket.subtree_capacity
            )
            for template in set(articulation_template[bucket.articulations.numpy()].tolist()):
                images[template] = layout.pack_template(templates[template])
        self.template_image_start = wp.array(
            np.concatenate(([0], np.cumsum([image.size for image in images])))[:-1].astype(np.int32)
            if images
            else np.zeros(1, dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        self.template_image = wp.array(
            np.concatenate([*images, np.zeros(1, dtype=np.uint32)]), dtype=wp.uint32, device=self.device
        )

        def offsets(name: str) -> np.ndarray:
            return np.concatenate(([0], np.cumsum([len(template[name]) for template in templates]))).astype(np.int32)

        def concatenated(name: str, rebase: str | None = None) -> wp.array:
            # Start tables of nested lists are rebased onto their template's slice of ``rebase``.
            base = offsets(rebase) if rebase is not None else np.zeros(len(templates) + 1, dtype=np.int32)
            values = [template[name] + base[index] for index, template in enumerate(templates)]
            return wp.array(
                np.concatenate(values or [np.zeros(1, dtype=np.int32)]).astype(np.int32),
                dtype=wp.int32,
                device=self.device,
            )

        def level_offsets(name: str) -> wp.array:
            # A template's levels run from its first start slot to the slot before the next
            # template's: each start table holds one more slot than levels.
            return wp.array(offsets(name), dtype=wp.int32, device=self.device)

        data = DenseTreeData()
        data.factor = wp.zeros(max(1, factor_size), dtype=wp.float32, device=self.device)
        data.factor_start = wp.array(factor_start, dtype=wp.int32, device=self.device)
        data.articulation_template = wp.array(articulation_template, dtype=wp.int32, device=self.device)
        data.template_dof_start = wp.array(offsets("dof_entry"), dtype=wp.int32, device=self.device)
        data.template_path_start = wp.array(offsets("path"), dtype=wp.int32, device=self.device)
        data.dof_entry = concatenated("dof_entry")
        data.path = concatenated("path")
        data.desc_start = concatenated("desc_start", "desc_row")
        data.desc_row = concatenated("desc_row")
        data.desc_dof = concatenated("desc_dof")
        data.template_height_start = level_offsets("height_start")
        data.height_start = concatenated("height_start", "height_dof")
        data.height_dof = concatenated("height_dof")
        data.template_depth_start = level_offsets("depth_start")
        data.depth_start = concatenated("depth_start", "depth_dof")
        data.depth_dof = concatenated("depth_dof")
        self.data = data
        self.template_joint_start = wp.array(offsets("subtree_start"), dtype=wp.int32, device=self.device)
        self.subtree_start = concatenated("subtree_start", "subtree_joint")
        self.subtree_joint = concatenated("subtree_joint")
        self.dof_joint = concatenated("dof_joint")
        self.joint_last_dof = concatenated("joint_last_dof")
        self.joint_parent_local = concatenated("joint_parent")
        self.joint_depth = concatenated("joint_depth")
        self.template_max_depth = wp.array(
            np.asarray([template["max_depth"] for template in templates] or [0], dtype=np.int32), device=self.device
        )
        self.level_pair_start = concatenated("level_pair_start", "level_pair")
        self.level_pair = concatenated("level_pair")
        # Flat float views of the per-pose motion subspaces and packed spatial inertias.
        self.motion = wp.array(
            ptr=system.joint_s.ptr, dtype=wp.float32, shape=(6 * system.joint_s.shape[0],), device=self.device
        )
        self.body_inertia = wp.array(
            ptr=system.body_i_s.ptr, dtype=wp.float32, shape=(21 * system.body_i_s.shape[0],), device=self.device
        )
        fallback_mask = np.zeros(articulation_count, dtype=np.int32)
        fallback_mask[factor_start < 0] = 1
        self.fallback_mask = wp.array(fallback_mask, dtype=wp.int32, device=self.device)
        self._fallback_enabled = wp.zeros(max(1, articulation_count), dtype=wp.int32, device=self.device)
        # Set by the owner when ABA-only consumers (coupled contacts, loops) also need the
        # articulated-body factor.
        self.requires_aba = False
        self._solution: wp.array[wp.float32] | None = None

    def factor_mass(self) -> None:
        """Assemble and factor every articulation's generalized mass at the current pose."""
        system = self.system
        model = self.model
        for bucket in self.buckets:
            wp.launch(
                _make_dense_factor_kernel(
                    bucket.max_dofs, bucket.joint_capacity, bucket.entry_capacity, bucket.subtree_capacity
                ),
                dim=bucket.articulations.shape[0] * _LANES,
                block_dim=_LANES,
                inputs=[
                    bucket.articulations,
                    model.articulation_start,
                    model.articulation_end,
                    model.joint_child,
                    model.joint_qd_start,
                    self.motion,
                    self.body_inertia,
                    system.joint_factor_diagonal,
                    self.template_image,
                    self.template_image_start,
                    self.data,
                ],
                device=self.device,
            )

    def form_link_responses(self, cross_tree: wp.array[wp.int32], bodies: BodyContainer) -> None:
        """Factor the trees that ``cross_tree`` flags or whose links hold deferred wrenches at
        the current pose, form their links' responses, and seed their deferred rates.

        See :func:`_make_dense_link_response_kernel`.
        """
        for bucket in self.buckets:
            wp.launch(
                _make_dense_link_response_kernel(
                    bucket.max_dofs,
                    bucket.joint_capacity,
                    bucket.entry_capacity,
                    bucket.path_capacity,
                    bucket.subtree_capacity,
                ),
                dim=bucket.articulations.shape[0] * _LANES,
                block_dim=_LANES,
                inputs=[
                    bucket.articulations,
                    self.motion,
                    self.body_inertia,
                    self.system.joint_factor_diagonal,
                    self.template_image,
                    self.template_image_start,
                    cross_tree,
                    bodies,
                ],
                device=self.device,
            )

    def solve(self, rhs: wp.array[wp.float32]) -> wp.array[wp.float32]:
        """Return ``M^-1 rhs`` for the last factored mass; ``rhs`` is overwritten."""
        if self._solution is None:
            self._solution = wp.zeros_like(rhs)
        wp.launch(
            _dense_solve_kernel,
            dim=int(self.model.articulation_count) * _LANES,
            block_dim=_LANES,
            inputs=[self.model.articulation_start, self.model.joint_qd_start, self.data, rhs],
            outputs=[self._solution],
            device=self.device,
        )
        return self._solution

    def fallback_enabled(self, enabled: wp.array[wp.int32]) -> wp.array[wp.int32]:
        """Return ``enabled`` restricted to the trees on the articulated-body path."""
        wp.launch(
            _mask_kernel,
            dim=self.fallback_mask.shape[0],
            inputs=[enabled, self.fallback_mask],
            outputs=[self._fallback_enabled],
            device=self.device,
        )
        return self._fallback_enabled

    def build_contact_rows(self, contact, bodies: BodyContainer, cc: ContactContainer, prepare: bool) -> None:
        """Build the packed rows and responses of a :class:`ReducedContactBlockSystem` page."""
        for bucket in self.buckets:
            wp.launch(
                _make_dense_contact_rows_kernel(bucket.max_dofs, bucket.entry_capacity, contact.patch_rows),
                dim=bucket.articulations.shape[0] * _LANES,
                block_dim=_LANES,
                inputs=[
                    bucket.articulations,
                    bodies,
                    contact.enabled,
                    contact.point_count,
                    contact.row_count,
                    contact.row_body,
                    contact.row_wrench,
                    contact.row_body_pair,
                    contact.row_wrench_pair,
                    contact.point_contact,
                    cc,
                    contact.max_page_count,
                    contact.page_index,
                    wp.bool(prepare),
                    self.joint_last_dof,
                    self.template_joint_start,
                    self.motion,
                ],
                outputs=[contact.packed_jacobian, contact.packed_response],
                device=self.device,
            )


__all__ = ["DENSE_DOF_BUCKETS", "DenseTreeOperator", "dense_dof_bucket", "dense_tree_solve"]
