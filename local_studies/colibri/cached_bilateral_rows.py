# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Explicit local row cache; no canonical fields are repurposed."""

import ast
import inspect
import types

import warp as wp

from local_studies.colibri.tail_first_dispatch import _compile_function
from newton._src.solvers.phoenx.constraints import bilateral_joint


@wp.struct
class CachedBilateralRows:
    wrench0: wp.array2d[wp.spatial_vector]
    wrench1: wp.array2d[wp.spatial_vector]
    bias: wp.array2d[wp.float32]


def allocate(count, device):
    """Allocate one exact stored wrench pair and bias per block row."""
    cache = CachedBilateralRows()
    cache.wrench0 = wp.zeros((count, 6), dtype=wp.spatial_vector, device=device)
    cache.wrench1 = wp.zeros((count, 6), dtype=wp.spatial_vector, device=device)
    cache.bias = wp.zeros((count, 6), dtype=wp.float32, device=device)
    return cache


def factories():
    """Build fused cache preparation and a row-index-independent callback."""
    namespace = types.SimpleNamespace(**vars(bilateral_joint))
    namespace.CachedBilateralRows = CachedBilateralRows
    prepare = inspect.getsource(bilateral_joint.prepare_bilateral_joint_blocks.func)
    prepare = prepare.replace("def prepare_bilateral_joint_blocks(", "def prepare_cached_bilateral_rows(")
    prepare = prepare.replace(
        "constraints: ConstraintContainer, bodies: BodyContainer, copy_state: CopyStateContainer",
        "constraints: ConstraintContainer, bodies: BodyContainer, copy_state: CopyStateContainer, cache: CachedBilateralRows",
    )
    marker = "            data.response0[cid, i] ="
    location = prepare.index(marker)
    prepare = (
        prepare[:location]
        + (
            "            cache.wrench0[cid, i] = data.wrench0[structural, local]\n"
            "            cache.wrench1[cid, i] = data.wrench1[structural, local]\n"
            "            cache.bias[cid, i] = data.bias[structural, local]\n"
        )
        + prepare[location:]
    )
    # Keep factor preparation arithmetic unchanged, including its original
    # wrench reads. The cache only changes loads during subsequent iteration.
    iterate = inspect.getsource(bilateral_joint.iterate_bilateral_joint_block.func)
    iterate = iterate.replace("def iterate_bilateral_joint_block(", "def iterate_cached_bilateral_rows(")
    iterate = iterate.replace("    use_bias: wp.bool,", "    use_bias: wp.bool,\n    cache: CachedBilateralRows,")
    iterate = iterate.replace("data.wrench0[structural, local]", "cache.wrench0[cid, i]")
    iterate = iterate.replace("data.wrench1[structural, local]", "cache.wrench1[cid, i]")
    iterate = iterate.replace("data.bias[structural, local]", "cache.bias[cid, i]")
    iterate = iterate.replace("    structural = data.structural_index[cid]\n", "")
    iterate = iterate.replace("        local = data.row_local[row]\n", "")
    ast.parse(prepare)
    ast.parse(iterate)
    return (
        _compile_function(prepare, namespace, "prepare_cached_bilateral_rows"),
        _compile_function(iterate, namespace, "iterate_cached_bilateral_rows"),
    )


def cached_sweep(phase, soft, callback):
    """Retain group ordering and inequality/contact dispatch with cached rows."""
    from newton._src.solvers.phoenx import solver_phoenx_kernels as kernels
    from newton._src.solvers.phoenx.dispatch import color_groups

    tree = ast.parse(inspect.getsource(color_groups.get_sweep_kernel))

    class Rewrite(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == "sweep":
                node.args.args.append(
                    ast.arg(arg="cache", annotation=ast.Name(id="CachedBilateralRows", ctx=ast.Load()))
                )
            return self.generic_visit(node)

        def visit_Expr(self, node):
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "dispatch"
            ):
                replacement = ast.parse(
                    "if cid < num_joints:\n"
                    "    if enabled[cid] == 1:\n"
                    "        cached_iterate(constraints, cid, bodies, particles, copies, num_bodies, slab, phase == 'iterate', cache)\n"
                    "        joint_inequality(constraints, cid, bodies, particles, copies, num_bodies, slab, idt, 1.0, phase == 'iterate')\n"
                    "else:\n"
                    "    pass"
                ).body[0]
                replacement.orelse = [node]
                return replacement
            return self.generic_visit(node)

    tree = Rewrite().visit(tree)
    ast.fix_missing_locations(tree)
    namespace = types.SimpleNamespace(**vars(color_groups))
    namespace.CachedBilateralRows = CachedBilateralRows
    namespace.cached_iterate = callback
    namespace.joint_inequality = kernels.joint_constraint_iterate_inequality
    source = ast.unparse(tree).replace("phase == 'iterate'", str(phase == "iterate"))
    return _compile_function(source, namespace, "get_sweep_kernel")(phase, soft)


def install():
    """Install a grouped-rigid-only local launch adapter; return a restore hook."""
    from newton._src.solvers.phoenx.articulations.block_joint_system import BlockJointSystem
    from newton._src.solvers.phoenx.dispatch import color_groups

    prepared, callback = factories()
    original_bind = BlockJointSystem.bind_world
    original_launch = wp.launch
    caches = {}
    stats = {"bind": 0, "prepare": 0, "sweep": 0}
    replacements = {}
    for phase in ("iterate", "relax"):
        for soft in (False, True):
            replacements[color_groups.get_sweep_kernel(phase, soft)] = cached_sweep(phase, soft, callback)

    def bind(system, world, mapping):
        original_bind(system, world, mapping)
        stats["bind"] += 1
        cache = allocate(world.num_joints, world.device)
        caches[int(world.constraints.bilateral.row_count.ptr)] = cache

    def launch(kernel, *positional, **keywords):
        replacement = replacements.get(kernel)
        if replacement is None and kernel.key == "get_sweep_kernel__locals__sweep":
            for phase in ("iterate", "relax"):
                for soft in (False, True):
                    if kernel is color_groups.get_sweep_kernel(phase, soft):
                        replacement = cached_sweep(phase, soft, callback)
                        replacements[kernel] = replacement
        is_prepare = kernel is bilateral_joint.prepare_bilateral_joint_blocks
        if replacement is None and not is_prepare:
            return original_launch(kernel, *positional, **keywords)
        positional = list(positional)
        keywords = dict(keywords)
        inputs = list(keywords["inputs"] if "inputs" in keywords else positional[1])
        if inputs[0].bilateral.row_count is None:
            return original_launch(kernel, *positional, **keywords)
        key = int(inputs[0].bilateral.row_count.ptr)
        if key not in caches:
            return original_launch(kernel, *positional, **keywords)
        inputs.append(caches[key])
        stats["prepare" if is_prepare else "sweep"] += 1
        if "inputs" in keywords:
            keywords["inputs"] = inputs
        else:
            positional[1] = inputs
        return original_launch(prepared if is_prepare else replacement, *positional, **keywords)

    BlockJointSystem.bind_world = bind
    wp.launch = launch

    def restore():
        BlockJointSystem.bind_world = original_bind
        wp.launch = original_launch

    restore.stats = stats
    return restore


if __name__ == "__main__":
    import runpy
    import sys

    install()
    sys.argv = sys.argv[1:]
    runpy.run_module(sys.argv[0], run_name="__main__")
