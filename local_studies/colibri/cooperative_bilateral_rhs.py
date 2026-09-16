# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Local eight-lane RHS, scalar exact solve/scatter, unchanged color ordering."""

import ast
import inspect
import types

import warp as wp

from local_studies.colibri.tail_first_dispatch import _compile_function
from newton._src.solvers.phoenx.constraints import bilateral_joint


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    unsigned int mask = 0xffu << (threadIdx.x & 24);
    return __shfl_sync(mask, value, source_lane, 8);
#else
    return value;
#endif
""")
def shuffle_rhs_double(value: wp.float64, source_lane: wp.int32) -> wp.float64: ...


def callback_factory():
    """Parallelize independent RHS rows; retain each row's FP64 dot order."""
    source = inspect.getsource(bilateral_joint.iterate_bilateral_joint_block.func)
    source = source.replace("def iterate_bilateral_joint_block(", "def iterate_eight_lane_joint_rhs(")
    source = source.replace("    use_bias: wp.bool,", "    use_bias: wp.bool,\n    tile_lane: wp.int32,")
    source = source.replace(
        "    rhs = Vec6d()\n    for i in range(count):",
        "    rhs_value = wp.float64(0.0)\n    if tile_lane < count:\n        i = tile_lane",
        1,
    )
    source = source.replace("        rhs[i] = -residual", "        rhs_value = -residual", 1)
    source = source.replace(
        "    lower = data.lower[cid]",
        (
            "    rhs = Vec6d()\n"
            "    for row_lane in range(count):\n"
            "        rhs[row_lane] = shuffle_rhs_double(rhs_value, row_lane)\n"
            "    if tile_lane != 0:\n"
            "        return\n"
            "    lower = data.lower[cid]"
        ),
        1,
    )
    namespace = types.SimpleNamespace(**vars(bilateral_joint))
    namespace.shuffle_rhs_double = shuffle_rhs_double
    return _compile_function(source, namespace, "iterate_eight_lane_joint_rhs")


def sweep_factory(phase, soft, callback):
    """Assign eight lanes per existing constraint slot, leader-only contacts."""
    from newton._src.solvers.phoenx import solver_phoenx_kernels as kernels
    from newton._src.solvers.phoenx.dispatch import color_groups

    tree = ast.parse(inspect.getsource(color_groups.get_sweep_kernel))

    class Rewrite(ast.NodeTransformer):
        def visit_Expr(self, node):
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "dispatch"
            ):
                replacement = ast.parse(
                    "if cid < num_joints:\n"
                    "    if enabled[cid] == 1:\n"
                    "        cooperative_rhs(constraints, cid, bodies, particles, copies, num_bodies, slab, USE_BIAS, lane % 8)\n"
                    "        if lane % 8 == 0:\n"
                    "            joint_inequality(constraints, cid, bodies, particles, copies, num_bodies, slab, idt, 1.0, USE_BIAS)\n"
                    "elif lane % 8 == 0:\n"
                    "    pass".replace("USE_BIAS", str(phase == "iterate"))
                ).body[0]
                replacement.orelse[0].body = [node]
                return replacement
            return self.generic_visit(node)

    tree = Rewrite().visit(tree)
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree).replace("starts[color] + lane", "starts[color] + lane / 8")
    namespace = types.SimpleNamespace(**vars(color_groups))
    namespace.cooperative_rhs = callback
    namespace.joint_inequality = kernels.joint_constraint_iterate_inequality
    return _compile_function(source, namespace, "get_sweep_kernel")(phase, soft)


def install():
    """Replace grouped iterate/relax launches locally, preserving preparation."""
    from newton._src.solvers.phoenx.dispatch import color_groups

    callback = callback_factory()
    original = wp.launch
    replacements = {}

    def launch(kernel, *positional, **keywords):
        replacement = replacements.get(kernel)
        if replacement is None and kernel.key == "get_sweep_kernel__locals__sweep":
            for phase in ("iterate", "relax"):
                for soft in (False, True):
                    if kernel is color_groups.get_sweep_kernel(phase, soft):
                        replacement = sweep_factory(phase, soft, callback)
                        replacements[kernel] = replacement
        if replacement is None:
            return original(kernel, *positional, **keywords)
        positional = list(positional)
        keywords = dict(keywords)
        if "dim" in keywords:
            keywords["dim"] = (keywords["dim"][0], 256)
        else:
            positional[0] = (positional[0][0], 256)
        keywords["block_dim"] = 256
        return original(replacement, *positional, **keywords)

    wp.launch = launch

    def restore():
        wp.launch = original

    return restore


if __name__ == "__main__":
    import runpy
    import sys

    install()
    sys.argv = sys.argv[1:]
    runpy.run_module(sys.argv[0], run_name="__main__")
