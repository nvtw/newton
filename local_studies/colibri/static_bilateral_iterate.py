# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Local exact constant-index bilateral solve; production is unchanged."""

import ast
import inspect
import runpy
import sys

from local_studies.colibri.static_bilateral_prepare import FixedLoops
from local_studies.colibri.tail_first_dispatch import _compile_function
from newton._src.solvers.phoenx.constraints import bilateral_joint as module


class ReverseRows(ast.NodeTransformer):
    """Express descending active rows with constant upper bound before expansion."""

    def visit_For(self, node):
        if isinstance(node.target, ast.Name) and node.target.id == "reverse":
            node.target.id = "i"
            node.iter = ast.parse("range(5, -1, -1)", mode="eval").body
            node.body = [
                ast.If(
                    test=ast.parse("i < count", mode="eval").body,
                    body=node.body[1:],
                    orelse=[],
                )
            ]
        return self.generic_visit(node)


def source():
    """Expand guarded loops while preserving every active arithmetic operation."""
    tree = ast.parse(inspect.getsource(module.iterate_bilateral_joint_block.func))
    tree = ReverseRows().visit(tree)
    tree = FixedLoops().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def candidate():
    """Compile the local expanded callback."""
    return _compile_function(source(), module, "iterate_bilateral_joint_block")


def install():
    """Replace the callback only in this diagnostic process."""
    from newton._src.solvers.phoenx import solver_phoenx_kernels

    solver_phoenx_kernels.iterate_bilateral_joint_block = candidate()


if __name__ == "__main__":
    install()
    sys.argv = sys.argv[1:]
    runpy.run_module(sys.argv[0], run_name="__main__")
