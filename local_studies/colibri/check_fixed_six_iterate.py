# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Process-local six-point contact sweep specialization."""

import ast
import inspect
import linecache
import runpy
import sys

from newton._src.solvers.phoenx import solver_phoenx_kernels as kernels
from newton._src.solvers.phoenx.constraints import constraint_contact_cloth as contact
from newton._src.solvers.phoenx.dispatch import color_groups


class BoundContactLoop(ast.NodeTransformer):
    """Bound the point loop while retaining sequential row arithmetic."""

    def visit_For(self, node):
        node = self.generic_visit(node)
        if ast.unparse(node.iter) == "range(contact_count)":
            node.iter = ast.parse("range(6)", mode="eval").body
            node.body = [ast.If(test=ast.parse("i < contact_count", mode="eval").body, body=node.body, orelse=[])]
        return node


def install():
    """Install only chunk-six ordinary rigid iteration callbacks."""
    tree = BoundContactLoop().visit(ast.parse(inspect.getsource(contact._make_contact_iterate_at)))
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree)
    filename = "/tmp/colibri_fixed_six_iterate_factory.py"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(vars(contact))
    exec(compile(source, filename, "exec"), namespace)
    factory = namespace["_make_contact_iterate_at"]
    saved = {}
    for soft in (False, True):
        suffix = "" if soft else "_no_soft_pd"
        for use_bias, prefix in ((True, "contact_iterate_at"), (False, "contact_relax_at")):
            name = prefix + suffix
            saved[name] = getattr(contact, name)
            setattr(contact, name, factory(cloth_support=False, use_bias=use_bias, has_soft_contact_pd=soft))
        replacement = contact._make_contact_iterate_entry(
            has_sleeping=False, has_soft_contact_pd=soft, staged_body_properties=False
        )
        setattr(kernels, "contact_iterate_no_sleep" + suffix, replacement)
    for name, value in saved.items():
        setattr(contact, name, value)
    kernels._make_singleworld_rigid_contact_dispatch_func.cache_clear()
    kernels._make_singleworld_dispatch_func.cache_clear()
    color_groups.get_sweep_kernel.cache_clear()


if __name__ == "__main__":
    if "--contact-chunk-size" not in sys.argv or sys.argv[sys.argv.index("--contact-chunk-size") + 1] != "6":
        raise ValueError("This bounded prototype requires --contact-chunk-size 6")
    install()
    runpy.run_module("local_studies.colibri.check_canonical_groups", run_name="__main__")
