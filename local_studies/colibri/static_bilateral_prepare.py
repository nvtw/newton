# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Local fixed-bound factorization prototype; preserve active-row arithmetic."""

import ast
import copy
import inspect
import itertools
import json
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri.tail_first_dispatch import _compile_function
from newton._src.solvers.phoenx.constraints import bilateral_joint as module
from newton._src.solvers.phoenx.tests.test_block_joint_policy import make_model, make_solver


class Substitute(ast.NodeTransformer):
    def __init__(self, name, value):
        self.name, self.value = name, value

    def visit_Name(self, node):
        if node.id == self.name and isinstance(node.ctx, ast.Load):
            return ast.copy_location(ast.Constant(self.value), node)
        return node


class FixedLoops(ast.NodeTransformer):
    def visit_If(self, node):
        node = self.generic_visit(node)
        if not node.body and not node.orelse:
            return []
        return node

    def visit_For(self, node):
        if (
            not isinstance(node.iter, ast.Call)
            or not isinstance(node.iter.func, ast.Name)
            or node.iter.func.id != "range"
        ):
            return self.generic_visit(node)
        guarded = any(isinstance(arg, ast.Name) and arg.id == "count" for arg in node.iter.args)
        bounds = Substitute("count", 6).visit(copy.deepcopy(node.iter))
        values = eval(
            compile(ast.fix_missing_locations(ast.Expression(bounds)), "<loop-bounds>", "eval"), {"range": range}
        )
        result = []
        for value in values:
            body = [Substitute(node.target.id, value).visit(copy.deepcopy(statement)) for statement in node.body]
            if guarded:
                body = [
                    ast.If(
                        test=ast.Compare(
                            left=ast.Constant(value), ops=[ast.Lt()], comparators=[ast.Name(id="count", ctx=ast.Load())]
                        ),
                        body=body,
                        orelse=[],
                    )
                ]
            for statement in body:
                converted = self.visit(statement)
                result.extend(converted if isinstance(converted, list) else [converted])
        return result


def candidate():
    tree = FixedLoops().visit(ast.parse(inspect.getsource(module.prepare_bilateral_joint_blocks.func)))
    ast.fix_missing_locations(tree)
    return _compile_function(ast.unparse(tree), module, "prepare_bilateral_joint_blocks")


def main():
    kernel = candidate()
    model = make_model(40.0)
    model.joint_effort_limit.assign(np.asarray([1000.0], dtype=np.float32))
    solver = make_solver(model, mass_splitting=True, max_colored_partitions=0)
    state = model.state()
    solver.step(state, state, model.control(), None, 0.01)
    world = solver.world
    data = world.constraints.bilateral
    rng = np.random.default_rng(7819)
    fields = ("response0", "response1", "lower", "diagonal", "valid")
    cases = []
    original_wrenches = {field: getattr(data, field).numpy().copy() for field in ("wrench0", "wrench1")}
    original_mass = world.bodies.inverse_mass.numpy().copy()
    original_inertia = world.bodies.inverse_inertia_world.numpy().copy()
    for geometry, ratio, split in itertools.product(("authored", "random", "scaled"), (1.0, 400.0), (False, True)):
        mass, inertia = original_mass.copy(), original_inertia.copy()
        mass[-1] /= ratio
        inertia[-1] /= ratio
        world.bodies.inverse_mass.assign(mass)
        world.bodies.inverse_inertia_world.assign(inertia)
        world._copy_state.highest_index_in_use.assign(np.asarray([int(split)], dtype=np.int32))
        counts = np.arange(len(world._copy_state.count_per_node), dtype=np.int32) * 9 + 1
        world._copy_state.count_per_node.assign(counts)
        for field, original in original_wrenches.items():
            value = original.copy()
            if geometry != "authored":
                value = rng.normal(size=value.shape).astype(np.float32)
                if geometry == "scaled":
                    value *= np.logspace(-2, 2, value.shape[1], dtype=np.float32)[None, :, None]
            getattr(data, field).assign(value)
        for count in range(7):
            data.row_count.assign(np.full(data.row_count.shape, count, dtype=np.int32))
            outputs = []
            for active in (module.prepare_bilateral_joint_blocks, kernel):
                for field in fields:
                    getattr(data, field).zero_()
                wp.launch(
                    active, world.num_joints, [world.constraints, world.bodies, world._copy_state], device=model.device
                )
                outputs.append({field: getattr(data, field).numpy() for field in fields})
            case = {"count": count, "geometry": geometry, "mass_ratio": ratio, "split": split}
            for field in fields:
                if not np.all(np.isfinite(outputs[0][field])):
                    raise AssertionError(f"Nonfinite reference: {case}, {field}")
                np.testing.assert_array_equal(outputs[0][field], outputs[1][field], err_msg=f"{case}, field={field}")
            cases.append({**case, "exact": True, "valid_blocks": int(np.sum(outputs[0]["valid"]))})
    report = {"cases": cases, "passed": True}
    Path("/tmp/colibri_static_bilateral_prepare.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
