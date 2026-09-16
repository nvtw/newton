"""Replay identical captured G1 packed-contact inputs against pre-cache kernels."""

import ast
import importlib.util
from pathlib import Path

import numpy as np
import warp as wp

from newton.examples.robot.example_robot_policy import Example
from newton.viewer import ViewerNull

source = Path("/tmp/reduced_contact_block_before_mobility_cache.py").read_text()
node = next(
    n
    for n in ast.parse(source).body
    if isinstance(n, ast.FunctionDef) and n.name == "_make_solve_generalized_contact_tile_ops"
)
factory = (
    ast.get_source_segment(source, node)
    .replace("_make_solve_generalized_contact_tile_ops", "_make_reference_ops")
    .replace("reduced_contact_generalized_solve_", "reduced_contact_generalized_reference_")
)
path = Path("/tmp/g1_mobility_reference_factory.py")
path.write_text(
    'from newton._src.solvers.phoenx.articulations import reduced_contact_block as _production\nfor _key, _value in vars(_production).items():\n    if not _key.startswith("__"):\n        globals()[_key] = _value\n\n'
    + factory
)
spec = importlib.util.spec_from_file_location("g1_mobility_reference_factory", path)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def clone(value, arrays):
    if isinstance(value, wp.array):
        key = (value.ptr, value.shape, value.strides, str(value.dtype))
        if key not in arrays:
            arrays[key] = wp.clone(value)
        return arrays[key]
    if hasattr(value, "_cls"):
        result = value._cls()
        for name in value._cls.vars:
            setattr(result, name, clone(getattr(value, name), arrays))
        return result
    return value


wp.init()
example = Example(ViewerNull(), Example.create_parser().parse_args(["--robot", "g1_29dof", "--solver", "phoenx"]))
for _ in range(20):
    example.step()
block = example.solver._reduced_articulation.contact_block_system
candidate = block.solve_kernel
baseline = reference._make_reference_ops(block.contact_dof_width)[1]
original_launch = wp.launch_tiled
captured = []


def launch(*args, **kwargs):
    kernel = kwargs.get("kernel", args[0] if args else None)
    if kernel is candidate and not captured:
        arrays = {}
        values = [clone(x, arrays) for x in kwargs["inputs"] + kwargs.get("outputs", [])]
        captured.append((values, kwargs["dim"], kwargs.get("block_dim", 32)))
    return original_launch(*args, **kwargs)


example.graph = None
wp.launch_tiled = launch
try:
    example.step()
finally:
    wp.launch_tiled = original_launch
assert captured, "No packed generalized-contact kernel captured"
values, dim, block_dim = captured[0]
print("CAPTURED", block.contact_dof_width, "point_counts", values[8].numpy())
for iterations in (1, 2, 8):
    for cap in (0, 1, 3, 32):
        arrays_a = {}
        arrays_b = {}
        a = [clone(x, arrays_a) for x in values]
        b = [clone(x, arrays_b) for x in values]
        a[4] = b[4] = iterations
        counts = np.minimum(a[8].numpy(), cap)
        a[8].assign(counts)
        b[8].assign(counts)
        original_launch(baseline, dim=dim, inputs=a, block_dim=block_dim, device=example.model.device)
        original_launch(candidate, dim=dim, inputs=b, block_dim=block_dim, device=example.model.device)
        for key, array in arrays_a.items():
            x = array.numpy()
            y = arrays_b[key].numpy()
            np.testing.assert_array_equal(
                x.view(np.uint8), y.view(np.uint8), err_msg=f"iterations={iterations} cap={cap} array={key}"
            )
# Exercise the cache's final lane using a degenerate full page: repeated point
# geometry/rows and a shared impulse slot, retaining the same serial row order.
page = int(values[16].numpy()[0])
packed_articulation = page
counts = values[8].numpy()
counts[packed_articulation] = 32
values[8].assign(counts)
for index in (9, 10, 11, 12, 13, 14):
    data = values[index].numpy()
    data[packed_articulation, :] = data[packed_articulation, 0]
    values[index].assign(data)
data = values[15].numpy()
for point in range(32):
    data[packed_articulation, 3 * point : 3 * point + 3] = data[packed_articulation, :3]
values[15].assign(data)
for index in (17, 18):
    data = values[index].numpy()
    rows = data[page * 96 : page * 96 + 3].copy()
    for point in range(32):
        data[page * 96 + 3 * point : page * 96 + 3 * point + 3] = rows
    values[index].assign(data)
arrays_a = {}
arrays_b = {}
a = [clone(x, arrays_a) for x in values]
b = [clone(x, arrays_b) for x in values]
a[4] = b[4] = 8
original_launch(baseline, dim=dim, inputs=a, block_dim=block_dim, device=example.model.device)
original_launch(candidate, dim=dim, inputs=b, block_dim=block_dim, device=example.model.device)
for key, array in arrays_a.items():
    np.testing.assert_array_equal(array.numpy().view(np.uint8), arrays_b[key].numpy().view(np.uint8))
print("FROZEN_CACHE_BITWISE_PASS 13 cases including full32point page")
