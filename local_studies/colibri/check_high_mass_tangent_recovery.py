"""Diagnostic only: retain physical friction but remove tangent position recovery."""

import ast
import inspect
import types
from pathlib import Path

from newton._src.solvers.phoenx.benchmarks.bench_high_mass_ratio import main
from newton._src.solvers.phoenx.constraints import constraint_contact_cloth as contact

source = inspect.getsource(contact._make_contact_prepare_for_iteration_at)
old = "friction_bias_factor = wp.float32(0.08)"
assert source.count(old) == 1
source = source.replace(old, "friction_bias_factor = wp.float32(0.0)")
path = Path("/tmp/high_mass_no_tangent_recovery_factory.py")
path.write_text(source)
module = types.ModuleType("high_mass_no_tangent_recovery_factory")
module.__dict__.update(vars(contact))
module.__dict__["__name__"] = "high_mass_no_tangent_recovery_factory"
exec(compile(source, str(path), "exec"), module.__dict__)
factory = module._make_contact_prepare_for_iteration_at
for node in ast.parse(inspect.getsource(contact)).body:
    if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
        continue
    if not isinstance(node.value.func, ast.Name) or node.value.func.id != "_make_contact_prepare_for_iteration_at":
        continue
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.value.keywords}
    if kwargs.get("cloth_support"):
        continue
    setattr(contact, node.targets[0].id, factory(**kwargs))
main()
