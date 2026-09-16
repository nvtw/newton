"""Causal control only: current preparation/history with the old projection equations."""

import ast
import inspect

from local_studies.colibri.freeze_high_mass_operator import legacy_factory
from newton._src.solvers.phoenx.benchmarks.bench_high_mass_ratio import main
from newton._src.solvers.phoenx.constraints import constraint_contact_cloth as contact

factory = legacy_factory()
for node in ast.parse(inspect.getsource(contact)).body:
    if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
        continue
    if not isinstance(node.value.func, ast.Name) or node.value.func.id != "_make_contact_iterate_at":
        continue
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.value.keywords}
    if kwargs.get("cloth_support"):
        continue
    name = node.targets[0].id
    setattr(contact, name, factory(**kwargs))
main()
