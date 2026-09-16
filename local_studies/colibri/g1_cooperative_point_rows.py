"""Generate a local eight-lane point-row experiment; production is untouched.

The scalar ancestor/forward order is retained. Only independent six-DoF dot
products and matrix rows are distributed. Root projection stores have unique
lane/DoF ownership; other stores use lane zero. Subgroup barriers protect
predecessor response reads.
"""

import ast
import importlib.util
import re
from pathlib import Path


def make_candidate(directory=Path("/tmp/g1_cooperative_point_rows"), scalar_lane_values=False):
    """Materialize unique Warp source so kernel identity and inspection are explicit."""
    from newton._src.solvers.phoenx.articulations import reduced_contact_block as production

    source = Path(production.__file__).read_text()
    factory = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_make_build_packed_rows_ops"
    )
    text = ast.get_source_segment(source, factory)
    text = text.replace("_make_build_packed_rows_ops", "_make_cooperative_point_rows")
    text = text.replace('"reduced_contact_rows_packed"', '"local_g1_cooperative_point_rows"')
    text = text.replace("_build_packed_generalized_row", "_build_cooperative_generalized_row")
    text = text.replace("_build_packed_generalized_contact_rows_kernel", "_build_cooperative_contact_rows_kernel")
    tree = ast.parse(text)
    fn = tree.body[0]
    helper = next(
        n for n in fn.body if isinstance(n, ast.FunctionDef) and n.name == "_build_cooperative_generalized_row"
    )
    helper.args.args.extend(ast.parse("def f(lane: wp.int32, mask: wp.uint32): pass").body[0].args.args)

    class RootRows(ast.NodeTransformer):
        def visit_For(self, node):
            # All four root loops compute independent vector/matrix rows. The
            # vector values are gathered before ordered wrench accumulation.
            if isinstance(node.target, ast.Name) and node.target.id == "dof_row":
                body = node.body[0]
                assert isinstance(body, ast.If)
                stores = [n for n in ast.walk(body) if isinstance(n, ast.Assign)]
                names = {
                    n.targets[0].value.id
                    for n in stores
                    if isinstance(n.targets[0], ast.Subscript) and isinstance(n.targets[0].value, ast.Name)
                }
                vector = "projected" if "projected" in names else "rhs" if "rhs" in names else None
                if vector:
                    local = ast.parse(ast.unparse(body).replace("dof_row", "lane")).body[0]
                    for assignment in ast.walk(local):
                        if isinstance(assignment, ast.Assign):
                            assignment.cooperative_unique = True
                    gather = ast.parse(
                        f"for component in range(6):\n    {vector}[component] = _shuffle_reduced_float({vector}[lane % 6], component, 8, mask)"
                    ).body[0]
                    # Lanes6/7 own no component, but participate in every shuffle.
                    return [local, gather]
                # Matrix row loops include an ordered scatter after the inner
                # column loop. Separate only the independent matrix work.
                inner = body.body[0]
                assert isinstance(inner, ast.For)
                matrix = "reduced" if "reduced" in ast.unparse(inner) else "generalized_delta"
                partial = ast.parse(
                    "if lane < dof_count:\n"
                    + "\n".join("    " + line for line in ast.unparse(inner).replace("dof_row", "lane").splitlines())
                ).body[0]
                gather = ast.parse(
                    f"for component in range(6):\n    {matrix}[component] = _shuffle_reduced_float({matrix}[lane % 6], component, 8, mask)"
                ).body[0]
                body.body = body.body[1:]
                return [partial, gather, node]
            return self.generic_visit(node)

    helper = RootRows().visit(helper)

    class LeaderStores(ast.NodeTransformer):
        def visit_Assign(self, node):
            if getattr(node, "cooperative_unique", False):
                return node
            target = node.targets[0]
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id in {"joint_work", "packed_jacobian", "packed_response", "body_response"}
            ):
                return ast.If(test=ast.parse("lane == 0", mode="eval").body, body=[node], orelse=[])
            return node

    helper = LeaderStores().visit(helper)
    forward = next(
        n for n in helper.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "joint"
    )
    forward.body.insert(0, ast.parse("_sync_contact_group(mask)").body[0])
    # First forward iteration also publishes backward joint_work writes.
    kernel = next(
        n for n in fn.body if isinstance(n, ast.FunctionDef) and n.name == "_build_cooperative_contact_rows_kernel"
    )
    kernel.body[:1] = ast.parse(
        "articulation, local_thread = wp.tid()\nrow = local_thread // 8\nlane = local_thread % 8\nmask = wp.uint32(255) << wp.uint32((local_thread % 32) // 8 * 8)"
    ).body
    begin = next(
        i
        for i, n in enumerate(kernel.body)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "previous_body"
    )
    end = next(
        i
        for i, n in enumerate(kernel.body)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "inverse_mass"
    )
    kernel.body[begin:end] = [
        ast.If(test=ast.parse("lane == 0", mode="eval").body, body=kernel.body[begin:end], orelse=[]),
        ast.parse("_sync_contact_group(mask)").body[0],
    ]
    branch = next(
        n for n in kernel.body if isinstance(n, ast.If) and ast.unparse(n.test) == "current_body_pair >= wp.int32(0)"
    )
    branch.body = [ast.If(test=ast.parse("lane == 0", mode="eval").body, body=branch.body, orelse=[])]
    call = branch.orelse[0].value
    call.args.extend([ast.Name(id="lane", ctx=ast.Load()), ast.Name(id="mask", ctx=ast.Load())])
    index = kernel.body.index(branch)
    kernel.body.insert(index + 1, ast.parse("if lane != 0:\n    return").body[0])
    ast.fix_missing_locations(tree)
    directory.mkdir(parents=True, exist_ok=True)
    generated = ast.unparse(tree)
    if scalar_lane_values:
        generated = generated.replace("cooperative", "cooperative_scalar")
        generated = re.sub(
            r"(?m)^(\s*)(projected|reduced|rhs|generalized_delta) = _vec6\(0.0\)$",
            lambda match: match.group(0) + "\n" + match.group(1) + "lane_" + match.group(2) + " = wp.float32(0.0)",
            generated,
        )
        for vector in ("projected", "reduced", "rhs", "generalized_delta"):
            generated = generated.replace(vector + "[lane % 6]", "lane_" + vector)
            generated = generated.replace(vector + "[lane]", "lane_" + vector)
    path = directory / ("candidate_scalar.py" if scalar_lane_values else "candidate.py")
    path.write_text(
        "from newton._src.solvers.phoenx.articulations import reduced_contact_block as _p\nfrom newton._src.solvers.phoenx.articulations.reduced import _shuffle_reduced_float\nfor _key, _value in vars(_p).items():\n    if not _key.startswith('__'):\n        globals()[_key] = _value\n\n"
        + generated
        + "\n"
    )
    spec = importlib.util.spec_from_file_location("g1_cooperative_point_rows_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory_name = "_make_cooperative_scalar_point_rows" if scalar_lane_values else "_make_cooperative_point_rows"
    return getattr(module, factory_name)(False), path
