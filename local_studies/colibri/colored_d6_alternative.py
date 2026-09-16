# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Corrected friction experiment on full-coordinate colored physical joint blocks.

Local runner only. Reuse exact reviewed staged friction/history import order;
replace only native direct joint/contact ownership with ordinary block PGS.
No mass splitting, changed drive coefficients, damping, or SOR change.
"""

import hashlib
from pathlib import Path


def transformed_source():
    """Assert source anchors before constructing a reviewable runner variant."""
    path = Path(__file__).with_name("native_conditioned_two_body.py")
    source = path.read_text()
    changes = {
        '"path": args.native_path,': '"path": "colored_block_pgs",',
        'joint_solver="direct",': 'joint_solver="block_pgs",',
        "        original_solver(solver, model, *positional, **options)": """        options["solver_iterations"] = int(os.environ.get("COLIBRI_COLORED_SWEEPS", "1"))
        assert options["solver_iterations"] >= 1
        original_solver(solver, model, *positional, **options)""",
        '''        if args.native_path == "direct":
            assert report["direct_tree_contacts"] or report["direct_contact_response"], (
                "No joint-conditioned contact owner"
            )
        else:
            assert any(report["reduced_owned"]), "No reduced joint ownership"''': """        assert args.native_path == "direct", "This control is maximal/full-coordinate only"
        assert not report["direct_tree_contacts"] and not report["direct_contact_response"]
        assert solver._reduced_articulation is None
        assert type(solver._direct_equality_system).__name__ == "BlockJointSystem"
        assert solver.joint_solver == "block_pgs" and not solver.world.mass_splitting_enabled
        report["alternative_scope"] = "Full-coordinate ordinary-color joint/contact PGS; identical physical implicit drive, point law and source settings"
        import inspect
        from newton._src.solvers.phoenx.constraints import bilateral_joint, contact_projection
        projection_source = inspect.getsource(contact_projection._friction_normal_lambda.func)
        assert "return wp.max(lambda_n, wp.float32(0.0))" in projection_source
        report["ordinary_joint_module"] = {"path": bilateral_joint.__file__, "sha256": digest(Path(bilateral_joint.__file__))}
        report["ordinary_contact_module"] = {"path": contact_projection.__file__, "sha256": digest(Path(contact_projection.__file__))}
        if os.environ.get("COLIBRI_COLORED_PHASES") == "1":
            from local_studies.colibri.capture_colored_d6_phases import install
            phase_audits.append(install(solver, output))
        print("COLORED_BLOCK_PGS_OWNERSHIP_GATE_PASS", flush=True)""",
    }
    for old, new in changes.items():
        assert source.count(old) == 1, (old, source.count(old))
        source = source.replace(old, new)
    compile(source, str(path), "exec")
    return path, source


def main():
    path, source = transformed_source()
    print("COLORED_D6_RUNNER_SHA256", hashlib.sha256(source.encode()).hexdigest(), flush=True)
    exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})


if __name__ == "__main__":
    main()
