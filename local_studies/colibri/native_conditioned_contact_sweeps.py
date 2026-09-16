"""Local corrected-callback matched biased/unbiased sweep-count control."""

# ruff: noqa: PLC0415 -- overlays must be installed before Newton imports.
import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE = Path("/tmp/colibri_merged_friction_proposal")


class OwnedFinder(importlib.abc.MetaPathFinder):
    """Use the reviewed owned-callback law/history changes without canonical edits."""

    def find_spec(self, fullname, path=None, target=None):
        prefix = "newton._src.solvers.phoenx.articulations."
        if fullname.startswith(prefix) and fullname[len(prefix) :] in (
            "direct_contact_gs",
            "maximal_contact_gs",
            "reduced_contact_block",
        ):
            return importlib.util.spec_from_file_location(fullname, STAGE / (fullname.replace(".", "/") + ".py"))
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-path", choices=("direct", "reduced"), required=True)
    parser.add_argument("--contact-sweeps", type=int, choices=(1, 2, 4, 8), required=True)
    args, remaining = parser.parse_known_args()
    assert "--body-count" in remaining and remaining[remaining.index("--body-count") + 1] == "2"
    output = Path(remaining[remaining.index("--output") + 1])
    paths = sorted((ROOT / "newton").rglob("*.py"))
    paths.append(ROOT / "local_studies/colibri/native_conditioned_two_body.py")
    paths += sorted(p for p in (ROOT / "newton/examples/assets/colibri").rglob("*") if p.is_file())

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    before = {str(p): digest(p) for p in paths}
    manifest = json.loads((STAGE / "manifest.json").read_text())
    for relative, expected in manifest["staged"].items():
        assert digest(STAGE / relative) == expected, relative
    assert not any(
        "newton._src.solvers.phoenx.articulations." + name in sys.modules
        for name in ("direct_contact_gs", "maximal_contact_gs", "reduced_contact_block")
    ), "Contaminated owned-callback process"
    # Owned callbacks must be selected before any Newton package import.
    sys.meta_path.insert(0, OwnedFinder())
    from local_studies.colibri import combined_gap_matching, friction_break_state

    original_sources = friction_break_state.sources

    def sources():
        original, modified = original_sources()
        for name in modified:
            modified[name] = (STAGE / "newton/_src/solvers/phoenx/constraints" / (name + ".py")).read_text()
        old = "solver_gap > wp.float32(0.0) or drift_sq > slip_threshold * slip_threshold"
        assert modified["constraint_contact_cloth"].count(old) == 1
        modified["constraint_contact_cloth"] = modified["constraint_contact_cloth"].replace(
            old, "solver_gap > slip_threshold or drift_sq > slip_threshold * slip_threshold"
        )
        return original, modified

    friction_break_state.sources = sources
    differential = os.environ.get("COLIBRI_DIFFERENTIAL_REFERENCE") == "1"
    if differential:
        from local_studies.colibri import friction_differential_reference

        friction_differential_reference.install()
    # Apply the same gap criterion above while retaining the staged patch-only guard.
    combined_gap_matching.friction_gap_history.install = friction_break_state.install
    _original, modified, directory, pipeline_constructor, installed = combined_gap_matching.install()
    import numpy as np

    import newton
    from newton.solvers import SolverPhoenX

    owned_loaded = {}
    for short in ("direct_contact_gs", "maximal_contact_gs", "reduced_contact_block"):
        fullname = "newton._src.solvers.phoenx.articulations." + short
        module = sys.modules.get(fullname)
        if module is not None:
            expected = STAGE / (fullname.replace(".", "/") + ".py")
            assert Path(module.__file__).resolve() == expected.resolve(), (fullname, module.__file__)
            owned_loaded[short] = {"path": module.__file__, "sha256": digest(Path(module.__file__))}
    assert set(owned_loaded) == {"direct_contact_gs", "maximal_contact_gs", "reduced_contact_block"}
    solver_module = sys.modules["newton._src.solvers.phoenx.solver_phoenx"]
    maximal_module = sys.modules["newton._src.solvers.phoenx.articulations.maximal_contact_gs"]
    projection_module = sys.modules["newton._src.solvers.phoenx.constraints.contact_projection"]
    assert solver_module.iterate_maximal_contact_runs_kernel is maximal_module.iterate_maximal_contact_runs_kernel
    assert (
        maximal_module.contact_project_friction_metric_with_break
        is projection_module.contact_project_friction_metric_with_break
    )
    from local_studies.colibri.check_owned_friction_binding import check

    binding_gate = check(sys.modules["newton._src.solvers.phoenx.articulations.maximal_contact_gs"])

    original_solver = SolverPhoenX.__init__
    report = {
        "path": args.native_path,
        "arguments": remaining,
        "source_sha256": before,
        "actual_owned_modules": owned_loaded,
        "owned_binding_gate": binding_gate,
    }
    constructed = []
    phase_audits = []

    def construct(solver, model, *positional, **options):
        properties = (
            "body_mass",
            "body_inertia",
            "body_com",
            "joint_target_ke",
            "joint_target_kd",
            "joint_target_mode",
            "joint_damping",
            "joint_armature",
            "joint_target_q",
            "joint_target_qd",
        )
        saved = {
            name: getattr(model, name).numpy().copy() for name in properties if getattr(model, name, None) is not None
        }
        options.update(
            articulation_mode="maximal" if args.native_path == "direct" else "reduced",
            joint_solver="direct",
            mass_splitting_color_group_size=0,
            contact_chunk_size=0,
            parallel_contact_prepare=False,
            mass_splitting=False,
            velocity_iterations=args.contact_sweeps,
        )
        assert (
            options["substeps"] == 30
            and options["solver_iterations"] == args.contact_sweeps
            and options["sor_boost"] == 1.0
        )
        report["sweep_budget"] = {
            "biased_per_substep": args.contact_sweeps,
            "unbiased_final_substep": args.contact_sweeps,
            "biased_per_frame": 60 * args.contact_sweeps,
            "unbiased_per_frame": 2 * args.contact_sweeps,
            "scope": "Increased native iteration work; no matched-one-sweep-budget claim",
        }
        original_solver(solver, model, *positional, **options)
        constructed.append(solver)
        if differential:
            from newton._src.solvers.phoenx.constraints import contact_container

            assert solver.world._contact_container.lambdas.shape[0] == 27
            assert hasattr(contact_container, "cc_get_friction_reference_delta")
            report["differential_helper_module"] = contact_container.__file__
            report["differential_helper_source_sha256"] = digest(Path(contact_container.__file__))
            report["differential_history_rows"] = solver.world._contact_container.lambdas.shape[0]
            report["normalized_rotation_increment"] = os.environ.get("COLIBRI_DIFFERENTIAL_NORMALIZED") == "1"
        for name, value in saved.items():
            np.testing.assert_array_equal(getattr(model, name).numpy(), value, err_msg=name)
        report["properties"] = {name: value.tolist() for name, value in saved.items()}
        report["options"] = {k: v for k, v in options.items() if k != "collision_pipeline"}
        report["direct_tree_contacts"] = bool(solver._direct_tree_contacts)
        report["direct_contact_response"] = solver._direct_contact_response is not None
        report["reduced_owned"] = (
            solver._reduced_articulation.owned_joint_mask_np.tolist() if solver._reduced_articulation else None
        )
        if args.native_path == "direct":
            assert report["direct_tree_contacts"] or report["direct_contact_response"], (
                "No joint-conditioned contact owner"
            )
        else:
            assert any(report["reduced_owned"]), "No reduced joint ownership"
        if os.environ.get("COLIBRI_CAPTURE_FRICTION_HISTORY") == "1":
            from local_studies.colibri.capture_native_friction_history import install

            phase_audits.append(install(solver, output))
        print("NATIVE_CONDITIONED_CONFIG", json.dumps(report["options"]), flush=True)

    SolverPhoenX.__init__ = construct
    sys.argv = ["staged_base_mechanism", *remaining]
    try:
        runpy.run_module("local_studies.colibri.staged_base_mechanism", run_name="__main__")
    finally:
        SolverPhoenX.__init__ = original_solver
        newton.CollisionPipeline.__init__ = pipeline_constructor
        report["production_unchanged"] = before == {str(p): digest(p) for p in paths}
        report["overlay_sha256"] = {
            name: hashlib.sha256(value.encode()).hexdigest() for name, value in modified.items()
        }
        report["staged_owned_sha256"] = {
            key: value for key, value in manifest["staged"].items() if "/articulations/" in key
        }
        report["matching_pipelines"] = len(installed)
        report["overlay_directory"] = str(directory)
        if constructed:
            solver = constructed[0]
            snapshot = {}
            for prefix, obj, names in (
                ("body", solver.bodies, ("position", "orientation", "velocity", "angular_velocity")),
                (
                    "direct",
                    solver._direct_equality_system,
                    (
                        "row_joint",
                        "row_local",
                        "row_dynamic",
                        "row_dof",
                        "row_wrench0",
                        "row_wrench1",
                        "row_bias",
                        "row_scale",
                        "dynamic_mass",
                        "velocity_reference",
                        "accumulated_impulse",
                        "joint_to_structural",
                    ),
                ),
                (
                    "reduced",
                    solver._reduced_articulation,
                    ("joint_qd_internal", "joint_factor_diagonal", "joint_implicit_force"),
                ),
            ):
                for name in names:
                    value = getattr(obj, name, None)
                    if value is not None and hasattr(value, "numpy"):
                        snapshot[prefix + "_" + name] = value.numpy().copy()
            for prefix, obj in (
                ("contact", solver.world._contact_container),
                ("contact_solve", solver.world._contact_container_solve),
                ("contact_views", solver.world._contact_views),
                ("contact_columns", solver.world._contact_cols),
                ("contact_response", solver._maximal_contact_response),
                ("direct_contact_response", solver._direct_contact_response),
            ):
                if obj is None:
                    continue
                for name in dir(obj):
                    if name.startswith("_"):
                        continue
                    value = getattr(obj, name, None)
                    if hasattr(value, "numpy"):
                        snapshot[prefix + "_" + name] = value.numpy().copy()
            schedule = getattr(solver.world, "_maximal_contact_schedule", None)
            if schedule is not None:
                snapshot["owned_contact_mobility"] = schedule.mobility.numpy().copy()
            snapshot["contact_valid_count"] = solver.world._cc_valid_count.numpy().copy()
            np.savez_compressed(output.with_suffix(".native_state.npz"), **snapshot)
        if output.exists():
            result = json.loads(output.read_text())
            result["actual_native_configuration"] = report.get("options")
            result["color_group_size"] = report.get("options", {}).get("mass_splitting_color_group_size")
            output.write_text(json.dumps(result, indent=2))
        output.with_suffix(".native_conditioned.json").write_text(json.dumps(report, indent=2))
        for finish_audit in phase_audits:
            finish_audit()
        assert report["production_unchanged"]
        assert "options" in report, "Configured constructor was not invoked"


if __name__ == "__main__":
    main()
