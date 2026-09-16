"""Run near-blocked mu=0 regression with exact native fixture settings."""

import importlib
import os
import sys
import unittest
from pathlib import Path

from local_studies.colibri.native_conditioned_two_body import STAGE, OwnedFinder


def main():
    """Load corrected owned callbacks before Newton and compare native states."""
    assert not any(name.startswith("newton") for name in sys.modules)
    sys.meta_path.insert(0, OwnedFinder())
    from local_studies.colibri import combined_gap_matching, friction_break_state, friction_differential_reference

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
    friction_differential_reference.install()
    combined_gap_matching.friction_gap_history.install = friction_break_state.install
    combined_gap_matching.install()
    import newton
    from local_studies.colibri import tangent_recovery_trial

    for short in ("direct_contact_gs", "maximal_contact_gs", "reduced_contact_block"):
        name = "newton._src.solvers.phoenx.articulations." + short
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve() == (STAGE / (name.replace(".", "/") + ".py")).resolve()
    solver_type = newton.solvers.SolverPhoenX
    original_init = solver_type.__init__
    original_step = solver_type.step
    captured = []

    def init(solver, *args, **kwargs):
        kwargs["joint_solver"] = "direct"
        return original_init(solver, *args, **kwargs)

    def step(solver, *args, **kwargs):
        result = original_step(solver, *args, **kwargs)
        captured.append((args[1].body_q.numpy().copy(), args[1].body_qd.numpy().copy()))
        return result

    solver_type.__init__ = init
    solver_type.step = step
    path = "newton._src.solvers.phoenx.tests.test_maximal_contact_mobility.TestMaximalContactMobility.test_joint_position_recovery_does_not_launch_internal_contact"
    for candidate in (False, True):
        if candidate:
            tangent_recovery_trial.install()
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName(path))
        assert result.testsRun == 1 and result.wasSuccessful()
    assert len(captured) == 2
    for old, new in zip(captured[0], captured[1], strict=True):
        assert old.tobytes() == new.tobytes()
    print("PASS: near-blocked mu0 baseline/candidate native q and qd byte-identical")


if __name__ == "__main__":
    os.environ["COLIBRI_DIFFERENTIAL_NORMALIZED"] = "1"
    main()
