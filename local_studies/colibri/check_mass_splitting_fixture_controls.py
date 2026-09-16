"""Fail-before controls for the integration fixture's old setup/reset defects."""

import json
from pathlib import Path
from unittest.mock import patch

import warp as wp

from newton._src.solvers.phoenx.mass_splitting.tests import test_solver_integration as fixture


def main():
    wp.init()
    device = wp.get_preferred_device()
    test = fixture.TestMassSplittingPhysicsEquivalence()
    original_launch = wp.launch
    results = {}

    def skip_initialization(*args, **kwargs):
        kernel = kwargs.get("kernel", args[0] if args else None)
        if kernel is not None and kernel.key == "init_phoenx_bodies_kernel":
            return None
        return original_launch(*args, **kwargs)

    with patch.object(wp, "launch", skip_initialization):
        scene = fixture._build_box_stack_scene(1, mass_splitting=True, device=device)
    try:
        test._run_n_frames(*scene[:6], 10, 1.0 / 60.0)
    except AssertionError as error:
        results["missing_mass_initialization_rejected"] = str(error)
    else:
        raise AssertionError("Missing mass initialization was not detected")

    scene = fixture._build_box_stack_scene(1, mass_splitting=True, device=device)
    with patch.object(fixture, "_sync_phoenx_to_newton", lambda *args: None):
        try:
            test._run_n_frames(*scene[:6], 10, 1.0 / 60.0)
        except AssertionError as error:
            assert "not less than" in str(error), str(error)
            results["reset_each_frame_rejected"] = str(error)
        else:
            raise AssertionError("Resetting every frame was not detected")
    Path("/tmp/mass_splitting_fixture_negative_controls.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results))


if __name__ == "__main__":
    main()
