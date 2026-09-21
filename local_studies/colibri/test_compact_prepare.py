# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare bounded prepare launches with the original grid on changing contacts."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

import newton
from local_studies.colibri.compact_prepare_dispatch import install
from newton._src.solvers.phoenx.tests.test_rigid_split_prepare import run_scene


class TestCompactPrepare(unittest.TestCase):
    def test_bounded_launches_preserve_physical_state(self):
        """Match all contact/body arrays while reducing idle prepare lanes."""
        original = PhoenXWorld._singleworld_head_plus_tail_sweep
        install()
        candidate = PhoenXWorld._singleworld_head_plus_tail_sweep
        PhoenXWorld._singleworld_head_plus_tail_sweep = original
        initialize = newton.solvers.SolverPhoenX.__init__
        launch = wp.launch
        for split in (False, True):
            for chunk in (1, 3, 6, 128, 129):
                with self.subTest(split=split, chunk=chunk):

                    def init(solver, *args, chunk_size=chunk, **kwargs):
                        kwargs["contact_chunk_size"] = chunk_size
                        initialize(solver, *args, **kwargs)

                    dimensions = []

                    def record(kernel, *args, recorded_dimensions=dimensions, **kwargs):
                        if "_get_parallel_contact_prepare_kernel" in kernel.func.__qualname__:
                            recorded_dimensions.append(kwargs["dim"])
                        return launch(kernel, *args, **kwargs)

                    with patch.object(newton.solvers.SolverPhoenX, "__init__", init):
                        reference = run_scene(True, mass_splitting=split)
                        with (
                            patch.object(PhoenXWorld, "_singleworld_head_plus_tail_sweep", candidate),
                            patch.object(wp, "launch", record),
                        ):
                            actual = run_scene(True, mass_splitting=split)
                    self.assertTrue(dimensions)
                    self.assertTrue(all(dim[1] == min(128, chunk) for dim in dimensions))
                    for before, after in zip(reference, actual, strict=True):
                        for name in before:
                            np.testing.assert_array_equal(
                                before[name].view(np.uint8), after[name].view(np.uint8), err_msg=name
                            )


if __name__ == "__main__":
    unittest.main()
