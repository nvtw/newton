# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Reuse actual callback invariants with only expected copy counts adapted."""

import inspect
import unittest

from local_studies.colibri import test_mass_split_bilateral as fixtures
from local_studies.colibri.slab_adapter import install


def suite(width):
    """Keep physical assertions unchanged; group the fixture's three colors."""
    namespace = dict(vars(fixtures))
    original_install = fixtures.install_fused

    def install_fused(solver, *args, **kwargs):
        original_install(solver, *args, **kwargs)
        install(solver, colors_per_slab=width)

    namespace["install_fused"] = install_fused
    source = inspect.getsource(fixtures.TestMassSplitBilateral)
    source = source.replace(
        "self.assertGreaterEqual(int(counts[ids[0] + 1]), 3)",
        f"self.assertEqual(int(counts[ids[0] + 1]), {(3 + width - 1) // width})",
    ).replace(
        "self.assertEqual(int(counts[links[0] + 1]), 3)",
        f"self.assertEqual(int(counts[links[0] + 1]), {(3 + width - 1) // width})",
    ).replace(
        "self.assertEqual(set(root_counts), {1, 2})",
        "self.assertEqual(set(root_counts), " + ("{1, 2}" if width < 3 else "{1}") + ")",
    )
    exec(compile(source, __file__ + f".width{width}", "exec"), namespace)
    return unittest.defaultTestLoader.loadTestsFromTestCase(namespace["TestMassSplitBilateral"])


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(unittest.TestSuite([suite(1), suite(2), suite(8)]))
    raise SystemExit(not result.wasSuccessful())
