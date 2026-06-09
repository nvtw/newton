# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for :meth:`PhoenXWorld._pre_compile_dispatch_kernels`.

The hook eagerly instantiates the factory-built PGS dispatcher kernels at
solver construction and hands them to :func:`wp.force_load` so the
``module="unique"`` kernels can compile across worker threads instead of
serialising on the first ``step()`` call.

These tests cover three properties the implementation could break:

1. When ``wp.config.load_module_max_workers`` is ``None``, ``0``, or
   ``1`` the hook is a no-op; behaviour stays bit-identical to the
   pre-hook release.
2. When ``load_module_max_workers > 1`` the hook calls
   :func:`wp.force_load` exactly once with the modules backing every
   kernel the active ``step_layout`` will launch (six for
   ``single_world``, and the active multi-world scheduler kernels for ``multi_world``).
3. End-to-end the solver still produces correct rest-state output when
   parallel compile is enabled (a small box-on-ground sim settles to
   ``z = 0``).
"""

from __future__ import annotations

import unittest
from unittest import mock

import warp as wp

if not wp.get_preferred_device().is_cuda:
    raise unittest.SkipTest("PhoenX tests require CUDA")

from newton._src.solvers.phoenx.solver_phoenx_kernels import (
    get_block_world_kernel,
    get_fast_tail_kernel,
)
from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene


class _RestoreLoadModuleMaxWorkers:
    """Context manager that restores ``wp.config.load_module_max_workers``
    on exit so a test that flips it doesn't leak into the rest of the
    process. The harness reuses one CUDA context across tests."""

    def __init__(self, value):
        self._value = value
        self._saved = None

    def __enter__(self):
        self._saved = wp.config.load_module_max_workers
        wp.config.load_module_max_workers = self._value
        return self

    def __exit__(self, exc_type, exc, tb):
        wp.config.load_module_max_workers = self._saved


def _build_minimal_scene(step_layout: str) -> _PhoenXScene:
    """Tiny single-box-on-plane scene shared across tests.

    Construction is what triggers ``_pre_compile_dispatch_kernels``; we
    don't actually step the world here. The scene parameters are the
    smallest spec that still exercises the dispatcher kernels (joints =
    0, cloth = 0, soft-tets = 0) so kernel instantiation paths match
    the production dispatch.
    """
    scene = _PhoenXScene(step_layout=step_layout, substeps=1, solver_iterations=1)
    scene.add_ground_plane()
    scene.add_box((0.0, 0.0, 1.0), (0.1, 0.1, 0.1))
    scene.finalize()
    return scene


class TestPreCompileDispatchKernelsNoop(unittest.TestCase):
    """Hook must not call wp.force_load when parallel compile is off."""

    def test_unset_workers_skips_force_load(self):
        with _RestoreLoadModuleMaxWorkers(None), mock.patch("warp.force_load") as force_load:
            _build_minimal_scene(step_layout="single_world")
            force_load.assert_not_called()

    def test_zero_workers_skips_force_load(self):
        with _RestoreLoadModuleMaxWorkers(0), mock.patch("warp.force_load") as force_load:
            _build_minimal_scene(step_layout="single_world")
            force_load.assert_not_called()

    def test_one_worker_skips_force_load(self):
        with _RestoreLoadModuleMaxWorkers(1), mock.patch("warp.force_load") as force_load:
            _build_minimal_scene(step_layout="single_world")
            force_load.assert_not_called()


class TestPreCompileDispatchKernelsFires(unittest.TestCase):
    """When workers > 1 the hook must call wp.force_load with the modules
    backing the kernels the active dispatcher will launch."""

    def test_single_world_loads_six_singleworld_kernels(self):
        with _RestoreLoadModuleMaxWorkers(4), mock.patch("warp.force_load") as force_load:
            world = _build_minimal_scene(step_layout="single_world").world
            force_load.assert_called_once()
            kwargs = force_load.call_args.kwargs
            # device + modules + max_workers must all be threaded through.
            self.assertIs(kwargs["device"], world.device)
            self.assertEqual(kwargs["max_workers"], 4)
            loaded_modules = {module.get_module_identifier() for module in kwargs["modules"]}
            expected_modules = {kernel.module.get_module_identifier() for kernel in world._singleworld_kernels()}
            # The six head/fused kernels share a module pair-wise (one
            # warp Module per ``module="unique"`` kernel) so the set
            # should contain every distinct module the factory produced.
            self.assertEqual(loaded_modules, expected_modules)

    def test_multi_world_loads_active_scheduler_kernels(self):
        with _RestoreLoadModuleMaxWorkers(4), mock.patch("warp.force_load") as force_load:
            world = _build_minimal_scene(step_layout="multi_world").world
            force_load.assert_called_once()
            loaded_modules = {module.get_module_identifier() for module in force_load.call_args.kwargs["modules"]}
            dispatch_kw = world._dispatch_specialization_flags()
            expected_modules = set()
            if world._multi_world_scheduler == "block_world" and world._block_world_supported():
                block_world_kw = {
                    "revolute_only": dispatch_kw["revolute_only"],
                    "has_joints": dispatch_kw["has_joints"],
                    "has_contacts": world.max_contact_columns > 0,
                    "has_sleeping": dispatch_kw["has_sleeping"],
                    "has_soft_contact_pd": dispatch_kw["has_soft_contact_pd"],
                    "enable_column_timers": dispatch_kw["enable_column_timers"],
                    "family_split": world._fast_tail_family_split(),
                    "block_dim": world._multi_world_block_dim,
                }
                expected_modules.add(
                    get_block_world_kernel(kind="prepare_plus_iterate", **block_world_kw).module.get_module_identifier()
                )
                expected_modules.add(
                    get_block_world_kernel(kind="relax", **block_world_kw).module.get_module_identifier()
                )
            else:
                base_fast_tail_kw = {
                    **dispatch_kw,
                    "has_contacts": world.max_contact_columns > 0,
                }
                for fixed_tpw in world._fast_tail_auto_fixed_choices():
                    fast_tail_kw = {
                        **base_fast_tail_kw,
                        "fixed_tpw": fixed_tpw,
                        "guard_tpw": world._tpw_auto,
                        "family_split": world._fast_tail_family_split(),
                    }
                    expected_modules.add(
                        get_fast_tail_kernel(kind="prepare_plus_iterate", **fast_tail_kw).module.get_module_identifier()
                    )
                    expected_modules.add(
                        get_fast_tail_kernel(kind="relax", **fast_tail_kw).module.get_module_identifier()
                    )
            self.assertEqual(loaded_modules, expected_modules)


class TestPreCompileDispatchKernelsBehaviour(unittest.TestCase):
    """Parallel compile path must not change simulation output."""

    def test_box_settles_with_parallel_compile_enabled(self):
        """Drop a box onto the ground; confirm it rests at z >= 0 within a
        sub-mm tolerance with ``load_module_max_workers = 4``. Catches a
        broken parallel load that produces corrupted PTX (the box would
        either not settle or drift through the plane).
        """
        with _RestoreLoadModuleMaxWorkers(4):
            scene = _build_minimal_scene(step_layout="single_world")
            for _ in range(60):
                scene.step()
            settled_z = float(scene.body_position(0)[2])
            # Box half-extent 0.1 m; centre should rest near 0.1 m above
            # the plane. Allow 1 mm of jitter / penetration so the test
            # isn't sensitive to PGS iteration count.
            self.assertGreater(settled_z, 0.099)
            self.assertLess(settled_z, 0.105)


if __name__ == "__main__":
    unittest.main()
