# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact-sensor tests for :class:`SolverPhoenX`.

Two layers of coverage:

1. **Kernel wiring** -- :class:`TestContactImpulseToForceKernel` calls
   :func:`_contact_impulse_to_force_wrapper_kernel` directly with a
   synthetic :class:`ContactContainer`, a known non-identity
   ``sort_perm`` permutation, and known ``rigid_contact_count`` /
   ``has_perm`` flags. Asserts each output slot lands at
   ``force_out[sort_perm[k]]`` with the correct sign and magnitude.
   This is the **definitive** regression test for the two
   ``update_contacts`` fixes: it bypasses the sort itself, the narrow
   phase, and all physics, so it catches both the sort-permutation
   wiring and the "force on shape0 vs shape1" sign convention
   regardless of scene geometry.

2. **End-to-end physics** -- :class:`TestSensorContactPhoenX` drives
   :class:`~newton.sensors.SensorContact` through the full
   ``collide -> step -> update_contacts -> sensor.update`` pipeline
   inside a CUDA graph. Validates that physically-meaningful settled
   forces (``m*g``, stack weight propagation, friction at rest) come
   through the sensor unchanged. These are tighter than typical
   "settle" tolerances because the readback is just unpacking already-
   converged solver impulses; per-frame numerical noise is the only
   slack budget.

   These integration tests catch the sign-convention bug (any negation
   error flips Fz on a settled body) but NOT the sort-permutation bug
   in isolation: when all contacts share a single body pair (e.g. a
   compound body on a flat ground), sort_perm is undefined within the
   group and the misalignment swaps physically symmetric values that
   look identical to the sensor. The kernel test above is what pins
   the sort_perm wiring.

All tests skip without CUDA (PhoenX is GPU-only) and use CUDA graph
capture in the integration tests; graph-capture-safe is the shipping
execution mode for ``SolverPhoenX``, so a fix that breaks graph
capture must surface here.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.constraints.contact_container import (
    contact_container_zeros,
)
from newton._src.solvers.phoenx.solver_kernels import (
    _contact_impulse_to_force_wrapper_kernel,
)
from newton.sensors import SensorContact

_G = 9.81
_FPS = 120
_DT = 1.0 / _FPS


def _step_with_sensor_graph(
    model: newton.Model,
    solver: newton.solvers.SolverPhoenX,
    state_in: newton.State,
    state_out: newton.State,
    control: newton.Control,
    contacts: newton.Contacts,
    sensors: list[SensorContact],
    *,
    n_frames: int,
) -> None:
    """Run ``n_frames`` of (collide, step, update_contacts, sensor.update)
    through CUDA graph capture.

    First call performs a warm-up step eagerly (compiles every kernel
    Warp will touch and primes any lazy scratch allocations); the next
    step is recorded into a graph and replayed for the remaining
    frames.
    """
    device = wp.get_device()
    assert device.is_cuda, "graph-captured sensor tests require CUDA"

    def _frame() -> None:
        state_in.clear_forces()
        model.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, _DT)
        solver.update_contacts(contacts, state_out)
        for s in sensors:
            s.update(state_out, contacts)
        # Mirror the new pose back into ``state_in`` so the next frame
        # begins from the just-integrated state. Tests reuse a single
        # ``state_in`` / ``state_out`` pair to keep the graph
        # composition simple.
        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)

    if n_frames < 1:
        return

    _frame()  # warm-up
    if n_frames == 1:
        return

    with wp.ScopedCapture(device=device) as capture:
        _frame()
    graph = capture.graph

    for _ in range(n_frames - 2):
        wp.capture_launch(graph)


def _make_solver(model: newton.Model, *, step_layout: str = "multi_world") -> newton.solvers.SolverPhoenX:
    return newton.solvers.SolverPhoenX(
        model,
        substeps=4,
        solver_iterations=20,
        velocity_iterations=2,
        step_layout=step_layout,
    )


# ContactContainer host-side row layout mirrors ``contact_container.py``.
_IMP_ROW_NORMAL_LAMBDA = 0
_IMP_ROW_TANGENT1_LAMBDA = 1
_IMP_ROW_TANGENT2_LAMBDA = 2
_MAN_ROW_NORMAL_X = 0
_MAN_ROW_TANGENT1_X = 3


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX contact-sensor tests require CUDA")
class TestContactImpulseToForceKernel(unittest.TestCase):
    """Direct kernel test for ``_contact_impulse_to_force_wrapper_kernel``.

    Constructs a synthetic :class:`ContactContainer` with distinguishable
    impulses per slot, hands the kernel a known non-identity sort
    permutation, and verifies each output row lands at
    ``force_out[sort_perm[k]]`` with the negated impulse-times-frame.
    Bypasses the entire physics + ingest pipeline so a regression in
    either the sort_perm wiring or the sign convention triggers a
    deterministic failure regardless of scene, broad-phase output
    order, or radix-sort stability.
    """

    def _make_cc_with_impulses(self, n: int, lam_n: list[float], device):
        """Allocate a ContactContainer, fill ``n`` slots with axis-aligned
        normals (+Z), tangents (+X), zero tangent impulses, and the
        provided ``lam_n`` per slot.
        """
        cc = contact_container_zeros(n, device=device)
        impulse_host = cc.impulses.numpy()
        impulse_host[:] = 0.0
        impulse_host[_IMP_ROW_NORMAL_LAMBDA, :] = np.asarray(lam_n, dtype=np.float32)
        impulse_host[_IMP_ROW_TANGENT1_LAMBDA, :] = 0.0
        impulse_host[_IMP_ROW_TANGENT2_LAMBDA, :] = 0.0
        cc.impulses.assign(impulse_host)

        manifold_host = cc.lambdas.numpy()
        manifold_host[:] = 0.0
        manifold_host[_MAN_ROW_NORMAL_X + 0, :] = 0.0  # n.x
        manifold_host[_MAN_ROW_NORMAL_X + 1, :] = 0.0  # n.y
        manifold_host[_MAN_ROW_NORMAL_X + 2, :] = 1.0  # n.z
        manifold_host[_MAN_ROW_TANGENT1_X + 0, :] = 1.0  # t1.x (orthogonal to n)
        manifold_host[_MAN_ROW_TANGENT1_X + 1, :] = 0.0
        manifold_host[_MAN_ROW_TANGENT1_X + 2, :] = 0.0
        cc.lambdas.assign(manifold_host)
        return cc

    def _launch_readback(
        self,
        n: int,
        lam_n: list[float],
        sort_perm: np.ndarray,
        has_perm: int,
        n_active: int | None = None,
        idt: float = 1.0,
    ) -> np.ndarray:
        """Run one kernel launch and return ``force_out`` linear part as
        ``(n, 3)`` numpy array. ``n_active`` defaults to ``n``.
        """
        device = wp.get_device("cuda:0")
        cc = self._make_cc_with_impulses(n, lam_n, device)
        sort_perm_wp = wp.array(sort_perm.astype(np.int32), dtype=wp.int32, device=device)
        rigid_contact_count = wp.array(
            np.asarray([n if n_active is None else n_active], dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )
        force_out = wp.zeros(n, dtype=wp.spatial_vector, device=device)
        wp.launch(
            _contact_impulse_to_force_wrapper_kernel,
            dim=n,
            inputs=[rigid_contact_count, cc, wp.float32(idt), sort_perm_wp, wp.int32(has_perm)],
            outputs=[force_out],
            device=device,
        )
        # spatial_vector is (linear, angular) -- linear is the first
        # vec3 (top), angular is the second. Slice the first 3 cols.
        return force_out.numpy().reshape(n, 6)[:, :3]

    def test_sort_perm_routes_force_to_newton_slot(self) -> None:
        """A non-identity ``sort_perm`` must move each kernel-thread's
        wrench to ``force_out[sort_perm[k]]``. With distinct ``lam_n``
        per slot the expected vector at every newton-order slot is
        unambiguous, so a regression to ``force_out[k]`` (the pre-fix
        path) corrupts every entry.

        Sign convention: ``f = -lam_n * n_hat`` because Newton stores
        force on shape0 while phoenx accumulates impulses on shape1.
        """
        n = 4
        # Distinct, easily-identifiable values. With idt=1.0 and
        # +Z normal, expected force at sorted_k is f = (0, 0, -lam_n).
        lam_n = [10.0, 20.0, 30.0, 40.0]
        # Permutation: sort_perm[sorted_k] -> newton_k. Picked so every
        # slot lands somewhere different from sorted_k.
        sort_perm = np.array([2, 0, 3, 1], dtype=np.int32)

        out = self._launch_readback(n=n, lam_n=lam_n, sort_perm=sort_perm, has_perm=1)

        # Expected: force_out[sort_perm[k]] = (0, 0, -lam_n[k]).
        expected = np.zeros((n, 3), dtype=np.float32)
        for sorted_k in range(n):
            newton_k = int(sort_perm[sorted_k])
            expected[newton_k, 2] = -lam_n[sorted_k]
        np.testing.assert_allclose(out, expected, atol=1.0e-5, rtol=0.0)

    def test_has_perm_zero_writes_at_thread_index(self) -> None:
        """``has_perm=0`` must take the identity path even when the
        ``sort_perm`` array is non-trivial. Guards against a regression
        that always reads ``sort_perm`` (wrong) or always ignores it
        (would also break the grouping path).
        """
        n = 3
        lam_n = [1.0, 2.0, 3.0]
        # Non-identity sort_perm -- but has_perm=0, so it must be ignored.
        sort_perm = np.array([2, 1, 0], dtype=np.int32)

        out = self._launch_readback(n=n, lam_n=lam_n, sort_perm=sort_perm, has_perm=0)

        expected = np.zeros((n, 3), dtype=np.float32)
        expected[:, 2] = -np.asarray(lam_n, dtype=np.float32)  # force_out[k] = (0, 0, -lam_n[k])
        np.testing.assert_allclose(out, expected, atol=1.0e-5, rtol=0.0)

    def test_friction_components_assemble_correctly(self) -> None:
        """``f = -(lam_n * n + lam_t1 * t1 + lam_t2 * t2) * idt`` with
        ``t2 = cross(n, t1)``. With n=+Z, t1=+X, t2=+Y the three
        impulse components map to (-x, -y, -z) under the sign flip.
        Pin every component so a regression in any one column or in
        the t2 reconstruction is caught.
        """
        device = wp.get_device("cuda:0")
        n = 1
        cc = contact_container_zeros(n, device=device)
        impulse_host = cc.impulses.numpy()
        impulse_host[:] = 0.0
        # Distinct values per axis so a swap is observable.
        impulse_host[_IMP_ROW_NORMAL_LAMBDA, 0] = 5.0  # lam_n
        impulse_host[_IMP_ROW_TANGENT1_LAMBDA, 0] = 7.0  # lam_t1
        impulse_host[_IMP_ROW_TANGENT2_LAMBDA, 0] = 11.0  # lam_t2
        cc.impulses.assign(impulse_host)

        manifold_host = cc.lambdas.numpy()
        manifold_host[:] = 0.0
        manifold_host[_MAN_ROW_NORMAL_X + 2, 0] = 1.0  # n = +Z
        manifold_host[_MAN_ROW_TANGENT1_X + 0, 0] = 1.0  # t1 = +X
        cc.lambdas.assign(manifold_host)

        sort_perm = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=device)
        rigid_contact_count = wp.array(np.array([1], dtype=np.int32), dtype=wp.int32, device=device)
        force_out = wp.zeros(n, dtype=wp.spatial_vector, device=device)
        idt = 2.0  # exercise non-unit dt scaling
        wp.launch(
            _contact_impulse_to_force_wrapper_kernel,
            dim=n,
            inputs=[rigid_contact_count, cc, wp.float32(idt), sort_perm, wp.int32(0)],
            outputs=[force_out],
            device=device,
        )
        # t2 = n x t1 = (+Z) x (+X) = +Y, so the impulse decomposes
        # to lam_n*Z + lam_t1*X + lam_t2*Y; force = -idt * that.
        expected = -idt * np.array([7.0, 11.0, 5.0], dtype=np.float32)  # (-t1, -t2, -n)
        out = force_out.numpy().reshape(n, 6)[0, :3]
        np.testing.assert_allclose(out, expected, atol=1.0e-5, rtol=0.0)

    def test_inactive_slots_left_zero(self) -> None:
        """Slots beyond ``rigid_contact_count[0]`` must not be written
        (the kernel ``return``s early). Stale tail data in ContactContainer
        from a previous frame would otherwise leak into ``force_out``,
        which the sensor reads in full ``rigid_contact_max`` extent.
        """
        n = 4
        # All slots have non-zero lambdas so a regression that drops
        # the bound check would leave non-zero values in tail entries.
        lam_n = [100.0, 200.0, 300.0, 400.0]
        sort_perm = np.array([0, 1, 2, 3], dtype=np.int32)

        # Only the first 2 slots are active.
        out = self._launch_readback(n=n, lam_n=lam_n, sort_perm=sort_perm, has_perm=0, n_active=2)

        expected = np.zeros((n, 3), dtype=np.float32)
        expected[0, 2] = -100.0
        expected[1, 2] = -200.0
        # Slots 2, 3 must remain zero (force_out was zero-initialized).
        np.testing.assert_allclose(out, expected, atol=1.0e-5, rtol=0.0)


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX contact-sensor tests require CUDA")
class TestSensorContactPhoenX(unittest.TestCase):
    """End-to-end checks: ``SensorContact`` + ``SolverPhoenX``."""

    def _settle_with_sensor(self, model, solver, sensors, n_frames):
        contacts = model.contacts()
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        _step_with_sensor_graph(
            model,
            solver,
            state_in,
            state_out,
            control,
            contacts,
            sensors,
            n_frames=n_frames,
        )
        return state_out, contacts

    # ------------------------------------------------------------------
    # Single-body baseline: non-grouping path
    # ------------------------------------------------------------------

    def test_sphere_on_ground_total_force(self) -> None:
        """Single sphere on a plane -- ``|total_force| ~= m*g`` after
        settling. Catches:

        * sign-convention regressions in
          :func:`_contact_impulse_to_force_wrapper_kernel` (Fz upward
          fails immediately if the sign is wrong; ``-m*g`` is what
          the broken kernel reports);
        * lateral-force leak (a non-zero in-plane component would
          indicate that the negated tangents leaked into the readback
          or that the sign was inconsistently applied across axes);
        * total-force magnitude regressions of more than 1.5 % from
          ``m*g``.

        Tolerance is tight (1.5 %) because the readback is just
        unpacking already-converged solver impulses -- the only error
        budget is per-frame settling oscillation, which 120 frames at
        4 substeps reduces below 1 % on this scene.
        """
        mass = 2.0
        radius = 0.1
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(
            xform=wp.transform((0.0, 0.0, radius + 0.05), wp.quat_identity()),
            mass=mass,
            inertia=(
                (0.4 * mass * radius * radius, 0.0, 0.0),
                (0.0, 0.4 * mass * radius * radius, 0.0),
                (0.0, 0.0, 0.4 * mass * radius * radius),
            ),
        )
        builder.add_shape_sphere(
            body,
            radius=radius,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
            label="ball",
        )
        model = builder.finalize()

        sensor = SensorContact(model, sensing_obj_shapes="ball", verbose=False)
        solver = _make_solver(model)

        # Non-compound scene: grouping must be off so we exercise the
        # ``has_perm=0`` branch of the readback kernel.
        self.assertFalse(
            solver.world._enable_body_pair_grouping,
            "single-shape scene should not opt into body-pair grouping",
        )

        self._settle_with_sensor(model, solver, [sensor], n_frames=120)

        total = sensor.total_force.numpy()[0]
        expected = mass * _G
        self.assertGreater(float(total[2]), 0.0, f"Fz should be upward, got {total}")
        rel_err = abs(float(total[2]) - expected) / expected
        self.assertLess(
            rel_err,
            0.015,
            f"sphere total Fz = {float(total[2]):.4f} N vs m*g = {expected:.4f} N (rel err {rel_err:.2%})",
        )
        lateral = math.hypot(float(total[0]), float(total[1]))
        self.assertLess(
            lateral,
            0.01 * expected,
            f"lateral force should be ~0, got ({total[0]:.4f}, {total[1]:.4f}) N",
        )

    # ------------------------------------------------------------------
    # Regression: compound-body grouping path
    # ------------------------------------------------------------------

    def test_compound_body_on_ground_total_force(self) -> None:
        """Compound body (two shapes, same body) on a plane: end-to-end
        smoke test that body-level sensing aggregates correctly when
        :attr:`SolverPhoenX.world._enable_body_pair_grouping` is on.

        Note: this test does **not** isolate the sort-permutation bug.
        When all contacts share a single body pair (``compound_body``
        vs ``world``), the radix sort within that group is order-
        undefined and the misalignment swaps physically symmetric
        contact slots whose Fz are equal by construction; the
        body-level row sum is invariant under that swap.
        :class:`TestContactImpulseToForceKernel` is what pins the
        sort_perm wiring -- this test guards the integration: any
        regression that breaks aggregation across multiple per-shape
        contacts (drop-out in the launch, count gating, etc.) shows
        up here.
        """
        mass = 4.0
        half = 0.05
        spacing = 2.05 * half
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        ixx = mass / 3.0 * (half * half + half * half)
        body = builder.add_body(
            xform=wp.transform((0.0, 0.0, half + 0.02), wp.quat_identity()),
            mass=mass,
            inertia=((ixx, 0, 0), (0, ixx, 0), (0, 0, ixx)),
        )
        # Two shapes attached to the same body -> compound. Distinct
        # labels so the sensor pattern resolves to a unique rule.
        for i, dx in enumerate((-spacing * 0.5, spacing * 0.5)):
            builder.add_shape_box(
                body,
                xform=wp.transform((dx, 0.0, 0.0), wp.quat_identity()),
                hx=half,
                hy=half,
                hz=half,
                cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
                label=f"compound_part_{i}",
            )
        model = builder.finalize()

        # Sensor sees the body, not individual shapes -- the body-level
        # aggregation forces the pipeline to walk every contact slot
        # (including ones from both shapes), exposing any
        # newton-order-vs-sorted-order mismatch.
        sensor = SensorContact(model, sensing_obj_bodies=[body], verbose=False)
        solver = _make_solver(model, step_layout="single_world")

        # The compound body must have triggered grouping; if not, this
        # test isn't actually exercising the regression path.
        self.assertTrue(
            solver.world._enable_body_pair_grouping,
            "compound-body scene should opt into body-pair grouping",
        )
        self.assertIsNotNone(
            solver.world._ingest_scratch.sort_perm,
            "grouping path must allocate sort_perm",
        )

        self._settle_with_sensor(model, solver, [sensor], n_frames=180)

        total = sensor.total_force.numpy()[0]
        expected = mass * _G
        self.assertGreater(float(total[2]), 0.0, f"Fz should be upward, got {total}")
        rel_err = abs(float(total[2]) - expected) / expected
        self.assertLess(
            rel_err,
            0.02,
            f"compound total Fz = {float(total[2]):.4f} N vs m*g = {expected:.4f} N "
            f"(rel err {rel_err:.2%}); aggregated forces from both shapes should equal weight",
        )
        lateral = math.hypot(float(total[0]), float(total[1]))
        self.assertLess(
            lateral,
            0.02 * expected,
            f"lateral force on compound body should be ~0, got ({total[0]:.4f}, {total[1]:.4f}) N",
        )

    # ------------------------------------------------------------------
    # Per-counterpart breakdown: force_matrix
    # ------------------------------------------------------------------

    def test_per_counterpart_force_matrix(self) -> None:
        """Two stacked boxes: the bottom box's ``force_matrix`` row must
        split between (ground -> bottom) and (top_box -> bottom).
        Three independent invariants pin the readback:

        1. Ground supports ``2*m*g`` upward (catches sign on the
           plane-vs-box contact and ensures the per-counterpart split
           routes the heavier load to the right column).
        2. Top box pushes ``-m*g`` downward (catches sign / direction
           mix-up between the two contacts attached to the same
           sensing row).
        3. ``f_from_ground + f_from_top == total_force`` to within
           1e-3 relative; any mismatch means the sensor either
           dropped a contact or double-counted one.

        These three together are the strongest end-to-end check
        available: if any one of force, shape0, shape1, or normal is
        misattributed for either of the two contacts, at least one
        invariant fails.
        """
        mass = 1.0
        half = 0.1
        builder = newton.ModelBuilder()
        builder.add_ground_plane(label="ground")
        ixx = mass / 3.0 * (half * half + half * half)
        bottom = builder.add_body(
            xform=wp.transform((0.0, 0.0, half + 0.01), wp.quat_identity()),
            mass=mass,
            inertia=((ixx, 0, 0), (0, ixx, 0), (0, 0, ixx)),
        )
        builder.add_shape_box(
            bottom,
            hx=half,
            hy=half,
            hz=half,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
            label="bottom_box",
        )
        top = builder.add_body(
            xform=wp.transform((0.0, 0.0, 3 * half + 0.02), wp.quat_identity()),
            mass=mass,
            inertia=((ixx, 0, 0), (0, ixx, 0), (0, 0, ixx)),
        )
        builder.add_shape_box(
            top,
            hx=half,
            hy=half,
            hz=half,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
            label="top_box",
        )
        model = builder.finalize()

        sensor = SensorContact(
            model,
            sensing_obj_shapes="bottom_box",
            counterpart_shapes=["ground", "top_box"],
            verbose=False,
        )
        solver = _make_solver(model)

        self._settle_with_sensor(model, solver, [sensor], n_frames=240)

        # Resolve which column corresponds to which counterpart -- the
        # sensor preserves the user's flat list order (globals first).
        cp_indices = sensor.counterpart_indices[0]
        ground_shape = model.shape_label.index("ground")
        top_shape = model.shape_label.index("top_box")
        ground_col = cp_indices.index(ground_shape)
        top_col = cp_indices.index(top_shape)

        fm = sensor.force_matrix.numpy()  # (1, n_counterparts, 3)
        f_from_ground = fm[0, ground_col]
        f_from_top = fm[0, top_col]
        weight = mass * _G

        # Ground pushes bottom up by the combined weight (2 m*g) within 2 %.
        rel_err_ground = abs(float(f_from_ground[2]) - 2.0 * weight) / (2.0 * weight)
        self.assertLess(
            rel_err_ground,
            0.02,
            f"Fz from ground onto bottom = {float(f_from_ground[2]):.4f} N "
            f"vs 2*m*g = {2.0 * weight:.4f} N (rel err {rel_err_ground:.2%})",
        )
        # Top box pushes down on bottom by m*g within 2 %.
        rel_err_top = abs(float(f_from_top[2]) + weight) / weight
        self.assertLess(
            rel_err_top,
            0.02,
            f"Fz from top onto bottom = {float(f_from_top[2]):.4f} N "
            f"vs -m*g = {-weight:.4f} N (rel err {rel_err_top:.2%})",
        )
        # Off-axis components on each counterpart row must be ~zero --
        # any non-zero lateral component means a non-vertical contact
        # leaked into the wrong column (i.e. shape attribution bug).
        self.assertLess(
            math.hypot(float(f_from_ground[0]), float(f_from_ground[1])),
            0.02 * weight,
            f"f_from_ground lateral should be ~0, got {f_from_ground}",
        )
        self.assertLess(
            math.hypot(float(f_from_top[0]), float(f_from_top[1])),
            0.02 * weight,
            f"f_from_top lateral should be ~0, got {f_from_top}",
        )

        # Per-counterpart sum must equal total_force (within tolerance).
        # SensorContact also exposes total_force when measure_total=True
        # (default), so the two views should reconcile.
        total = sensor.total_force.numpy()[0]
        per_cp_sum = f_from_ground + f_from_top
        np.testing.assert_allclose(
            per_cp_sum,
            total,
            atol=1.0e-3,
            rtol=1.0e-4,
            err_msg=(
                f"per-counterpart sum {per_cp_sum} should match total_force {total} "
                "(any mismatch indicates double-counting or dropped contacts)"
            ),
        )

        # Net Fz on the bottom = +2 m*g - m*g = +m*g (its own weight is
        # the inertial term carried by gravity, not a contact force).
        self.assertGreater(float(total[2]), 0.0, f"net contact Fz on bottom should be upward, got {total}")
        rel_err_total = abs(float(total[2]) - weight) / weight
        self.assertLess(
            rel_err_total,
            0.02,
            f"net contact Fz on bottom = {float(total[2]):.4f} N vs m*g = {weight:.4f} N (rel err {rel_err_total:.2%})",
        )

    # ------------------------------------------------------------------
    # Friction (tangential) decomposition
    # ------------------------------------------------------------------

    def test_friction_normal_decomposition_consistent(self) -> None:
        """At rest, ``total_force = total_force_normal + total_force_friction``
        must reconstruct exactly (up to float roundoff), and the
        friction component must be orthogonal to the contact normal.

        The sensor's :func:`accumulate_contact_forces_kernel` derives
        friction by projecting ``contacts.force`` onto
        ``rigid_contact_normal`` and storing the residual. Two
        invariants pin the readback:

        1. The decomposition reconstructs the total within tight
           floating-point tolerance.
        2. The friction residual is small relative to the weight at
           rest (no driving lateral force, so tangential impulses
           must be near zero -- a non-trivial friction here flags
           either a solver-side normal-row tangent leak OR (more
           relevant to this test) a mismatch between the cc-stored
           normal and ``rigid_contact_normal``: if those two normals
           disagree, the friction projection picks up a fictitious
           tangential component proportional to the normal-component
           of the readback force, which is huge.
        """
        mass = 1.0
        half = 0.1
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        ixx = mass / 3.0 * (half * half + half * half)
        body = builder.add_body(
            xform=wp.transform((0.0, 0.0, half + 0.01), wp.quat_identity()),
            mass=mass,
            inertia=((ixx, 0, 0), (0, ixx, 0), (0, 0, ixx)),
        )
        builder.add_shape_box(
            body,
            hx=half,
            hy=half,
            hz=half,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
            label="cube",
        )
        model = builder.finalize()

        sensor = SensorContact(model, sensing_obj_shapes="cube", verbose=False)
        solver = _make_solver(model)

        self._settle_with_sensor(model, solver, [sensor], n_frames=240)

        total = sensor.total_force.numpy()[0]
        friction = sensor.total_force_friction.numpy()[0]
        weight = mass * _G

        # Total normal-direction force converges to m*g.
        rel_err_total = abs(float(total[2]) - weight) / weight
        self.assertLess(
            rel_err_total,
            0.02,
            f"total Fz = {float(total[2]):.4f} N vs m*g = {weight:.4f} N (rel err {rel_err_total:.2%})",
        )
        # Friction at rest with no lateral driver must be tiny (< 1 % of weight).
        # If cc.normal and rigid_contact_normal disagreed, this would
        # be O(weight) (the projection of an upward force against a
        # not-quite-up normal yields a large tangential residual).
        f_mag = float(np.linalg.norm(friction))
        self.assertLess(
            f_mag,
            0.01 * weight,
            f"friction magnitude {f_mag:.4f} N at rest should be << weight {weight:.4f} N (got {friction})",
        )

        # Decomposition must be self-consistent: if friction is
        # ``f - (f.n) n``, then ``f - friction`` must be parallel to
        # the contact normal axis (here +Z).
        normal_residual = total - friction
        self.assertLess(
            math.hypot(float(normal_residual[0]), float(normal_residual[1])),
            1.0e-3,
            f"total - friction must be along the normal (here +Z), got {normal_residual}",
        )

    # ------------------------------------------------------------------
    # Tilted-normal contacts: friction decomposition with non-+Z normals
    # ------------------------------------------------------------------

    def test_tilted_normal_corner_force_balance(self) -> None:
        """Sphere wedged in a corner (ground + vertical wall) with
        tilted gravity. Pinning multiple invariants:

        1. Total force on the sphere = -gravity (force balance at rest).
        2. Each per-counterpart force has a normal component along the
           respective contact's outward normal AND a tangential
           (friction) component perpendicular to it.
        3. The sensor's :attr:`force_matrix_friction` field exactly
           reproduces the tangential decomposition derived from
           :attr:`force_matrix` and ``rigid_contact_normal``.

        This is the test that catches a normal-direction mismatch
        between cc.normal (used to assemble the readback) and
        rigid_contact_normal (used for the friction projection).
        Identical-vertical-normal scenes can't distinguish the two.
        """
        mass, radius = 1.0, 0.05
        gx, gz = 4.0, -_G  # tilt gravity into +X so ball presses against +X wall

        builder = newton.ModelBuilder()
        builder.add_ground_plane(label="ground")
        builder.add_shape_box(
            -1,
            xform=wp.transform(p=wp.vec3(0.25, 0.0, 0.5)),
            hx=0.05,
            hy=0.5,
            hz=0.5,
            label="wall",
        )
        ixx = 0.4 * mass * radius * radius
        body = builder.add_body(
            xform=wp.transform((0.05, 0.0, radius + 0.01), wp.quat_identity()),
            mass=mass,
            inertia=((ixx, 0, 0), (0, ixx, 0), (0, 0, ixx)),
        )
        builder.add_shape_sphere(
            body,
            radius=radius,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
            label="ball",
        )
        model = builder.finalize()
        model.set_gravity((gx, 0.0, gz))

        sensor = SensorContact(
            model,
            sensing_obj_shapes="ball",
            counterpart_shapes=["ground", "wall"],
            verbose=False,
        )
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=4,
            solver_iterations=40,
            velocity_iterations=2,
            default_friction=2.0,
        )
        self._settle_with_sensor(model, solver, [sensor], n_frames=600)

        total = sensor.total_force.numpy()[0]
        expected_total = np.array([-mass * gx, 0.0, -mass * gz], dtype=np.float32)
        # Tilted-gravity scenes have a small residual oscillation after
        # 600 frames; tolerate 0.5 % rel err on the magnitude.
        np.testing.assert_allclose(
            total,
            expected_total,
            atol=0.05,
            rtol=5.0e-3,
            err_msg=f"total force on sphere must equal -gravity, got {total} vs {expected_total}",
        )

        cp = sensor.counterpart_indices[0]
        g_col = cp.index(model.shape_label.index("ground"))
        w_col = cp.index(model.shape_label.index("wall"))
        f_g = sensor.force_matrix.numpy()[0, g_col]
        f_w = sensor.force_matrix.numpy()[0, w_col]

        # Both contacts must carry a meaningful load (not converge to zero).
        # Ground supports z; wall supports x. Scale guards: more than 5 % of total.
        self.assertGreater(
            abs(float(f_g[2])),
            0.05 * float(np.linalg.norm(expected_total)),
            f"ground should carry vertical load, got {f_g}",
        )
        self.assertGreater(
            abs(float(f_w[0])),
            0.05 * float(np.linalg.norm(expected_total)),
            f"wall should carry horizontal load, got {f_w}",
        )
        # Per-counterpart sum must reconcile with total.
        np.testing.assert_allclose(
            f_g + f_w,
            total,
            atol=1.0e-2,
            err_msg=f"per-counterpart sum {f_g + f_w} should match total {total}",
        )

        # Friction decomposition: ``force_matrix_friction[i,j]`` should
        # equal ``f - (f.n) n`` analytically. Pick the contact normal
        # from the contacts buffer post-step; phoenx's stored normal
        # must agree with what the sensor uses.
        fm_fric = sensor.force_matrix_friction.numpy()

        # Ground normal is +Z; wall normal is along -X (wall at +X side of ball).
        n_ground = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        n_wall = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        analyt_fric_g = f_g - np.dot(f_g, n_ground) * n_ground
        analyt_fric_w = f_w - np.dot(f_w, n_wall) * n_wall
        np.testing.assert_allclose(
            fm_fric[0, g_col],
            analyt_fric_g,
            atol=1.0e-2,
            err_msg=(
                f"sensor friction at ground {fm_fric[0, g_col]} should match analytical "
                f"{analyt_fric_g} = f - (f.n)n. Mismatch flags a normal-direction "
                "disagreement between cc.normal (readback) and rigid_contact_normal "
                "(sensor projection)."
            ),
        )
        np.testing.assert_allclose(
            fm_fric[0, w_col],
            analyt_fric_w,
            atol=1.0e-2,
            err_msg=(f"sensor friction at wall {fm_fric[0, w_col]} should match analytical {analyt_fric_w}"),
        )

    # ------------------------------------------------------------------
    # Articulated robot: contact + joint coupling
    # ------------------------------------------------------------------

    def test_articulated_robot_per_link_weight(self) -> None:
        """Two-link revolute robot lying flat on the ground: each link
        must report exactly its own weight in vertical contact force.

        The free joint anchoring link_a and the revolute joint
        connecting the two links carry no vertical load in this
        equilibrium (revolute axis is +Y so internal joint moments
        live in the X-Z plane but cancel under symmetric gravity
        loading). If phoenx's contact + ADBS constraint coupling
        leaked any vertical impulse into the contact rows, per-link
        Fz would deviate from m*g.
        """
        mass = 1.0
        half = 0.05

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        ixx = mass / 3.0 * (half * half + half * half)
        link_a = builder.add_body(
            xform=wp.transform((0.0, 0.0, half + 0.005), wp.quat_identity()),
            mass=mass,
            inertia=((ixx, 0, 0), (0, ixx, 0), (0, 0, ixx)),
        )
        builder.add_shape_box(
            link_a,
            hx=half,
            hy=half,
            hz=half,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
            label="link_a",
        )
        link_b = builder.add_body(
            xform=wp.transform((2.05 * half, 0.0, half + 0.005), wp.quat_identity()),
            mass=mass,
            inertia=((ixx, 0, 0), (0, ixx, 0), (0, 0, ixx)),
        )
        builder.add_shape_box(
            link_b,
            hx=half,
            hy=half,
            hz=half,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
            label="link_b",
        )
        builder.add_joint_free(parent=-1, child=link_a)
        builder.add_joint_revolute(
            parent=link_a,
            child=link_b,
            axis=newton.Axis.Y,
            parent_xform=wp.transform((1.025 * half, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform((-1.025 * half, 0.0, 0.0), wp.quat_identity()),
        )
        model = builder.finalize()

        sensor_a = SensorContact(model, sensing_obj_shapes="link_a", verbose=False)
        sensor_b = SensorContact(model, sensing_obj_shapes="link_b", verbose=False)
        solver = newton.solvers.SolverPhoenX(model, substeps=4, solver_iterations=40, velocity_iterations=2)

        self._settle_with_sensor(model, solver, [sensor_a, sensor_b], n_frames=600)

        f_a = sensor_a.total_force.numpy()[0]
        f_b = sensor_b.total_force.numpy()[0]
        weight = mass * _G

        # Each link reports exactly its own weight upward.
        for label, f in (("link_a", f_a), ("link_b", f_b)):
            rel_err = abs(float(f[2]) - weight) / weight
            self.assertLess(
                rel_err,
                0.02,
                f"{label} Fz = {float(f[2]):.4f} N vs m*g = {weight:.4f} N (rel err {rel_err:.2%})",
            )
        # Sum equals total weight.
        np.testing.assert_allclose(
            float(f_a[2] + f_b[2]),
            2.0 * weight,
            rtol=0.01,
            err_msg=f"sum of per-link Fz = {f_a[2] + f_b[2]:.4f} should be 2 m*g = {2 * weight:.4f}",
        )

    # ------------------------------------------------------------------
    # Multi-substep semantics
    # ------------------------------------------------------------------

    def test_substep_steady_state_invariant(self) -> None:
        """``substeps=1`` and ``substeps=8`` must converge to the same
        steady-state contact force.

        The readback divides by ``_last_dt`` (the substep dt), so the
        sensor reports a per-substep average. In steady state every
        substep carries the same impulse, so the per-frame readout
        must equal m*g regardless of substep count. A regression that
        forgot to scale by 1/dt or used the wrong dt would surface
        as a per-substep multiplier on Fz.
        """

        def settle_and_read(substeps: int) -> float:
            mass, radius = 1.0, 0.05
            builder = newton.ModelBuilder()
            builder.add_ground_plane()
            ixx = 0.4 * mass * radius * radius
            body = builder.add_body(
                xform=wp.transform((0.0, 0.0, radius + 0.05), wp.quat_identity()),
                mass=mass,
                inertia=((ixx, 0, 0), (0, ixx, 0), (0, 0, ixx)),
            )
            builder.add_shape_sphere(
                body,
                radius=radius,
                cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
                label="ball",
            )
            model = builder.finalize()
            sensor = SensorContact(model, sensing_obj_shapes="ball", verbose=False)
            solver = newton.solvers.SolverPhoenX(model, substeps=substeps, solver_iterations=20, velocity_iterations=2)
            self._settle_with_sensor(model, solver, [sensor], n_frames=120)
            return float(sensor.total_force.numpy()[0][2])

        fz_1 = settle_and_read(1)
        fz_8 = settle_and_read(8)
        weight = 1.0 * _G
        for label, fz in (("substeps=1", fz_1), ("substeps=8", fz_8)):
            rel_err = abs(fz - weight) / weight
            self.assertLess(
                rel_err,
                0.02,
                f"{label}: settled Fz = {fz:.4f} N vs m*g = {weight:.4f} N (rel err {rel_err:.2%})",
            )
        # Both substep counts converge to the same answer.
        self.assertLess(
            abs(fz_1 - fz_8),
            0.02 * weight,
            f"substeps=1 ({fz_1:.4f}) and substeps=8 ({fz_8:.4f}) disagree by more than 2%",
        )

    # ------------------------------------------------------------------
    # Attribute-request flow
    # ------------------------------------------------------------------

    def test_request_force_attribute_flow(self) -> None:
        """Constructing ``SensorContact`` before ``model.contacts()``
        must implicitly request the ``force`` extended attribute, so
        the resulting ``Contacts`` object has ``force`` allocated and
        ``solver.update_contacts()`` works without manual setup.
        """
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(
            xform=wp.transform((0.0, 0.0, 0.2), wp.quat_identity()),
            mass=1.0,
            inertia=((1.0e-2, 0, 0), (0, 1.0e-2, 0), (0, 0, 1.0e-2)),
        )
        builder.add_shape_sphere(
            body,
            radius=0.1,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
            label="ball",
        )
        model = builder.finalize()

        # Sensor first -- must request "force" before the Contacts
        # buffer is built.
        SensorContact(model, sensing_obj_shapes="ball", verbose=False)
        self.assertIn("force", model.get_requested_contact_attributes())

        # ``SolverPhoenX.__init__`` builds a CollisionPipeline; the
        # subsequent ``model.contacts()`` should pick up "force".
        _make_solver(model)
        contacts = model.contacts()
        self.assertIsNotNone(
            contacts.force,
            "SensorContact-then-SolverPhoenX flow should yield Contacts with force allocated",
        )


if __name__ == "__main__":
    unittest.main()
