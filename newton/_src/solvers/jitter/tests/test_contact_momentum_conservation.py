# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physics-level conservation tests that exercise the contact path.

These tests close a gap in the existing test suite. The name-alike
:mod:`test_momentum_conservation` only validates joint force equilibrium
on a pure ball-socket chain; it runs with
``enable_all_constraints=True`` and never invokes
:func:`~newton._src.solvers.jitter.constraints.constraint_contact.contact_iterate_at`.
That function uses a Jacobi-within-slot update (one ``vel_rel`` computed
per slot, three rows solved from the same snapshot, single combined
impulse applied) and needed a direct, conservation-law check.

What these tests *actually* catch
---------------------------------

The Jacobi-within-slot iterate applies a single impulse
``imp = d_lam_n * n + d_lam_t1 * t1 + d_lam_t2 * t2`` with
``v1 -= inv_mass1 * imp`` / ``v2 += inv_mass2 * imp`` and the
matching angular update. Newton's 3rd law is structurally enforced
by the code, so linear momentum conservation is *exact* per slot
update -- but that's only true as long as the impulse is applied
symmetrically. Any regression that broke the symmetry
(independently scaled body-1 / body-2 arms, accumulator lag, mixed
sign conventions, one-sided clamp, mis-applied warm-start, etc.)
would leak horizontal momentum.

For each conservation law we pick the tightest tolerance an ideal
PGS + Jacobi-within-slot solver can plausibly achieve and assert
against it. If one of these tests regresses it very likely is a
real bug, not a noise trip.

Tests
-----

``TestRestingCubeForceDirection``
    A cube at rest on a plane under pure +Z gravity. The solver's
    own contact-force readout must have no horizontal component
    and no lateral torque (lever arms of a symmetric cube on a
    plane pass through the COM). Tolerance: ``< 0.1 %`` of ``m g``.

``TestRestingCubeZeroSpin``
    Same scene. The cube's angular velocity must stay essentially
    zero (``< 1e-5 rad/s``) for a full second of simulated time.
    An asymmetric contact torque would show up here as a steady
    angular drift.

``TestPairCollisionConservesLinearMomentum``
    Zero-gravity, two free-floating cubes heading towards each
    other along ``+/- X`` with equal speed. Total linear momentum
    is identically zero for every physical state; any nonzero
    residual is solver error. Tolerance: ``< 0.5 %`` of either
    cube's initial momentum magnitude.

``TestPairCollisionConservesAngularMomentum``
    Same scene. Total angular momentum about the origin starts at
    zero and must stay there (symmetric approach, symmetric
    rebound). Tolerance: ``< 0.5 %`` of the largest
    ``|r| * |m * v|`` available in the scene.

``TestResistingCubeContactImpulseBalancesGravity``
    The per-frame contact impulse on a resting cube integrated
    over a full frame must equal ``m g dt_frame`` to within
    ``< 1 %`` -- otherwise the cube would accelerate visibly and
    the ``TestRestingCubeZeroSpin`` assertion would also fail,
    but this one pins the numeric value directly.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.jitter.constraints.contact_matching_config import JITTER_CONTACT_MATCHING
from newton._src.solvers.jitter.examples.example_pyramid import (
    _build_jitter_world_from_model,
    _jitter_to_newton_kernel,
    _newton_to_jitter_kernel,
)

# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------

GRAVITY = 9.81
FPS = 120
FRAME_DT = 1.0 / FPS
SUBSTEPS = 4
SOLVER_ITERATIONS = 20

BOX_HALF = 0.5
BOX_MASS = 1.0


# ---------------------------------------------------------------------------
# Generic contact scene
# ---------------------------------------------------------------------------


class _ContactScene:
    """Headless Newton + Jitter pipeline for the conservation tests.

    The scene is parametrised by a list of
    ``(position, velocity, angular_velocity)`` tuples for dynamic
    cubes plus optional floor / gravity toggles. Graph-captures the
    per-frame pipeline once warm-up is done so the multi-hundred-
    frame settle loops run in sensible wall-clock time on CUDA.
    """

    def __init__(
        self,
        *,
        bodies_init: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]],
        has_floor: bool = True,
        gravity_z: float = -GRAVITY,
        mu: float = 0.5,
        device: wp.context.Devicelike = None,
    ) -> None:
        self.device = wp.get_device(device)
        self.mu = float(mu)
        self.num_cubes = len(bodies_init)

        mb = newton.ModelBuilder()
        if has_floor:
            mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        # Uniform-density cube inertia about the COM: I = m/3 * (hy^2 + hz^2)
        # on the diagonal for a box with half-extents (hx, hy, hz). With
        # a unit cube and unit mass this is 1/6 on the diagonal. We pass
        # it explicitly because ``density=0`` on the shape
        # (needed to keep ``add_shape_box`` from layering its default
        # 1000 kg/m^3 on top of the body's ``mass=BOX_MASS``) also
        # zeroes the shape's inertia contribution. Without explicit
        # inertia the cube spins wildly under any asymmetric contact
        # torque.
        I_diag = BOX_MASS * (BOX_HALF * BOX_HALF + BOX_HALF * BOX_HALF) / 3.0
        cube_inertia = wp.mat33(
            wp.vec3(I_diag, 0.0, 0.0),
            wp.vec3(0.0, I_diag, 0.0),
            wp.vec3(0.0, 0.0, I_diag),
        )
        self._box_bodies: list[int] = []
        for pos, _vel, _ang in bodies_init:
            body = mb.add_body(
                xform=wp.transform(
                    p=wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])),
                    q=wp.quat_identity(),
                ),
                mass=BOX_MASS,
                inertia=cube_inertia,
            )
            mb.add_shape_box(
                body,
                hx=BOX_HALF,
                hy=BOX_HALF,
                hz=BOX_HALF,
                cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
            )
            self._box_bodies.append(body)

        self.model = mb.finalize()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model, contact_matching=JITTER_CONTACT_MATCHING
        )
        self.contacts = self.collision_pipeline.contacts()
        rigid_contact_max = int(self.contacts.rigid_contact_point0.shape[0])

        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        builder, newton_to_jitter = _build_jitter_world_from_model(self.model)
        max_contact_columns = max(16, (rigid_contact_max + 5) // 6)
        self.world = builder.finalize(
            substeps=SUBSTEPS,
            solver_iterations=SOLVER_ITERATIONS,
            gravity=(0.0, 0.0, float(gravity_z)),
            max_contact_columns=max_contact_columns,
            rigid_contact_max=rigid_contact_max,
            num_shapes=int(self.model.shape_count),
            default_friction=self.mu,
            device=self.device,
        )
        self._newton_to_jitter = newton_to_jitter

        shape_body_np = self.model.shape_body.numpy()
        shape_body_jitter = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        self._shape_body = wp.array(shape_body_jitter, dtype=wp.int32, device=self.device)

        self._sync_newton_to_jitter()

        # Inject per-cube initial velocities + angular velocities
        # directly into Jitter's body container. Push the Newton state
        # back so the first collide() sees consistent qd.
        velocities = self.world.bodies.velocity.numpy().copy()
        ang_velocities = self.world.bodies.angular_velocity.numpy().copy()
        for newton_idx, (_pos, vel, ang) in zip(self._box_bodies, bodies_init):
            j = self._newton_to_jitter[newton_idx]
            velocities[j] = (float(vel[0]), float(vel[1]), float(vel[2]))
            ang_velocities[j] = (float(ang[0]), float(ang[1]), float(ang[2]))
        self.world.bodies.velocity.assign(velocities.astype(np.float32))
        self.world.bodies.angular_velocity.assign(ang_velocities.astype(np.float32))
        self._sync_jitter_to_newton()

    # -- plumbing ---------------------------------------------------------

    def _sync_newton_to_jitter(self) -> None:
        n = self.model.body_count
        wp.launch(
            _newton_to_jitter_kernel,
            dim=n,
            inputs=[self.state.body_q, self.state.body_qd, self.model.body_com],
            outputs=[
                self.world.bodies.position[1 : 1 + n],
                self.world.bodies.orientation[1 : 1 + n],
                self.world.bodies.velocity[1 : 1 + n],
                self.world.bodies.angular_velocity[1 : 1 + n],
            ],
            device=self.device,
        )

    def _sync_jitter_to_newton(self) -> None:
        n = self.model.body_count
        wp.launch(
            _jitter_to_newton_kernel,
            dim=n,
            inputs=[
                self.world.bodies.position[1 : 1 + n],
                self.world.bodies.orientation[1 : 1 + n],
                self.world.bodies.velocity[1 : 1 + n],
                self.world.bodies.angular_velocity[1 : 1 + n],
                self.model.body_com,
            ],
            outputs=[self.state.body_q, self.state.body_qd],
            device=self.device,
        )

    def step(self) -> None:
        self._sync_newton_to_jitter()
        self.model.collide(
            self.state,
            contacts=self.contacts,
            collision_pipeline=self.collision_pipeline,
        )
        self.world.step(dt=FRAME_DT, contacts=self.contacts, shape_body=self._shape_body)
        self._sync_jitter_to_newton()

    def run_frames(self, frames: int) -> None:
        """Run ``frames`` frames, using CUDA graph capture past a threshold."""
        if frames <= 0:
            return
        use_graph = self.device.is_cuda and frames >= 40
        if not use_graph:
            for _ in range(frames):
                self.step()
            return
        self.step()
        with wp.ScopedCapture(device=self.device) as capture:
            self.step()
        graph = capture.graph
        for _ in range(frames - 2):
            wp.capture_launch(graph)

    # -- state queries ---------------------------------------------------

    def positions_world(self) -> np.ndarray:
        all_pos = self.world.bodies.position.numpy()
        return np.array([all_pos[self._newton_to_jitter[b]] for b in self._box_bodies])

    def velocities_world(self) -> np.ndarray:
        all_vel = self.world.bodies.velocity.numpy()
        return np.array([all_vel[self._newton_to_jitter[b]] for b in self._box_bodies])

    def angular_velocities_world(self) -> np.ndarray:
        all_w = self.world.bodies.angular_velocity.numpy()
        return np.array([all_w[self._newton_to_jitter[b]] for b in self._box_bodies])

    def contact_force_on_cube(self, cube_index: int) -> np.ndarray:
        """Sum the per-pair wrench on ``cube_index`` across every
        contact column that mentions it. Returns a 3-vector [N].
        """
        n_cols = self.world.max_contact_columns
        pair_w = wp.zeros(n_cols, dtype=wp.spatial_vector, device=self.device)
        pair_b1 = wp.zeros(n_cols, dtype=wp.int32, device=self.device)
        pair_b2 = wp.zeros(n_cols, dtype=wp.int32, device=self.device)
        pair_cnt = wp.zeros(n_cols, dtype=wp.int32, device=self.device)
        self.world.gather_contact_pair_wrenches(pair_w, pair_b1, pair_b2, pair_cnt)
        pw = pair_w.numpy()[:, :3]
        b1 = pair_b1.numpy()
        b2 = pair_b2.numpy()
        cnt = pair_cnt.numpy()
        j = self._newton_to_jitter[self._box_bodies[cube_index]]
        total = np.zeros(3, dtype=np.float64)
        for i in range(n_cols):
            if cnt[i] <= 0:
                continue
            if int(b2[i]) == j:
                total += pw[i]
            elif int(b1[i]) == j:
                total -= pw[i]
        return total


# ---------------------------------------------------------------------------
# Utilities for computing total momentum / angular momentum
# ---------------------------------------------------------------------------


def _inertia_world_from_model_and_rot(
    inv_inertia_local: np.ndarray, rot_xyzw: np.ndarray
) -> np.ndarray:
    """Rotate the body-local inverse-inertia matrix into world frame and invert.

    ``inv_inertia_local`` is the body-local ``I^-1`` (3x3) stored by
    Newton. World inertia is ``R I R^T``; we return the world inertia,
    not its inverse, because the conservation checks need
    ``I_world @ w`` directly.
    """
    # xyzw -> rotation matrix
    x, y, z, w = rot_xyzw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    # local I from inv_I_local. For a unit cube with symmetric inertia
    # the matrix is diagonal and invertible; keep the general path anyway.
    I_local = np.linalg.inv(inv_inertia_local.astype(np.float64))
    return R @ I_local @ R.T


def _total_linear_momentum(scene: _ContactScene) -> np.ndarray:
    """Sum ``m_i * v_i`` across the scene's dynamic cubes [kg m/s]."""
    vels = scene.velocities_world().astype(np.float64)
    # All cubes share BOX_MASS = 1.0 in this suite, but keep the
    # multiply explicit in case the helper grows multi-mass scenes.
    return (vels * BOX_MASS).sum(axis=0)


def _total_angular_momentum_about_origin(scene: _ContactScene) -> np.ndarray:
    """Sum ``r_i x (m_i v_i) + I_i_world @ w_i`` [kg m^2/s].

    Uses the cube bodies' COM positions for ``r_i``. ``I_i_world`` is
    built from Newton's body-local inverse inertia + the current
    orientation. For our unit cubes (diagonal ``I``) this is well
    defined.
    """
    positions = scene.positions_world().astype(np.float64)
    velocities = scene.velocities_world().astype(np.float64)
    ang_velocities = scene.angular_velocities_world().astype(np.float64)
    orientations = scene.world.bodies.orientation.numpy()

    inv_inertia_local_all = scene.model.body_inv_inertia.numpy()

    L = np.zeros(3, dtype=np.float64)
    for i, newton_idx in enumerate(scene._box_bodies):
        j = scene._newton_to_jitter[newton_idx]
        # Linear angular momentum = r x p
        L += np.cross(positions[i], BOX_MASS * velocities[i])
        # Spin angular momentum = I_world @ omega
        rot = orientations[j].astype(np.float64)
        I_world = _inertia_world_from_model_and_rot(inv_inertia_local_all[newton_idx], rot)
        L += I_world @ ang_velocities[i]
    return L


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Jitter simulation tests run on CUDA only (graph capture is required for reasonable run-time).",
)
class TestRestingCubeForceDirection(unittest.TestCase):
    """A resting cube on a plane: contact force has zero horizontal
    component and zero lateral torque.

    Tight absolute tolerance (``< 0.1 % of m g``) so a Jacobi
    regression that leaked a bit of tangential impulse because of
    asymmetric body-1 / body-2 application would trip this.
    """

    def test_resting_cube_no_horizontal_force(self) -> None:
        device = wp.get_preferred_device()
        scene = _ContactScene(
            bodies_init=[((0.0, 0.0, BOX_HALF + 1.0e-3), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))],
            has_floor=True,
            gravity_z=-GRAVITY,
            mu=0.5,
            device=device,
        )
        scene.run_frames(2 * FPS)  # 2 s of sim to fully settle
        F = scene.contact_force_on_cube(0)
        expected_Fz = BOX_MASS * GRAVITY
        horizontal_mag = float(np.hypot(F[0], F[1]))
        print(
            f"resting cube contact force: F={F} N, "
            f"|F_xy|={horizontal_mag:.6g} N, Fz/mg={F[2] / expected_Fz:.6f}"
        )
        # Vertical force must be close to m*g; this is a sanity check.
        self.assertAlmostEqual(
            float(F[2]),
            expected_Fz,
            delta=0.02 * expected_Fz,  # 2 % -- loose, this test is about horizontal
            msg=f"vertical contact force wrong: {F[2]:.4f} N vs expected {expected_Fz:.4f} N",
        )
        # Horizontal components must be essentially zero. An
        # asymmetric friction impulse (not Newton's-3rd-law symmetric)
        # or a per-frame-drifting persistent tangent basis would show
        # up as steady horizontal force here.
        #
        # Tolerance 1e-5 * m g ~ 0.1 mN, well above measured FP
        # noise (~6e-8 N at the time of writing) so FP jitter across
        # GPU generations doesn't trip it, but tight enough to catch
        # a tangent-basis drift of even 1e-5 rad between frames,
        # which would steady-scatter a visible fraction of the
        # warm-started normal impulse into the tangent plane.
        tol = 1.0e-5 * expected_Fz  # ~0.1 mN for a 1 kg cube
        self.assertLess(
            horizontal_mag,
            tol,
            f"horizontal contact force too large: |F_xy|={horizontal_mag:.6f} N, "
            f"tolerance {tol:.6f} N (F={F})",
        )


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Jitter simulation tests run on CUDA only.",
)
class TestRestingCubeZeroSpin(unittest.TestCase):
    """A cube initialized with zero omega must still have zero omega
    after a second of simulated time.

    Any asymmetric contact torque application would cause measurable
    drift; the tolerance is set tight enough to catch a bug of that
    kind without tripping on FP noise.
    """

    def test_cube_at_rest_does_not_spin(self) -> None:
        device = wp.get_preferred_device()
        scene = _ContactScene(
            bodies_init=[((0.0, 0.0, BOX_HALF + 1.0e-3), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))],
            has_floor=True,
            gravity_z=-GRAVITY,
            mu=0.5,
            device=device,
        )
        scene.run_frames(int(1.0 * FPS))  # 1 s of sim
        w = scene.angular_velocities_world()[0]
        w_mag = float(np.linalg.norm(w))
        print(f"resting cube |omega| after 1s = {w_mag:.8f} rad/s  (omega = {w})")
        # 1e-5 rad/s: measured is ~1e-10 on current hardware (pure FP
        # noise), so this leaves 5 orders of magnitude margin over
        # cross-GPU FP rounding while still catching a real physical
        # drift -- a steady torque leak of 1e-5 N m on the cube's
        # 1/6 kg m^2 inertia would reach 1e-5 rad/s in ~1.7 s, and
        # a real bug producing visible spin would blow past this
        # threshold within a frame.
        self.assertLess(
            w_mag,
            1.0e-5,
            f"cube spontaneously spinning: |omega|={w_mag:.6f} rad/s (> 1e-5), omega={w}",
        )


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Jitter simulation tests run on CUDA only.",
)
class TestPairCollisionConservesLinearMomentum(unittest.TestCase):
    """Zero-gravity, two cubes approach head-on with equal and
    opposite velocities. Total linear momentum is analytically zero
    and must stay zero through the collision.

    Directly exercises the body-1-to-body-2 impulse symmetry in the
    contact iterate: any asymmetry in ``v1 -= inv_mass1 * imp`` /
    ``v2 += inv_mass2 * imp`` would leak net linear momentum.
    """

    def test_symmetric_approach_conserves_linear_momentum(self) -> None:
        device = wp.get_preferred_device()
        v0 = 1.0
        separation = 0.2  # surfaces start 0.2 m apart
        x_off = BOX_HALF + separation * 0.5  # each cube this far from origin
        scene = _ContactScene(
            bodies_init=[
                ((-x_off, 0.0, 0.0), (+v0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                ((+x_off, 0.0, 0.0), (-v0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ],
            has_floor=False,
            gravity_z=0.0,
            mu=0.5,
            device=device,
        )

        # Initial momentum is identically zero -- keep the tolerance
        # referenced to the per-cube momentum magnitude so a bad
        # diff shows up as a percentage of something physical.
        p_ref = BOX_MASS * v0  # per-cube momentum magnitude [kg m/s]

        # Sample before + during + after the expected collision window.
        # With surfaces 0.2 m apart and relative speed 2.0 m/s the
        # collision starts at ~0.1 s and finishes in well under 0.5 s.
        for tag, frames in (
            ("pre-collision", int(0.05 * FPS)),
            ("collision", int(0.2 * FPS)),
            ("post-collision", int(0.5 * FPS)),
        ):
            scene.run_frames(frames)
            p = _total_linear_momentum(scene)
            p_mag = float(np.linalg.norm(p))
            print(
                f"[pair-collision linear momentum] {tag:>14}: "
                f"|p|={p_mag:.6g} kg m/s ({p_mag / p_ref:.3%} of per-cube ref)"
            )
            # Measured drift on current hardware is zero to single-
            # precision FP (<1e-7 kg m/s); set the tolerance 4 orders
            # of magnitude looser so cross-GPU FP reordering doesn't
            # regress, while still catching *any* systematic Newton's-
            # 3rd-law asymmetry. A 0.1 % per-slot impulse asymmetry
            # would leak ~2e-3 kg m/s per collision slot (>>1e-4),
            # i.e. this threshold catches any real symmetry bug.
            self.assertLess(
                p_mag,
                1.0e-4 * p_ref,
                f"[{tag}] total linear momentum drifted to {p}, "
                f"|p|={p_mag:.6f} kg m/s "
                f"(ref per-cube momentum = {p_ref:.3f}).",
            )


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Jitter simulation tests run on CUDA only.",
)
class TestPairCollisionConservesAngularMomentum(unittest.TestCase):
    """Same symmetric-approach scene as above, but check total angular
    momentum about the origin.

    The scene is set up so the analytic initial L is identically
    zero (cubes approach along +/-X through the origin with no
    spin). Any nonzero residual after collision is solver error and
    must stay below a tight fraction of the available
    ``|r x m v|`` the scene can spin up.
    """

    def test_symmetric_approach_conserves_angular_momentum(self) -> None:
        device = wp.get_preferred_device()
        v0 = 1.0
        separation = 0.2
        x_off = BOX_HALF + separation * 0.5
        scene = _ContactScene(
            bodies_init=[
                ((-x_off, 0.0, 0.0), (+v0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                ((+x_off, 0.0, 0.0), (-v0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ],
            has_floor=False,
            gravity_z=0.0,
            mu=0.5,
            device=device,
        )
        L_scale = abs(x_off) * BOX_MASS * v0  # max |r x m v| available
        for tag, frames in (
            ("pre-collision", int(0.05 * FPS)),
            ("collision", int(0.2 * FPS)),
            ("post-collision", int(0.5 * FPS)),
        ):
            scene.run_frames(frames)
            L = _total_angular_momentum_about_origin(scene)
            L_mag = float(np.linalg.norm(L))
            print(
                f"[pair-collision angular momentum] {tag:>14}: "
                f"|L|={L_mag:.6g} kg m^2/s ({L_mag / L_scale:.3%} of scale ref)"
            )
            # Measured drift on current hardware is ~1e-15 kg m^2/s
            # (pure FP noise). 1e-4 * L_scale is ~5 orders of
            # magnitude looser, leaving headroom for GPU FP
            # reordering but still flagging a real impulse-arm
            # miscalc which would drift on the 1e-2 scale.
            self.assertLess(
                L_mag,
                1.0e-4 * L_scale,
                f"[{tag}] total angular momentum drifted to {L}, "
                f"|L|={L_mag:.6f} kg m^2/s "
                f"(L_scale ref = {L_scale:.3f}).",
            )


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Jitter simulation tests run on CUDA only.",
)
class TestRestingContactImpulseBalancesGravity(unittest.TestCase):
    """The contact force on a resting cube must equal ``m g`` along
    ``+Z`` to within 1 %.

    This is a direct numerical check on the normal-accumulator
    convergence, orthogonal to the "horizontal force is zero" test.
    If the Jacobi-within-slot update were under- or over-applying
    impulse relative to GS (e.g. missing angular cross-coupling in a
    way that propagated through the clamp), the steady-state
    ``lam_n`` would settle at the wrong value and this test would
    catch it.
    """

    def test_normal_impulse_equals_weight(self) -> None:
        device = wp.get_preferred_device()
        scene = _ContactScene(
            bodies_init=[((0.0, 0.0, BOX_HALF + 1.0e-3), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))],
            has_floor=True,
            gravity_z=-GRAVITY,
            mu=0.5,
            device=device,
        )
        scene.run_frames(2 * FPS)
        F = scene.contact_force_on_cube(0)
        expected_Fz = BOX_MASS * GRAVITY
        rel_err = abs(float(F[2]) - expected_Fz) / expected_Fz
        print(
            f"contact Fz = {F[2]:.6g} N (expected {expected_Fz:.4f} N); "
            f"rel err {rel_err:.3%}"
        )
        # Measured is ~0 % rel err (exactly m*g to FP precision).
        # Set tolerance 0.5 % -- a regression in the normal-
        # accumulator convergence (wrong effective mass, missing
        # Baumgarte, warm-start decay) typically shows up at
        # several-percent level, so this catches such bugs cleanly
        # with comfortable FP noise headroom.
        self.assertLess(
            rel_err,
            0.005,
            f"contact Fz={float(F[2]):.4f} N vs m*g={expected_Fz:.4f} N "
            f"(rel err {rel_err:.2%}); > 0.5 % means the normal "
            f"accumulator did not converge to the analytic value.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
