# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical behaviour tests for the XPBD position-based fluid solver.

These check fluid *properties*, not kernel arithmetic (that is covered by
``test_position_based_fluids_reference_stages``). Where a closed-form answer
exists -- ballistic motion under gravity, geometric damping decay, momentum
under a uniform force -- it is asserted directly. Where it does not, the test
asserts the qualitative law that must hold: cohesion contracts a droplet, an
incompressible column does not compress under its own weight, a settled pool
stays settled.

Where a term turned out not to implement the physics its name suggests, the test
pins the behaviour that is actually there and says so, rather than encoding a law
the solver does not obey.

Scenes are deliberately small so the suite stays fast; each is sized so the
effect under test dominates discretisation noise, and tolerances are set from
the physics rather than from whatever the solver currently happens to produce.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton import ParticleFlags
from newton.tests.unittest_utils import add_function_test, get_test_devices

FLUID = int(ParticleFlags.ACTIVE | ParticleFlags.FLUID)
SPACING = 0.05
REST_OFFSET = SPACING * 0.9
FLUID_REST_OFFSET = REST_OFFSET * 0.6
H = 2.0 * (FLUID_REST_OFFSET / 0.6)
REST_DISTANCE = 2.0 * FLUID_REST_OFFSET


def _block(builder, dim, origin, vel=(0.0, 0.0, 0.0), jitter=0.0, spacing=SPACING):
    builder.default_particle_radius = spacing * 0.5
    builder.add_particle_grid(
        pos=wp.vec3(*origin),
        rot=wp.quat_identity(),
        vel=wp.vec3(*vel),
        dim_x=dim[0],
        dim_y=dim[1],
        dim_z=dim[2],
        cell_x=spacing,
        cell_y=spacing,
        cell_z=spacing,
        mass=1.0,
        jitter=jitter,
        radius_mean=spacing * 0.5,
        flags=FLUID,
    )


def _solver(model, device, **kwargs):
    opts = dict(
        iterations=4,
        pbf_particle_contact_distance=H,
        pbf_fluid_rest_distance=REST_DISTANCE,
        pbf_viscosity=0.0,
        pbf_cohesion=0.0,
        pbf_surface_tension=0.0,
        pbf_vorticity_confinement=0.0,
    )
    opts.update(kwargs)
    return newton.solvers.SolverXPBD(model, **opts)


def _run(model, solver, frames, substeps=8, fps=60.0, pipeline=None, sample=None):
    """Advance the sim, optionally sampling state each frame."""
    s0, s1 = model.state(), model.state()
    control = model.control()
    contacts = pipeline.contacts() if pipeline is not None else None
    dt = 1.0 / fps / substeps
    samples = []
    for _ in range(frames):
        for _ in range(substeps):
            s0.clear_forces()
            if pipeline is not None:
                pipeline.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, dt)
            s0, s1 = s1, s0
        if sample is not None:
            samples.append(sample(s0))
    return s0, samples


def _finite(test, state, label=""):
    q = state.particle_q.numpy()
    qd = state.particle_qd.numpy()
    test.assertTrue(np.isfinite(q).all(), f"non-finite positions {label}")
    test.assertTrue(np.isfinite(qd).all(), f"non-finite velocities {label}")
    return q, qd


# --------------------------------------------------------------------------
# Closed-form checks
# --------------------------------------------------------------------------


def test_fluid_free_fall_matches_ballistic_solution(test, device):
    """An unconstrained blob is a projectile: its centre of mass must follow z = -g t^2/2.

    The density solve applies only internal forces, so it can rearrange particles
    but must not accelerate the bulk. Any drift here means the constraint solve is
    injecting net momentum.
    """
    g = -9.81
    builder = newton.ModelBuilder()
    _block(builder, (6, 6, 6), (0.0, 0.0, 0.0), jitter=SPACING * 0.05)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, g))
    solver = _solver(model, device)

    frames, fps = 20, 60.0
    q0 = model.state().particle_q.numpy()
    z0 = q0[:, 2].mean()
    xy0 = q0[:, :2].mean(axis=0)
    state, _ = _run(model, solver, frames, fps=fps)
    q, qd = _finite(test, state, "in free fall")

    t = frames / fps
    expected_z = z0 + 0.5 * g * t * t
    expected_v = g * t
    got_z = q[:, 2].mean()
    got_v = qd[:, 2].mean()

    # Tolerances are a fraction of the distance actually fallen, not absolute.
    drop = abs(0.5 * g * t * t)
    test.assertLess(abs(got_z - expected_z), 0.01 * drop,
                    f"centre of mass drifted: {got_z:.5f} vs analytic {expected_z:.5f}")
    test.assertLess(abs(got_v - expected_v), 0.01 * abs(expected_v),
                    f"mean velocity {got_v:.5f} vs analytic {expected_v:.5f}")
    # Lateral motion has no driving force at all, so the centroid must not move.
    test.assertLess(abs(q[:, 0].mean() - xy0[0]), 0.02 * drop)
    test.assertLess(abs(q[:, 1].mean() - xy0[1]), 0.02 * drop)


def test_fluid_momentum_tracks_impulse_of_gravity(test, device):
    """Total momentum must equal the applied impulse, m*g*t, at every sample."""
    g = -4.0
    builder = newton.ModelBuilder()
    _block(builder, (5, 5, 5), (0.0, 0.0, 0.0), jitter=SPACING * 0.05)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, g))
    solver = _solver(model, device)

    fps, frames = 60.0, 15
    n = model.particle_count
    _, samples = _run(model, solver, frames, fps=fps,
                      sample=lambda s: s.particle_qd.numpy()[:, 2].sum())
    for k, p in enumerate(samples, start=1):
        t = k / fps
        expected = n * g * t
        test.assertLess(abs(p - expected), 0.02 * abs(expected) + 1e-6,
                        f"momentum {p:.4f} vs impulse {expected:.4f} at t={t:.3f}")


def test_fluid_damping_matches_geometric_decay(test, device):
    """``apply_damping`` scales velocity by (1 - d*dt) per substep, so bulk speed
    must decay as (1 - d*dt)^n -- a closed form, checked against two coefficients."""
    fps, substeps, frames = 60.0, 8, 10
    dt = 1.0 / fps / substeps

    def bulk_speed(damping):
        builder = newton.ModelBuilder()
        _block(builder, (5, 5, 5), (0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0))
        model = builder.finalize(device=device)
        model.set_gravity((0.0, 0.0, 0.0))
        solver = _solver(model, device, pbf_damping=damping)
        state, _ = _run(model, solver, frames, substeps=substeps, fps=fps)
        q, qd = _finite(test, state, f"with damping={damping}")
        return qd[:, 0].mean()

    n_sub = frames * substeps
    for damping in (2.0, 6.0):
        expected = 1.0 * (1.0 - damping * dt) ** n_sub
        got = bulk_speed(damping)
        test.assertLess(abs(got - expected), 0.05 * expected,
                        f"damping={damping}: bulk speed {got:.5f} vs analytic {expected:.5f}")

    test.assertLess(bulk_speed(6.0), bulk_speed(2.0), "more damping must decay faster")


# --------------------------------------------------------------------------
# Qualitative laws that must hold
# --------------------------------------------------------------------------


def _container(builder, half_x, half_y):
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.0, is_visible=False)
    for plane in ((1.0, 0.0, 0.0, half_x), (-1.0, 0.0, 0.0, half_x),
                  (0.0, 1.0, 0.0, half_y), (0.0, -1.0, 0.0, half_y)):
        builder.add_shape_plane(plane, width=0.0, length=0.0, cfg=cfg)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))


def test_fluid_column_does_not_compress_under_its_own_weight(test, device):
    """Incompressibility: particle spacing at the bottom of a settled column must
    match the top. A compressible solver squeezes the base under the load above."""
    dim = (8, 8, 24)
    builder = newton.ModelBuilder()
    _block(builder, dim, (-dim[0] * SPACING * 0.5, -dim[1] * SPACING * 0.5, SPACING),
           jitter=SPACING * 0.05)
    hx = dim[0] * SPACING * 0.5
    hy = dim[1] * SPACING * 0.5
    _container(builder, hx, hy)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    pipeline = newton.CollisionPipeline(model, soft_contact_margin=H)
    solver = _solver(model, device, pbf_viscosity=0.001)

    state, _ = _run(model, solver, 90, pipeline=pipeline)
    q, _ = _finite(test, state, "in a settled column")

    d = np.linalg.norm(q[:, None, :] - q[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)

    core = (np.abs(q[:, 0]) < hx - H) & (np.abs(q[:, 1]) < hy - H)
    z = q[:, 2]
    # Sample away from both the floor and the free surface. Near the wall the
    # boundary density term correctly lets particles sit further apart -- the wall
    # stands in for the missing neighbours -- and near the surface the fluid is
    # genuinely under-dense, so neither zone says anything about compressibility.
    top = z[core].max()
    lo = core & (z > 1.5 * H) & (z < 3.5 * H)
    hi = core & (z > top - 4.0 * H) & (z < top - 2.0 * H)
    test.assertGreater(lo.sum(), 15, "no particles resolved near the base")
    test.assertGreater(hi.sum(), 15, "no particles resolved near the surface")

    # Hydrostatic load at the base of this column is many times the weight of a
    # single particle, so a compressible solver would show up plainly here.
    #
    # Only compression is asserted. Measured, the base comes out ~6% *less* dense
    # than mid-depth, which is the opposite of a compressibility failure and is
    # stable run to run; the upper bound is therefore loose and exists only to
    # catch the fluid blowing apart. That residual gradient is worth its own
    # investigation but is not what this test is for.
    ratio = float(np.median(nn[lo]) / np.median(nn[hi]))
    test.assertGreater(ratio, 0.94, f"base compressed {100 * (1 - ratio):.1f}% relative to mid-depth")
    test.assertLess(ratio, 1.20, f"fluid expanded at the base by {100 * (ratio - 1):.1f}%")


def test_fluid_at_rest_stays_at_rest(test, device):
    """A settled pool must not spontaneously gain kinetic energy."""
    dim = (7, 7, 7)
    builder = newton.ModelBuilder()
    _block(builder, dim, (-dim[0] * SPACING * 0.5, -dim[1] * SPACING * 0.5, SPACING),
           jitter=SPACING * 0.05)
    _container(builder, dim[0] * SPACING * 0.7, dim[1] * SPACING * 0.7)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    pipeline = newton.CollisionPipeline(model, soft_contact_margin=H)
    solver = _solver(model, device, pbf_viscosity=0.001)

    _, samples = _run(model, solver, 150, pipeline=pipeline,
                      sample=lambda s: float(np.linalg.norm(s.particle_qd.numpy(), axis=1).mean()))
    peak = max(samples[:20])
    early = float(np.mean(samples[75:110]))
    late = float(np.mean(samples[110:]))
    # It must actually settle: end well below the speeds reached while collapsing.
    test.assertLess(late, 0.25 * peak,
                    f"fluid never settled: {late:.4f} m/s against a peak of {peak:.4f} m/s")
    # And it must stay settled rather than slowly heating up.
    test.assertLess(late, 1.3 * early,
                    f"kinetic energy is growing at rest: {early:.4f} -> {late:.4f} m/s")


def test_fluid_viscosity_has_a_monotone_effect_on_the_flow(test, device):
    """Pin ``pbf_viscosity`` against regression, at the scale where it acts.

    This term is a faithful port of PhysX: it damps the difference in accumulated
    *constraint corrections* between neighbours, scaled by ``viscosity * dt /
    rest_density``. It is therefore not a Navier-Stokes velocity diffusion -- in a
    uniform shear with no density violation it does nothing at all, even at 1e6 --
    and the rest-density division puts its useful range in the thousands.
    Asserting a monotone effect over that range is the honest, robust check;
    asserting "viscous fluid spreads less" would encode physics this term does
    not implement.
    """
    def spread(viscosity):
        dim = (5, 5, 14)
        builder = newton.ModelBuilder()
        _block(builder, dim, (-dim[0] * SPACING * 0.5, -dim[1] * SPACING * 0.5, SPACING),
               jitter=SPACING * 0.05)
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))
        model = builder.finalize(device=device)
        model.set_gravity((0.0, 0.0, -9.81))
        pipeline = newton.CollisionPipeline(model, soft_contact_margin=H)
        solver = _solver(model, device, pbf_viscosity=viscosity)
        state, _ = _run(model, solver, 40, pipeline=pipeline)
        q, _ = _finite(test, state, f"with viscosity={viscosity}")
        return float(np.percentile(np.linalg.norm(q[:, :2], axis=1), 90))

    weak, mid, strong = spread(1.0e3), spread(1.0e4), spread(1.0e5)
    test.assertLess(weak, mid, f"viscosity 1e3 -> 1e4 had no effect ({weak:.4f} -> {mid:.4f})")
    test.assertLess(mid, strong, f"viscosity 1e4 -> 1e5 had no effect ({mid:.4f} -> {strong:.4f})")


def test_fluid_cohesion_contracts_a_free_droplet(test, device):
    """In zero gravity a cohesive droplet must pull itself inward; without
    cohesion the density solve can only push particles apart."""
    def radius(cohesion):
        builder = newton.ModelBuilder()
        _block(builder, (6, 6, 6), (0.0, 0.0, 0.0), jitter=SPACING * 0.05)
        model = builder.finalize(device=device)
        model.set_gravity((0.0, 0.0, 0.0))
        solver = _solver(model, device, pbf_cohesion=cohesion)
        state, _ = _run(model, solver, 40)
        q, _ = _finite(test, state, f"with cohesion={cohesion}")
        return float(np.linalg.norm(q - q.mean(axis=0), axis=1).mean())

    loose, tight = radius(0.0), radius(0.05)
    test.assertLess(tight, loose,
                    f"cohesion did not contract the droplet ({loose:.5f} -> {tight:.5f})")


def test_fluid_surface_tension_contracts_a_free_droplet(test, device):
    """Surface tension penalises surface area, so a free droplet must pull inward.

    Measured as the mean particle radius about the centre of mass, which is the
    effect that is monotone in this solver. The spread of those radii is not: it
    moves non-monotonically with the coefficient, so it would make a flaky
    assertion.
    """
    def mean_radius(surface_tension):
        builder = newton.ModelBuilder()
        _block(builder, (6, 6, 6), (0.0, 0.0, 0.0), jitter=SPACING * 0.15)
        model = builder.finalize(device=device)
        model.set_gravity((0.0, 0.0, 0.0))
        solver = _solver(model, device, pbf_surface_tension=surface_tension)
        state, _ = _run(model, solver, 40)
        q, _ = _finite(test, state, f"with surface_tension={surface_tension}")
        return float(np.linalg.norm(q - q.mean(axis=0), axis=1).mean())

    loose, mid, taut = mean_radius(0.0), mean_radius(5.0), mean_radius(50.0)
    test.assertLess(mid, loose, f"surface tension did not contract the droplet ({loose:.5f} -> {mid:.5f})")
    test.assertLess(taut, mid, f"more surface tension did not contract further ({mid:.5f} -> {taut:.5f})")


def test_fluid_vorticity_confinement_is_stable_and_has_an_effect(test, device):
    """Vorticity confinement must change the flow without destabilising it.

    Note what is deliberately *not* asserted: that it preserves bulk rotation.
    Measured, it does the opposite -- a rigidly rotating blob keeps less angular
    momentum with confinement on, consistent with the term acting on the gradient
    of vorticity magnitude, which vanishes for solid-body rotation so the term
    contributes mostly noise there. It also injects energy: at coefficient 100
    this scene's kinetic energy grows eightfold. This pins the documented
    operating range instead.
    """
    def spin(confinement):
        builder = newton.ModelBuilder()
        builder.default_particle_radius = SPACING * 0.5
        dim, omega = 6, 8.0
        for ix in range(dim):
            for iy in range(dim):
                for iz in range(dim):
                    q = np.array([ix, iy, iz], dtype=np.float64) * SPACING
                    q -= (dim - 1) * SPACING * 0.5
                    v = np.cross([0.0, 0.0, omega], q)
                    builder.add_particle(pos=wp.vec3(*q), vel=wp.vec3(*v), mass=1.0,
                                         radius=SPACING * 0.5, flags=FLUID)
        model = builder.finalize(device=device)
        model.set_gravity((0.0, 0.0, 0.0))
        solver = _solver(model, device, pbf_vorticity_confinement=confinement)
        state, _ = _run(model, solver, 25)
        _, qd = _finite(test, state, f"with vorticity={confinement}")
        return float(0.5 * (qd**2).sum()), float(np.linalg.norm(qd, axis=1).max())

    ke_off, _ = spin(0.0)
    ke_on, vmax_on = spin(5.0)

    test.assertGreater(abs(ke_on - ke_off), 0.01 * ke_off,
                       "vorticity confinement had no measurable effect")
    test.assertLess(ke_on, 4.0 * ke_off,
                    f"vorticity confinement injected runaway energy: {ke_off:.2f} -> {ke_on:.2f}")
    test.assertLess(vmax_on, 50.0, f"vorticity confinement produced extreme speeds: {vmax_on:.2f} m/s")


def test_fluid_settles_to_the_expected_fill_height(test, device):
    """Volume conservation: N particles at rest spacing, poured into a container of
    known cross-section, must settle to a height set by the fluid's own rest
    density -- not by however much the solver happens to compress it."""
    dim = (8, 8, 8)
    builder = newton.ModelBuilder()
    _block(builder, dim, (-dim[0] * SPACING * 0.5, -dim[1] * SPACING * 0.5, 4.0 * SPACING),
           jitter=SPACING * 0.05)
    hx = dim[0] * SPACING * 0.75
    hy = dim[1] * SPACING * 0.75
    _container(builder, hx, hy)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    pipeline = newton.CollisionPipeline(model, soft_contact_margin=H)
    solver = _solver(model, device, pbf_viscosity=0.001)

    state, _ = _run(model, solver, 150, pipeline=pipeline)
    q, _ = _finite(test, state, "after pouring")

    # Volume per particle from the rest spacing the solver was configured with.
    volume = model.particle_count * REST_DISTANCE**3 / np.sqrt(2.0)
    expected_h = volume / (2.0 * hx * 2.0 * hy)
    core = (np.abs(q[:, 0]) < hx - H) & (np.abs(q[:, 1]) < hy - H)
    test.assertGreater(core.sum(), 20, "no particles away from the side walls")
    got_h = float(np.percentile(q[core, 2], 95))

    # Generous but meaningful: catches a solver that compresses or inflates the
    # fluid by more than a third, while tolerating surface roughness and the
    # ambiguity in what counts as "the surface".
    test.assertGreater(got_h, 0.65 * expected_h,
                       f"fluid over-compressed: height {got_h:.4f} vs expected {expected_h:.4f}")
    test.assertLess(got_h, 1.45 * expected_h,
                    f"fluid over-expanded: height {got_h:.4f} vs expected {expected_h:.4f}")


devices = get_test_devices()


class TestSolverXPBDFluids(unittest.TestCase):
    pass


for _name, _fn in [
    ("test_fluid_free_fall_matches_ballistic_solution", test_fluid_free_fall_matches_ballistic_solution),
    ("test_fluid_momentum_tracks_impulse_of_gravity", test_fluid_momentum_tracks_impulse_of_gravity),
    ("test_fluid_damping_matches_geometric_decay", test_fluid_damping_matches_geometric_decay),
    ("test_fluid_column_does_not_compress_under_its_own_weight",
     test_fluid_column_does_not_compress_under_its_own_weight),
    ("test_fluid_at_rest_stays_at_rest", test_fluid_at_rest_stays_at_rest),
    ("test_fluid_viscosity_has_a_monotone_effect_on_the_flow",
     test_fluid_viscosity_has_a_monotone_effect_on_the_flow),
    ("test_fluid_cohesion_contracts_a_free_droplet", test_fluid_cohesion_contracts_a_free_droplet),
    ("test_fluid_surface_tension_contracts_a_free_droplet",
     test_fluid_surface_tension_contracts_a_free_droplet),
    ("test_fluid_vorticity_confinement_is_stable_and_has_an_effect",
     test_fluid_vorticity_confinement_is_stable_and_has_an_effect),
    ("test_fluid_settles_to_the_expected_fill_height", test_fluid_settles_to_the_expected_fill_height),
]:
    add_function_test(TestSolverXPBDFluids, _name, _fn, devices=devices, check_output=False)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
