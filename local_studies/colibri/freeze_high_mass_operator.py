"""Compare old/new projections on identical prepared, full-friction contact rows."""

import ast
import json
import types
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.benchmarks.bench_high_mass_ratio import SolverSetting, _build_scene
from newton._src.solvers.phoenx.constraints import constraint_contact_cloth as contact
from newton._src.solvers.phoenx.tests.test_rigid_normal_first import (
    BodyContainer,
    ContactColumnContainer,
    ContactContainer,
    ContactViews,
    CopyStateContainer,
    ParticleContainer,
    constraint_bodies_make,
    contact_get_body1,
    contact_get_body2,
)


def legacy_factory():
    path = Path("/tmp/newton-g1-head-baseline/newton/_src/solvers/phoenx/constraints/constraint_contact_cloth.py")
    source = path.read_text()
    node = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_make_contact_iterate_at"
    )
    text = ast.get_source_segment(source, node)
    target = Path("/tmp/high_mass_legacy_iterate_factory.py")
    target.write_text(text + "\n")
    module = types.ModuleType("high_mass_legacy_iterate_factory")
    module.__dict__.update(vars(contact))
    module.__dict__["__name__"] = "high_mass_legacy_iterate_factory"
    exec(compile(text, str(target), "exec"), module.__dict__)
    return module._make_contact_iterate_at


def make_sweep(factory):
    iterate = factory(cloth_support=False, has_mass_splitting=False, use_bias=False, has_soft_contact_pd=False)

    @wp.kernel(enable_backward=False)
    def sweep(
        columns: ContactColumnContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        body_count: wp.int32,
        cc: ContactContainer,
        contacts: ContactViews,
        copies: CopyStateContainer,
        column_count: wp.int32,
    ):
        for cid in range(column_count):
            pair = constraint_bodies_make(contact_get_body1(columns, cid), contact_get_body2(columns, cid))
            iterate(
                columns,
                cid,
                0,
                bodies,
                particles,
                body_count,
                pair,
                wp.float32(480.0),
                cc,
                contacts,
                copies,
                0,
                wp.float32(1.0),
            )

    return sweep


def main():
    scene, _, _ = _build_scene(mass_ratio=400, setting=SolverSetting(8, 8), velocity_iterations=1, sor_boost=1)
    for _ in range(240):
        scene.step()
    world = scene.world
    cc = world._contact_container
    v = world.bodies.velocity.numpy()
    w = world.bodies.angular_velocity.numpy()
    inv_m = world.bodies.inverse_mass.numpy()
    inv_i = world.bodies.inverse_inertia_world.numpy()
    inv_i = inv_i[:, np.array([[0, 3, 4], [3, 1, 5], [4, 5, 2]])]
    active = inv_m > 0

    def energy():
        vv = world.bodies.velocity.numpy()[active]
        ww = world.bodies.angular_velocity.numpy()[active]
        return float(
            0.5 * np.sum(vv * vv / inv_m[active, None])
            + 0.5 * np.einsum("bi,bij,bj", ww, np.linalg.inv(inv_i[active]), ww)
        )

    count = int(world._ingest_scratch.num_contact_columns.numpy()[0])
    original = cc.impulses.numpy()
    data = {
        "velocity": v,
        "angular_velocity": w,
        "inverse_mass": inv_m,
        "inverse_inertia": inv_i,
        "impulses": original,
        "derived": cc.derived.numpy(),
        "lambdas": cc.lambdas.numpy(),
        "headers": world._contact_cols.data.numpy(),
    }
    np.savez("/tmp/high_mass_frozen_operator.npz", **data)
    results = []
    for label, factory in [
        ("head_projection", legacy_factory()),
        ("current_projection", contact._make_contact_iterate_at),
    ]:
        world.bodies.velocity.assign(v)
        world.bodies.angular_velocity.assign(w)
        cc.impulses.zero_()
        before = energy()
        kernel = make_sweep(factory)
        for iteration in range(8):
            wp.launch(
                kernel,
                dim=1,
                inputs=[
                    world._contact_cols,
                    world.bodies,
                    world.particles or ParticleContainer(),
                    world.num_bodies,
                    cc,
                    world._active_contact_views(),
                    world._copy_state or CopyStateContainer(),
                    count,
                ],
                device=scene.device,
            )
            results.append(
                {
                    "variant": label,
                    "sweep": iteration + 1,
                    "energy_before": before,
                    "energy_after": energy(),
                    "max_impulse": float(np.max(np.abs(cc.impulses.numpy()))),
                }
            )
    Path("/tmp/high_mass_frozen_operator.json").write_text(
        json.dumps({"column_count": count, "results": results}, indent=2) + "\n"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
