"""Local geometric search-floor control; physical state and gaps stay fixed."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
from local_studies.colibri.analyze_late_contacts import gaps
from newton._src.sim.collide import compute_shape_velocities
from newton.examples.kamino.example_kamino_colibri import build_scene
from newton.examples.phoenx.example_phoenx_colibri import CONTACT_OFFSETS


@wp.kernel
def floor_search(body: wp.array[int], physical_gap: wp.array[float], search_gap: wp.array[float], floor: float):
    shape = wp.tid()
    if body[shape] >= 0:
        search_gap[shape] = wp.max(search_gap[shape], physical_gap[shape] + floor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="/tmp/colibri_spatial_priority_velocity4_trace18000")
    parser.add_argument("--output", default="/tmp/colibri_gear_search_floor.json")
    args = parser.parse_args()
    trace = np.load(args.prefix + ".trace.npz")
    slot = int(np.flatnonzero(trace["step_ids"] == 29839)[0])
    pre, post = trace["generation_q"][4 * slot], trace["post_q"][4 * slot + 3]
    qd = trace["generation_qd"][4 * slot]
    builder = build_scene(
        body_count=36,
        fix_base=False,
        contact_gap=0.001,
        source_contact_offsets=True,
        mesh_cylinders=True,
        sdf_resolution=0,
    )
    for i, label in enumerate(builder.shape_label):
        if label in CONTACT_OFFSETS:
            builder.shape_gap[i] = CONTACT_OFFSETS[label]
    model = builder.finalize(skip_validation_joints=True)
    body = model.shape_body.numpy()
    physical = model.shape_gap.numpy().copy()
    margins_before = model.shape_margin.numpy().copy()
    state = model.state()
    records = []
    capacity = 262144
    for floor in (0.0, 0.002):
        for reduced in (False, True):
            pipeline = newton.CollisionPipeline(
                model,
                rigid_contact_max=capacity,
                contact_matching="disabled",
                reduce_contacts=reduced,
                speculative_contact_gap_max=0.005,
                speculative_contact_velocity_filter=False,
            )
            contacts = pipeline.contacts()
            state.body_q.assign(pre)
            state.body_qd.assign(qd)
            original_launch = wp.launch
            intercepted = []

            def launch(
                *positional,
                original_launch=original_launch,
                pipeline=pipeline,
                floor=floor,
                intercepted=intercepted,
                **kwargs,
            ):
                result = original_launch(*positional, **kwargs)
                kernel = kwargs.get("kernel", positional[0] if positional else None)
                if kernel is compute_shape_velocities:
                    before = pipeline._shape_search_gap.numpy().copy()
                    if floor:
                        original_launch(
                            floor_search,
                            model.shape_count,
                            inputs=[model.shape_body, model.shape_gap, pipeline._shape_search_gap, floor],
                            device=model.device,
                        )
                    intercepted.append(before)
                return result

            wp.launch = launch
            try:
                pipeline.collide(state, contacts, dt=1 / 120)
            finally:
                wp.launch = original_launch
            assert len(intercepted) == 1
            assert state.body_q.numpy().tobytes() == pre.tobytes()
            assert state.body_qd.numpy().tobytes() == qd.tobytes()
            assert model.shape_gap.numpy().tobytes() == physical.tobytes()
            assert model.shape_margin.numpy().tobytes() == margins_before.tobytes()
            search = pipeline._shape_search_gap.numpy()
            expected = intercepted[0].copy()
            expected[body >= 0] = np.maximum(expected[body >= 0], physical[body >= 0] + np.float32(floor))
            np.testing.assert_array_equal(search, expected)
            total = int(contacts.rigid_contact_count.numpy()[0])
            assert total < capacity, (floor, reduced, total, capacity)
            shapes = np.column_stack(
                (contacts.rigid_contact_shape0.numpy()[:total], contacts.rigid_contact_shape1.numpy()[:total])
            )
            selected = np.flatnonzero(np.all(shapes == [46, 48], axis=1))
            shapes = shapes[selected]
            p0 = contacts.rigid_contact_point0.numpy()[selected]
            p1 = contacts.rigid_contact_point1.numpy()[selected]
            normals = contacts.rigid_contact_normal.numpy()[selected]
            margins = np.column_stack(
                (contacts.rigid_contact_margin0.numpy()[selected], contacts.rigid_contact_margin1.numpy()[selected])
            )
            before = gaps(pre, body, shapes, p0, p1, normals, margins)
            after = gaps(post, body, shapes, p0, p1, normals, margins)
            artifact = Path(args.output).with_suffix(f".floor{floor:.3f}_{'reduced' if reduced else 'raw'}.npz")
            np.savez(
                artifact,
                shapes=shapes,
                point0=p0,
                point1=p1,
                normals=normals,
                margins=margins,
                pre_gap=before,
                post_gap=after,
                shape_search_gap=search,
                shape_search_gap_before=intercepted[0],
                physical_gap=physical,
                aabb_lower=pipeline.narrow_phase.shape_aabb_lower.numpy(),
                aabb_upper=pipeline.narrow_phase.shape_aabb_upper.numpy(),
            )
            record = {
                "floor": floor,
                "reduced": reduced,
                "total": total,
                "capacity": capacity,
                "pair_count": len(selected),
                "pair_search_gap": search[[46, 48]].tolist(),
                "pre_min_gap": float(before.min()) if len(before) else None,
                "post_min_gap": float(after.min()) if len(after) else None,
                "post_below_500um": int(np.count_nonzero(after < -0.0005)),
                "state_physical_gaps_and_margins_byte_unchanged": True,
                "artifact": str(artifact),
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    # The normal geometric AABB expansion runs after our floor, so its extra
    # scalar padding must equal the added search extension per shape.
    for reduced in (False, True):
        base, trial = [np.load(r["artifact"]) for r in records if r["reduced"] == reduced]
        delta = trial["shape_search_gap"] - base["shape_search_gap"]
        np.testing.assert_allclose(
            base["aabb_lower"] - trial["aabb_lower"], delta[:, None] * np.ones((1, 3)), rtol=0, atol=2e-7
        )
        np.testing.assert_allclose(
            trial["aabb_upper"] - base["aabb_upper"], delta[:, None] * np.ones((1, 3)), rtol=0, atol=2e-7
        )
    Path(args.output).write_text(
        json.dumps({"status": "PASS", "aabb_floor_enclosed": True, "records": records}, indent=2)
    )


if __name__ == "__main__":
    main()
