"""Local live search-floor control using the clean public validation runner."""

import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path

import warp as wp

import newton.examples
from local_studies.colibri.probe_gear_search_floor import floor_search
from newton._src.sim.collide import compute_shape_velocities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--velocity-iterations", type=int, choices=(1, 4), required=True)
    parser.add_argument("--frames", type=int, default=330)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    floor = 0.002
    original_launch, original_run = wp.launch, newton.examples.run
    collision_source = Path("newton/_src/sim/collide.py")
    collision_hash = hashlib.sha256(collision_source.read_bytes()).hexdigest()
    report = {
        "local_only": True,
        "search_extension_floor_m_per_dynamic_shape": floor,
        "configured_extension_cap_m_per_shape": 0.005,
        "extra_search_floor_kernels_per_frame": 2,
        "scope": "Extra search work and candidate contacts; unchanged physical offsets and 30 substeps at120Hz.",
        "velocity_iterations": args.velocity_iterations,
        "peak_live_contacts": 0,
        "peak_contact_columns": 0,
        "peak_body_copies": 0,
        "peak_total_body_copies": 0,
    }

    def launch(*positional, **kwargs):
        result = original_launch(*positional, **kwargs)
        kernel = kwargs.get("kernel", positional[0] if positional else None)
        if kernel is compute_shape_velocities:
            inputs, outputs = kwargs["inputs"], kwargs["outputs"]
            assert float(inputs[10]) == 0.005 and floor <= float(inputs[10])
            original_launch(
                floor_search,
                inputs[3].shape[0],
                inputs=[inputs[3], inputs[8], outputs[2], floor],
                device=inputs[3].device,
            )
        return result

    def observed_run(example, cli_args):
        model = example.model
        physical = model.shape_gap.numpy().tobytes()
        margins = model.shape_margin.numpy().tobytes()
        original_check = example.test_post_step

        def check():
            world = example.solver.world
            count = int(world._contact_views.rigid_contact_count.numpy()[0])
            report["peak_live_contacts"] = max(report["peak_live_contacts"], count)
            report["peak_contact_columns"] = max(
                report["peak_contact_columns"], int(world._ingest_scratch.num_contact_columns.numpy()[0])
            )
            copies = world._copy_state.count_per_node.numpy()
            report["peak_body_copies"] = max(report["peak_body_copies"], int(copies.max()))
            report["peak_total_body_copies"] = max(report["peak_total_body_copies"], int(copies.sum()))
            original_check()

        example.test_post_step = check
        try:
            return original_run(example, cli_args)
        finally:
            report["physical_gaps_and_margins_byte_unchanged"] = (
                model.shape_gap.numpy().tobytes() == physical and model.shape_margin.numpy().tobytes() == margins
            )
            assert report["physical_gaps_and_margins_byte_unchanged"]

    wp.launch, newton.examples.run = launch, observed_run
    sys.argv = [
        sys.argv[0],
        "--velocity-iterations",
        str(args.velocity_iterations),
        "--frames",
        str(args.frames),
        "--output",
        args.output,
    ]
    try:
        runpy.run_module("local_studies.colibri.check_clean_velocity_iterations", run_name="__main__")
    finally:
        wp.launch, newton.examples.run = original_launch, original_run
        report["collision_source_unchanged"] = (
            hashlib.sha256(collision_source.read_bytes()).hexdigest() == collision_hash
        )
        Path(args.output + ".floor.json").write_text(json.dumps(report, indent=2))
        assert report["collision_source_unchanged"]


if __name__ == "__main__":
    main()
