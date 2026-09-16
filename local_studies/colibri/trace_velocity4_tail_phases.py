"""Read-only native120Hz final-substep phase ring for the velocity4 tail failure."""

import argparse
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.solver import SolverPhoenX


@wp.kernel
def copy_matrix(counter: wp.array[wp.int32], phase: int, source: wp.array2d[float], output: wp.array3d[float]):
    i, j = wp.tid()
    output[4 * (counter[0] % 8) + phase, i, j] = source[i, j]


def vector_kernel(dtype):
    """Copy a typed device array into a ring without touching solver inputs."""

    @wp.kernel
    def copy_vector(counter: wp.array[wp.int32], phase: int, source: wp.array[dtype], output: wp.array2d[dtype]):
        i = wp.tid()
        output[4 * (counter[0] % 8) + phase, i] = source[i]

    return copy_vector


@wp.kernel
def advance(counter: wp.array[wp.int32], ids: wp.array[wp.int32]):
    step = counter[0]
    ids[step % 8] = step
    counter[0] = step + 1


def main():
    """Wrap actual solve/relax callbacks and preserve the full public trajectory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=4660)
    parser.add_argument("--output", default="/tmp/colibri_velocity4_native_tail")
    parser.add_argument("--reference", default="/tmp/colibri_native_velocity4_clean18000.npz")
    parser.add_argument("--reference-prefix-only", action="store_true")
    options = parser.parse_args()
    output = options.output
    trace = {}
    old_init = SolverPhoenX.__init__

    def initialize(self, model, *args, **kwargs):
        old_init(self, model, *args, **kwargs)
        w = self.world
        device = model.device
        counter = wp.zeros(1, dtype=wp.int32, device=device)
        ids = wp.full(8, -1, dtype=wp.int32, device=device)
        kernels = {}
        buffers = {}

        def allocate(name, array):
            # Allocate before the public Example starts CUDA graph capture.
            buffers[name] = wp.zeros((32, *array.shape), dtype=array.dtype, device=device)

        for name in (
            "position",
            "orientation",
            "velocity",
            "angular_velocity",
            "inverse_mass",
            "inverse_inertia_world",
        ):
            allocate("body_" + name, getattr(w.bodies, name))
        for name in ("lambdas", "derived", "impulses"):
            allocate("contact_" + name, getattr(w._contact_container, name))
        allocate("headers", w._contact_cols.data)
        allocate("column_count", w._ingest_scratch.num_contact_columns)
        allocate("copy_count", w._copy_state.count_per_node)
        allocate("joint_accumulated", w.constraints.bilateral.accumulated)
        for name, length, dtype in (
            ("generation_q", model.body_count, wp.transformf),
            ("generation_qd", model.body_count, wp.spatial_vectorf),
            ("post_q", model.body_count, wp.transformf),
            ("post_qd", model.body_count, wp.spatial_vectorf),
            ("rigid_contact_count", 1, wp.int32),
            ("rigid_contact_shape0", w.rigid_contact_max, wp.int32),
            ("rigid_contact_shape1", w.rigid_contact_max, wp.int32),
            ("rigid_contact_point0", w.rigid_contact_max, wp.vec3f),
            ("rigid_contact_point1", w.rigid_contact_max, wp.vec3f),
            ("rigid_contact_normal", w.rigid_contact_max, wp.vec3f),
            ("rigid_contact_margin0", w.rigid_contact_max, wp.float32),
            ("rigid_contact_margin1", w.rigid_contact_max, wp.float32),
        ):
            buffers[name] = wp.zeros((32, length), dtype=dtype, device=device)

        def capture(name, array, phase):
            assert name in buffers
            if array.ndim == 2:
                wp.launch(copy_matrix, array.shape, [counter, phase, array, buffers[name]], device=device)
            else:
                if array.dtype not in kernels:
                    kernels[array.dtype] = vector_kernel(array.dtype)
                wp.launch(kernels[array.dtype], array.shape, [counter, phase, array, buffers[name]], device=device)

        def phase_capture(phase):
            for name in (
                "position",
                "orientation",
                "velocity",
                "angular_velocity",
                "inverse_mass",
                "inverse_inertia_world",
            ):
                capture("body_" + name, getattr(w.bodies, name), phase)
            for name in ("lambdas", "derived", "impulses"):
                capture("contact_" + name, getattr(w._contact_container, name), phase)
            capture("headers", w._contact_cols.data, phase)
            capture("column_count", w._ingest_scratch.num_contact_columns, phase)
            capture("copy_count", w._copy_state.count_per_node, phase)
            capture("joint_accumulated", w.constraints.bilateral.accumulated, phase)

        cls = type(w._dispatcher)
        old_solve, old_relax = cls.solve, cls.relax

        def solve(dispatcher, idt):
            owned = dispatcher._world is w and w._current_substep_index == w.substeps - 1
            if owned:
                phase_capture(0)
            old_solve(dispatcher, idt)
            if owned:
                phase_capture(1)

        def relax(dispatcher, idt):
            owned = dispatcher._world is w and w._active_velocity_iterations > 0
            if owned:
                phase_capture(2)
            old_relax(dispatcher, idt)
            if owned:
                phase_capture(3)

        cls.solve, cls.relax = solve, relax
        old_step = self.step

        def step(state_in, state_out, control, contacts, dt):
            capture("generation_q", state_in.body_q, 0)
            capture("generation_qd", state_in.body_qd, 0)
            old_step(state_in, state_out, control, contacts, dt)
            capture("post_q", state_out.body_q, 3)
            capture("post_qd", state_out.body_qd, 3)
            views = w._contact_views
            for name in (
                "rigid_contact_count",
                "rigid_contact_shape0",
                "rigid_contact_shape1",
                "rigid_contact_point0",
                "rigid_contact_point1",
                "rigid_contact_normal",
                "rigid_contact_margin0",
                "rigid_contact_margin1",
            ):
                capture(name, getattr(views, name), 3)
            wp.launch(advance, 1, [counter, ids], device=device)

        self.step = step
        trace.update(
            buffers=buffers,
            counter=counter,
            ids=ids,
            model=model,
            dispatcher_class=cls,
            old_solve=old_solve,
            old_relax=old_relax,
        )

    SolverPhoenX.__init__ = initialize
    sys.argv = [sys.argv[0], "--velocity-iterations", "4", "--frames", str(options.frames), "--output", output]
    try:
        runpy.run_module("local_studies.colibri.check_clean_velocity_iterations", run_name="__main__")
    finally:
        SolverPhoenX.__init__ = old_init
        if trace:
            trace["dispatcher_class"].solve = trace["old_solve"]
            trace["dispatcher_class"].relax = trace["old_relax"]
            model = trace["model"]
            np.savez_compressed(
                output + ".trace.npz",
                **{name: array.numpy() for name, array in trace["buffers"].items()},
                step_ids=trace["ids"].numpy(),
                counter=trace["counter"].numpy(),
                body_labels=model.body_label,
                shape_labels=model.shape_label,
                shape_body=model.shape_body.numpy(),
                shape_gap=model.shape_gap.numpy(),
                shape_filter_pairs=np.array(sorted(model.shape_collision_filter_pairs), dtype=np.int32).reshape(-1, 2),
                body_mass=model.body_mass.numpy(),
                body_inertia=model.body_inertia.numpy(),
                body_com=model.body_com.numpy(),
            )
            baseline = np.load(options.reference)
            current = np.load(output + ".npz")
            keys = ("q", "qd", "q_history", "qd_history", "history_times")
            n = len(current["q_history"])
            if options.reference_prefix_only:
                prefix_count = min(n, len(baseline["q_history"]))
                if prefix_count == 0:
                    raise AssertionError("Trace comparison requires a nonempty reference prefix")
                matches = {
                    k + "_prefix": current[k][:prefix_count].tobytes() == baseline[k][:prefix_count].tobytes()
                    for k in ("q_history", "qd_history", "history_times")
                }
            elif n == len(baseline["q_history"]):
                matches = {k: current[k].tobytes() == baseline[k].tobytes() for k in keys}
            else:
                matches = {
                    k: current[k].tobytes() == baseline[k][:n].tobytes()
                    for k in ("q_history", "qd_history", "history_times")
                }
                matches["q"] = current["q"].tobytes() == baseline["q_history"][n - 1].tobytes()
                matches["qd"] = current["qd"].tobytes() == baseline["qd_history"][n - 1].tobytes()
            counts = trace["buffers"]["column_count"].numpy()[:, 0].reshape(8, 4)
            contacts = trace["buffers"]["rigid_contact_count"].numpy()[3::4, 0]
            valid_slots = trace["ids"].numpy() >= 0
            matches["all_ring_phase_entries_present"] = bool(np.all(counts[valid_slots] > 0))
            matches["all_ring_contact_entries_present"] = bool(np.all(contacts[valid_slots] > 0))
            metadata_path = Path(output + ".source.json")
            metadata = json.loads(metadata_path.read_text())
            metadata["phase_capture_hooks"] = True
            metadata["trace_reference"] = options.reference
            metadata["trace_reference_prefix_only"] = options.reference_prefix_only
            metadata["scope"] = "Read-only phase-instrumented stability run; timing includes ring capture"
            metadata_path.write_text(json.dumps(metadata, indent=2))
            Path(output + ".prefix.json").write_text(json.dumps(matches, indent=2))
            print("TRACE_PREFIX_GATE", matches, flush=True)
            if not all(matches.values()):
                raise AssertionError("Native phase trace changed the reference trajectory")


if __name__ == "__main__":
    main()
