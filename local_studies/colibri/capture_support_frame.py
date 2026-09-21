"""Read-only last-substep support audit with an independent full-trajectory gate."""

import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from local_studies.colibri.check_native_velocity_iterations import validate_reference
from newton._src.solvers.phoenx.solver import SolverPhoenX


@wp.kernel(enable_backward=False)
def advance(counter: wp.array[wp.int32]):
    counter[0] = counter[0] + 1


@wp.kernel(enable_backward=False)
def record1(source: wp.array[Any], target: wp.array2d[Any], counter: wp.array[wp.int32]):
    i = wp.tid()
    target[counter[0] % 60, i] = source[i]


@wp.kernel(enable_backward=False)
def record2(source: wp.array2d[Any], target: wp.array3d[Any], counter: wp.array[wp.int32]):
    i, j = wp.tid()
    target[counter[0] % 60, i, j] = source[i, j]


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--phase-output", default="/tmp/colibri_support_phases330.npz")
parser.add_argument("--reference-trajectory", default="/tmp/colibri_prepare_integrated_cooperative330.npz")
args, remaining = parser.parse_known_args()
output = Path(remaining[remaining.index("--output") + 1]).with_suffix(".npz")
root = Path.cwd()
paths = sorted((root / "newton/_src/solvers/phoenx").rglob("*.py"))
paths = [p for p in paths if "tests" not in p.parts]
paths += sorted((root / "newton/_src/geometry").rglob("*.py"))
paths += [
    root / "newton/examples/phoenx/example_phoenx_colibri.py",
    root / "newton/examples/kamino/example_kamino_colibri.py",
]


def hashes():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


before_hash = hashes()
reference_hash = hashlib.sha256(Path(args.reference_trajectory).read_bytes()).hexdigest()
captured = {}
original = SolverPhoenX.__init__


def initialize(self, *pos, **kw):
    original(self, *pos, **kw)
    w = self.world
    assert w._color_group_data is not None and not w._colored_contact_rows
    cc = w._contact_container
    copies = w._copy_state
    b = w.constraints.bilateral
    arrays = {
        "position": w.bodies.position,
        "orientation": w.bodies.orientation,
        "velocity": w.bodies.velocity,
        "angular_velocity": w.bodies.angular_velocity,
        "inverse_mass": w.bodies.inverse_mass,
        "inverse_inertia": w.bodies.inverse_inertia_world,
        "headers": w._contact_cols.data,
        "column_count": w._ingest_scratch.num_contact_columns,
        "lambdas": cc.lambdas,
        "derived": cc.derived,
        "impulses": cc.impulses,
        "joint_data": w.constraints.data,
        "copy_count": copies.count_per_node,
        "copy_velocity": copies.velocity,
        "copy_angular_velocity": copies.angular_velocity,
        "copy_section_end": copies.section_end,
        "copy_partition_list": copies.partition_list,
        "row_partition": w._color_group_data["row_partition"],
        "row_color": w._color_group_data["row_color"],
        "color_starts": w._color_group_data["starts"],
        "color_ids": w._color_group_data["ids"],
        "num_colors": w._color_group_data["num_colors"],
    }
    for name in (
        "row_count",
        "row_indices",
        "structural_index",
        "row_local",
        "row_dynamic",
        "wrench0",
        "wrench1",
        "bias",
        "reference",
        "dynamic_mass",
        "accumulated",
    ):
        arrays["joint_" + name] = getattr(b, name)
    slots = {}
    counter = wp.array([-1], dtype=wp.int32, device=w.device)

    def record(label, full=True):
        names = list(arrays) if full else ["position", "orientation", "velocity", "angular_velocity", "inverse_inertia"]
        if label not in slots:
            slots[label] = {
                name: wp.empty((60, *arrays[name].shape), dtype=arrays[name].dtype, device=w.device) for name in names
            }
        for name, target in slots[label].items():
            source = arrays[name]
            wp.launch(
                record1 if source.ndim == 1 else record2, source.shape, [source, target, counter], device=w.device
            )

    old_sweep = w._color_group_sweep
    last_phase = [None]

    def sweep(head, idt, contact_container=None):
        phase = ("warm", "biased", "relax")[w._singleworld_kernels()[::2].index(head)]
        last_phase[0] = phase
        record(phase + "_before")
        old_sweep(head, idt, contact_container)
        record(phase + "_solved")

    w._color_group_sweep = sweep
    old_average = w._mass_splitting_average_and_broadcast

    def average(inv_dt=None):
        old_average(inv_dt)
        if last_phase[0] is not None:
            record(last_phase[0] + "_averaged")

    w._mass_splitting_average_and_broadcast = average

    def hook(name, label):
        old = getattr(w, name)

        def call():
            if label == "forces":
                wp.launch(advance, 1, [counter], device=w.device)
            record(label + "_before", False)
            old()
            record(label + "_after", False)

        setattr(w, name, call)

    hook("_integrate_forces_and_gravity", "forces")
    hook("_integrate_positions", "integration")
    captured.update(slots=slots, world=w, model=self.model, counter=counter)


SolverPhoenX.__init__ = initialize
sys.argv = [sys.argv[0], *remaining]
try:
    runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
finally:
    SolverPhoenX.__init__ = original
    if captured:
        w = captured["world"]
        model = captured["model"]
        data = {label + "." + name: a.numpy() for label, slot in captured["slots"].items() for name, a in slot.items()}
        data.update(
            body_mass=model.body_mass.numpy(),
            body_inertia=model.body_inertia.numpy(),
            body_com=model.body_com.numpy(),
            labels=np.array(["world", *model.body_label]),
            dt=np.array([w.substep_dt]),
            last_substep_counter=captured["counter"].numpy(),
            num_joints=np.array([w.num_joints]),
        )
        np.savez_compressed(args.phase_output, **data)
        assert hashes() == before_hash, "Production source changed"
        assert hashlib.sha256(Path(args.reference_trajectory).read_bytes()).hexdigest() == reference_hash
        gate = validate_reference(output, args.reference_trajectory, "trajectory")
        Path(args.phase_output).with_suffix(".capture.json").write_text(
            json.dumps(
                {
                    "gate": gate,
                    "source_hashes": before_hash,
                    "scope": "Final60substep ring phase snapshots; generic read-only copy kernels; all-frame byte gate required",
                },
                indent=2,
            )
        )
