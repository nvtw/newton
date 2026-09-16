"""Read-only graph-compatible last pre/post relaxation capture for a component reference."""

import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.solver import SolverPhoenX

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--snapshot", type=Path, default=Path("/tmp/colibri_support_relax330.npz"))
options, remaining = parser.parse_known_args()
if "--output" in remaining:
    trajectory_path = Path(remaining[remaining.index("--output") + 1]).with_suffix(".npz")
    if options.snapshot.resolve() == trajectory_path.resolve():
        raise ValueError("Phase snapshot must differ from the trajectory snapshot")
sys.argv = [sys.argv[0], *remaining]
root = Path(__file__).resolve().parents[2]
source_paths = sorted((root / "newton/_src/solvers/phoenx").rglob("*.py"))
source_paths += [
    root / "newton/examples/phoenx/example_phoenx_colibri.py",
    root / "newton/examples/kamino/example_kamino_colibri.py",
]


def fingerprints():
    """Fingerprint source used by the captured solver and scene."""
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}


source_hashes = fingerprints()
captured = {}
original_init = SolverPhoenX.__init__


def initialize(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    w = self.world
    cc = w._contact_container
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
        "copy_count": w._copy_state.count_per_node,
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
    before = {name: wp.empty_like(array) for name, array in arrays.items()}
    after = {
        name: wp.empty_like(arrays[name]) for name in ("velocity", "angular_velocity", "impulses", "joint_accumulated")
    }
    copy_state = w._copy_state
    copies = {
        "velocity": wp.empty_like(copy_state.velocity),
        "angular_velocity": wp.empty_like(copy_state.angular_velocity),
    }
    old_average = w._mass_splitting_average_and_broadcast
    in_relaxation = False

    def average(inv_dt):
        if in_relaxation:
            wp.copy(copies["velocity"], copy_state.velocity)
            wp.copy(copies["angular_velocity"], copy_state.angular_velocity)
        old_average(inv_dt)

    w._mass_splitting_average_and_broadcast = average
    cls = type(w._dispatcher)
    old = cls.relax

    def relax(dispatcher, idt):
        nonlocal in_relaxation
        if dispatcher._world is w:
            for name, array in arrays.items():
                wp.copy(before[name], array)
        in_relaxation = dispatcher._world is w
        try:
            old(dispatcher, idt)
        finally:
            in_relaxation = False
        if dispatcher._world is w:
            for name, array in after.items():
                wp.copy(array, arrays[name])

    cls.relax = relax
    captured.update(before=before, after=after, world=w, copies=copies, dispatcher_class=cls, old_relax=old)


SolverPhoenX.__init__ = initialize
try:
    runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
finally:
    SolverPhoenX.__init__ = original_init
    if captured:
        w = captured["world"]
        captured["dispatcher_class"].relax = captured["old_relax"]
        if fingerprints() != source_hashes:
            raise RuntimeError("Solver or scene source changed during phase capture")
        options.snapshot.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            options.snapshot,
            **{name: array.numpy() for name, array in captured["before"].items()},
            **{"after_" + name: array.numpy() for name, array in captured["after"].items()},
            **{"pre_average_copy_" + name: array.numpy() for name, array in captured["copies"].items()},
            copy_section_end=w._copy_state.section_end.numpy(),
            copy_partition_list=w._copy_state.partition_list.numpy(),
            num_joints=np.array([w.num_joints]),
            dt=np.array([w.substep_dt]),
            joint_bounded_drive=w._direct_equality_system.row_bounded_drive.numpy(),
        )

        options.snapshot.with_suffix(".capture.json").write_text(
            json.dumps(
                {
                    "source_sha256": source_hashes,
                    "source_unchanged": True,
                    "arguments": remaining,
                    "warp_version": wp.__version__,
                    "scope": "Last before-relax physical state, pre-average copies and post-relax physical state from one run",
                },
                indent=2,
            )
            + "\n"
        )
