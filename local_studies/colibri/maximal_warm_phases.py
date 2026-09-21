import sys

sys.stdout = open("/tmp/colibri_maximal_warm_phases.log", "w", buffering=1)
import argparse

import numpy as np
import warp as wp

from local_studies.colibri.phoenx_scene import Example
from newton.viewer import ViewerNull

wp.config.quiet = True
e = Example(
    ViewerNull(),
    argparse.Namespace(
        body_count=15,
        mode="maximal",
        layout="single_world",
        substeps=10,
        iterations=8,
        outer_substeps=2,
        source_contact_offsets=True,
        mesh_cylinders=True,
    ),
)
for prior in range(16):
    e.step()
    e.test_post_step()
    if prior % 5 == 0:
        print("PRIOR", prior, flush=True)
e.graph = None
w = e.solver.world
d = e.solver._direct_equality_system
rj = np.asarray(d.topology.row_joint)
rl = d.row_local.numpy()
js = d.joint_to_structural.numpy()
pa = e.model.joint_parent.numpy() + 1
ch = e.model.joint_child.numpy() + 1
mask = d.row_dynamic.numpy() == 0


def report(label):
    v = np.concatenate((w.bodies.velocity.numpy(), w.bodies.angular_velocity.numpy()), axis=1)
    j0 = d.row_wrench0.numpy()[js[rj], rl]
    j1 = d.row_wrench1.numpy()[js[rj], rl]
    residual = np.sum(j0 * v[pa[rj]], axis=1) + np.sum(j1 * v[ch[rj]], axis=1)
    bias = d.row_bias.numpy()[js[rj], rl]
    print(
        label,
        "speed",
        np.max(abs(v)),
        "Jv",
        np.max(abs(residual[mask])),
        "Jv+b",
        np.max(abs((residual + bias)[mask])),
        "row",
        np.argmax(abs(residual * mask)),
        flush=True,
    )


old_launch = wp.launch


def traced_launch(kernel, *args, **kwargs):
    warm = "warm_start_maximal" in kernel.key
    if warm:
        report("before-projected-warm")
    result = old_launch(kernel, *args, **kwargs)
    if warm:
        report("after-projected-warm")
    return result


wp.launch = traced_launch
old_contact = w._solve_maximal_articulated_contacts
old_equal = d.solve
old_integrate = w._integrate_positions


def contacts(**kwargs):
    report("before-contact" + str(kwargs))
    old_contact(**kwargs)
    report("after-contact")


def equal(**kwargs):
    report("before-equality" + str(kwargs))
    old_equal(**kwargs)
    report("after-equality" + str(kwargs))


w._solve_maximal_articulated_contacts = contacts
d.solve = equal


def integrate():
    report("before-integrate")
    old_integrate()
    report("after-integrate")


w._integrate_positions = integrate
for frame in range(4):
    print("FRAME", frame, flush=True)
    e.step()
