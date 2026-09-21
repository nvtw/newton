"""Measure Colibri solver settings without editing the example."""

import argparse
import time

import warp as wp

import newton
from newton.examples.kamino.example_kamino_colibri import Example
from newton.viewer import ViewerNull

p = argparse.ArgumentParser()
p.add_argument("--contacts", type=int, default=256)
p.add_argument("--substeps", type=int, default=20)
p.add_argument("--frames", type=int, default=60)
p.add_argument("--dense", action="store_true")
p.add_argument("--iterations", type=int, default=16)
p.add_argument("--sweeps", type=int, default=2)
a = p.parse_args()
original_init = newton.solvers.SolverKamino.__init__


def init(self, model, *, config):
    config.collision_detector.max_contacts_per_world = a.contacts
    config.sparse_dynamics = not a.dense
    config.sparse_jacobian = not a.dense
    config.dvi.inequality_sweeps_per_iteration = a.sweeps
    config.dvi.max_alternating_iterations = a.iterations
    config.dvi.bilateral_solve_interval = a.iterations
    original_init(self, model, config=config)


newton.solvers.SolverKamino.__init__ = init
original_simulate = Example.simulate


def simulate(self):
    self.sim_substeps = a.substeps
    self.sim_dt = self.frame_dt / a.substeps
    original_simulate(self)


Example.simulate = simulate
e = Example(ViewerNull(), argparse.Namespace(body_count=36))
for _ in range(5):
    e.step()
wp.synchronize()
t = time.perf_counter()
for _ in range(a.frames):
    e.step()
wp.synchronize()
elapsed = time.perf_counter() - t
e.test_final()
print("BENCHMARK", vars(a), "seconds", elapsed, "fps", a.frames / elapsed, flush=True)
