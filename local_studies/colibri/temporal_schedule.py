"""Opt-in diagnostic: relax only after the final temporal substep."""

from functools import wraps

from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


def install_temporal_schedule():
    """Keep biased solves unchanged and defer velocity sweeps to update end."""
    original_step = PhoenXWorld.step
    original_forces = PhoenXWorld._integrate_forces_and_gravity

    @wraps(original_step)
    def step(world, *args, **kwargs):
        configured = world.velocity_iterations
        world._temporal_velocity_iterations = configured
        try:
            return original_step(world, *args, **kwargs)
        finally:
            world.velocity_iterations = configured

    @wraps(original_forces)
    def forces(world, *args, **kwargs):
        world.velocity_iterations = (
            world._temporal_velocity_iterations if world._current_substep_index == world.substeps - 1 else 0
        )
        return original_forces(world, *args, **kwargs)

    PhoenXWorld.step = step
    PhoenXWorld._integrate_forces_and_gravity = forces
