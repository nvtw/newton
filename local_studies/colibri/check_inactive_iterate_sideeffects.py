"""Frozen signed-zero and stale-multiplier comparison for inactive iteration."""

import numpy as np
import warp as wp

from local_studies.colibri.inactive_joint_iterate import iterate, joint, original
from newton._src.solvers.phoenx.tests.test_block_joint_policy import make_model, make_solver


def kernel(fn):
    @wp.kernel(module="unique")
    def run(
        c: joint.ConstraintContainer, b: joint.BodyContainer, p: joint.ParticleContainer, s: joint.CopyStateContainer
    ):
        fn(c, 0, b, p, s, 2, 0, 100.0, 1.0, False)

    return run


def main():
    model = make_model(0.0)
    solver = make_solver(model, mass_splitting=False)
    state = model.state()
    solver.step(state, state, model.control(), None, 0.01)
    world = solver.world
    saved = world.constraints.multipliers.numpy()
    saved[int(joint._MUL_ACC_FRICTION) // 4, 0, int(joint._MUL_ACC_FRICTION) % 4] = 0.125
    data = world.constraints.data.numpy()
    data.view(np.int32)[int(joint._OFF_CLAMP), 0] = 0
    data[int(joint._OFF_FRICTION_COEFFICIENT), 0] = 0
    world.constraints.data.assign(data)
    for value in (-0.0, 0.125):
        outputs = []
        for fn in (original, iterate):
            zero = np.full((2, 3), value, np.float32)
            world.bodies.velocity.assign(zero)
            world.bodies.angular_velocity.assign(zero)
            world.constraints.multipliers.assign(saved)
            wp.launch(
                kernel(fn),
                dim=1,
                inputs=[world.constraints, world.bodies, joint.ParticleContainer(), world._copy_state],
                device=model.device,
            )
            outputs.append(
                [
                    world.bodies.velocity.numpy(),
                    world.bodies.angular_velocity.numpy(),
                    world.constraints.multipliers.numpy(),
                ]
            )
        for name, a, b in zip(("linear", "angular", "lambda"), *outputs, strict=False):
            print(
                name, "bitexact", np.array_equal(a.view(np.uint32), b.view(np.uint32)), "numeric", np.array_equal(a, b)
            )
            np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))


if __name__ == "__main__":
    main()
