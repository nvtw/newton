"""Run independent cancellation fixtures with the actual compensated device helper."""

import argparse
import unittest

import warp as wp

from local_studies.colibri import test_external_point_velocity_review as cases
from local_studies.colibri.external_point_velocity_compensated import (
    compensated_relative_point_velocity,
    project_compensated_point_velocity,
)


@wp.kernel(enable_backward=False)
def evaluate(values: wp.array2d[wp.vec3f], endpoints: wp.array2d[wp.int32], result: wp.array2d[wp.vec3f]):
    i = wp.tid()
    motion = wp.spatial_vectorf(
        values[i, 7][0], values[i, 7][1], values[i, 7][2], values[i, 8][0], values[i, 8][1], values[i, 8][2]
    )
    velocity = compensated_relative_point_velocity(
        values[i, 0],
        values[i, 1],
        values[i, 2],
        values[i, 3],
        values[i, 4],
        values[i, 5],
        values[i, 6],
        0,
        1,
        motion,
        values[i, 9],
        endpoints[i, 0],
        endpoints[i, 1],
        values[i, 10],
        values[i, 11],
    )
    actual = wp.vec3f(0.0)
    original = wp.vec3f(0.0)
    for axis in range(3):
        direction = wp.vec3f(0.0)
        direction[axis] = 1.0
        actual[axis] = project_compensated_point_velocity(velocity, direction)
        original[axis] = cases.row_velocity_local(
            values[i, 0],
            values[i, 1],
            wp.vec3f(0.0),
            values[i, 2],
            values[i, 3],
            values[i, 4],
            values[i, 5],
            values[i, 6],
            0,
            1,
            motion,
            values[i, 9],
            endpoints[i, 0],
            endpoints[i, 1],
            values[i, 10],
            values[i, 11],
            direction,
        )
    result[i, 0] = actual
    result[i, 1] = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device == "cpu":
        wp.config.enable_cuda = False

    def run_cases(values, endpoints):
        output = wp.zeros((len(values), 2), dtype=wp.vec3f, device=args.device)
        wp.launch(
            evaluate,
            len(values),
            inputs=[
                wp.array(values, dtype=wp.vec3f, device=args.device),
                wp.array(endpoints, dtype=wp.int32, device=args.device),
                output,
            ],
            device=args.device,
        )
        return output.numpy().copy()

    cases.run_cases = run_cases
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(cases.TestExternalPointVelocityReview)
    )
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
