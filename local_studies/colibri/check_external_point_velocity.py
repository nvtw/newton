"""Compare actual FP32 point-vector helper with recorded native per-point Jv."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri.external_point_velocity import external_relative_point_velocity
from local_studies.colibri.external_point_velocity_compensated import (
    compensated_relative_point_velocity,
    project_compensated_point_velocity,
)


@wp.kernel
def evaluate(
    values: wp.array2d[wp.vec3f],
    endpoints: wp.array2d[int],
    motion: wp.spatial_vectorf,
    shift: wp.vec3f,
    root_body: int,
    child_body: int,
    output: wp.array2d[float],
):
    i = wp.tid()
    velocity = external_relative_point_velocity(
        values[i, 0],
        values[i, 1],
        values[i, 2],
        values[i, 3],
        values[i, 4],
        values[i, 5],
        values[i, 6],
        root_body,
        child_body,
        motion,
        shift,
        endpoints[i, 0],
        endpoints[i, 1],
        values[i, 7],
        values[i, 8],
    )
    for axis in range(3):
        output[i, axis] = wp.dot(velocity, values[i, 9 + axis])


@wp.kernel
def evaluate_compensated(
    values: wp.array2d[wp.vec3f],
    endpoints: wp.array2d[int],
    motion: wp.spatial_vectorf,
    shift: wp.vec3f,
    root_body: int,
    child_body: int,
    output: wp.array2d[float],
):
    i = wp.tid()
    velocity = compensated_relative_point_velocity(
        values[i, 0],
        values[i, 1],
        values[i, 2],
        values[i, 3],
        values[i, 4],
        values[i, 5],
        values[i, 6],
        root_body,
        child_body,
        motion,
        shift,
        endpoints[i, 0],
        endpoints[i, 1],
        values[i, 7],
        values[i, 8],
    )
    for axis in range(3):
        output[i, axis] = project_compensated_point_velocity(velocity, values[i, 9 + axis])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compensated", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="/tmp/colibri_external_point_velocity.json")
    args = parser.parse_args()
    if args.device == "cpu":
        wp.config.enable_cuda = False
    result = {}
    for phase in ("biased", "relax"):
        source = np.load("/tmp/colibri_point_sink_replay331.point_inputs_" + phase + ".npz")
        trace = np.load("/tmp/colibri_point_loaded_replay.point_trace_" + phase + ".npz")["trace"]
        headers = source["input5__data"].view(np.int32)
        root, child = map(int, source["input0__body_slot"][0, :2])
        v = source["input3__velocity"].copy()
        w = source["input3__angular_velocity"].copy()
        derived = source["input6__derived"]
        lambdas = source["input6__lambdas"]
        values, endpoints, reference, ids = [], [], [], []
        for col in source["input9"][: int(source["input10"][0])]:
            b0, b1, first, count = map(int, headers[[1, 2, 5, 6], col])
            assert (b0 in (root, child)) != (b1 in (root, child)), "Unsupported internal contact"
            for point in range(first, first + count):
                normal = lambdas[:3, point]
                tangent = lambdas[3:6, point]
                values.append(
                    np.stack(
                        [
                            v[root],
                            w[root],
                            w[child],
                            v[b0],
                            w[b0],
                            v[b1],
                            w[b1],
                            derived[9:12, point],
                            derived[12:15, point],
                            normal,
                            tangent,
                            np.cross(normal, tangent),
                        ]
                    )
                )
                endpoints.append((b0, b1))
                reference.append(trace[:3, point])
                ids.append(point)
                v[root] = trace[6:9, point]
                w[root] = trace[9:12, point]
                v[child] = trace[12:15, point]
                w[child] = trace[15:18, point]
        values = np.asarray(values, dtype=np.float32)
        reference = np.asarray(reference, dtype=np.float32)
        output = wp.zeros((len(values), 3), dtype=float, device=args.device)
        wp.launch(
            evaluate_compensated if args.compensated else evaluate,
            dim=len(values),
            inputs=[
                wp.array(values, dtype=wp.vec3f, device=args.device),
                wp.array(np.asarray(endpoints, dtype=np.int32), dtype=int, device=args.device),
                wp.spatial_vectorf(*source["input0__motion"][0, 1]),
                wp.vec3f(*source["input0__shift"][0, 1]),
                root,
                child,
                output,
            ],
            device=args.device,
        )
        actual = output.numpy().copy()
        error = np.abs(actual.astype(float) - reference.astype(float))
        result[phase] = {
            "points": len(values),
            "max_abs_error_m_s": float(error.max()),
            "max_native_Jv_m_s": float(np.abs(reference).max()),
            "max_error_by_direction_m_s": error.max(axis=0).tolist(),
            "worst_point": ids[int(np.unravel_index(error.argmax(), error.shape)[0])],
            "all_finite": bool(np.isfinite(actual).all()),
        }
        np.savez_compressed(
            Path(args.output).with_suffix("." + phase + ".npz"),
            actual=actual,
            reference=reference,
            values=values,
            point_ids=ids,
        )
        # Existing contact certificate scale is 1e-8 m/s. This tests the actual
        # recorded state, not a universal guarantee for arbitrary velocity scales.
        assert np.isfinite(actual).all() and error.max() < 1e-8, result[phase]
    result["scope"] = __doc__ + " Actual recorded states only; no arbitrary-scale cancellation guarantee."
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
