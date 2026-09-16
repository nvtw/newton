"""Expose loss of a small internal hinge mode in dense FP32 body responses."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri.cached_component_response import CachedComponentData, allocate, cached_cross
from newton._src.solvers.phoenx.articulations.maximal_contact_response import MaximalContactResponseData
from newton._src.solvers.phoenx.articulations.maximal_projector import MaximalTreeProjectorData
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.tests.test_maximal_contact_mobility import evaluate


@wp.kernel
def evaluate_cached(
    tree: MaximalTreeProjectorData,
    response: MaximalContactResponseData,
    cache: CachedComponentData,
    point: wp.vec3f,
    shift: wp.vec3f,
    direction: wp.vec3f,
    result: wp.array[float],
):
    result[0] = cached_cross(tree, response, cache, 0, 1, point, point + shift, direction, direction)


def main():
    """Compare the new cache with the existing stable internal-contact response."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-accurate-cache", action="store_true")
    args = parser.parse_args()
    device = "cpu"
    shift = np.array([-0.031, -0.027, -0.04], dtype=np.float32)
    mapping = np.eye(6, dtype=np.float32)
    x, y, z = shift
    mapping[:3, 3:] = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
    rng = np.random.default_rng(13)
    factor = (
        rng.normal(size=(6, 6)).astype(np.float32) * np.array([10, 10, 10, 100, 100, 100], dtype=np.float32)[:, None]
    )
    root = factor @ factor.T
    motion = np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)
    child = mapping @ root @ mapping.T + np.outer(motion, motion) * 17
    tree = MaximalTreeProjectorData()
    tree.depth = wp.array([[0, 1]], dtype=wp.int32, device=device)
    tree.parent = wp.array([[-1, 0]], dtype=wp.int32, device=device)
    tree.motion = wp.array([[motion, motion]], dtype=wp.spatial_vectorf, device=device)
    tree.inverse_d = wp.array([[0, 17]], dtype=wp.float32, device=device)
    response = MaximalContactResponseData()
    response.body_articulation = wp.array([0, 0], dtype=wp.int32, device=device)
    response.body_lane = wp.array([0, 1], dtype=wp.int32, device=device)
    response.conditional_map = wp.array([[np.eye(6), mapping]], dtype=wp.spatial_matrixf, device=device)
    response.mobility = wp.array([[root, child]], dtype=wp.spatial_matrixf, device=device)
    bodies = BodyContainer()
    bodies.position = wp.array([[0, 0, 0], -shift], dtype=wp.vec3f, device=device)
    full = np.block([[root, root @ mapping.T], [mapping @ root, child]])
    cache = allocate(1, device)
    basis = np.empty((1, 12, 2, 6), dtype=np.float32)
    for source in range(12):
        basis[0, source] = full[:, source].reshape(2, 6)
    cache.velocity.assign(basis)
    result = wp.zeros(1, dtype=float, device=device)
    point = wp.vec3f(0.017, -0.023, 0.009)
    records = []
    for transverse in (0.0, 0.001, 1.0):
        direction = wp.vec3f(transverse, 0, 1)
        wp.launch(evaluate, dim=1, inputs=[tree, response, bodies, point, direction, result], device=device)
        stable = float(result.numpy()[0])
        wp.launch(
            evaluate_cached,
            dim=1,
            inputs=[tree, response, cache, point, wp.vec3f(*shift), direction, result],
            device=device,
        )
        cached = float(result.numpy()[0])
        expected = 17 * ((0.023 + 0.027) * transverse) ** 2
        tolerance = 1e-11 + expected * 1e-5
        assert abs(stable - expected) <= tolerance
        records.append(
            {
                "transverse": transverse,
                "expected": expected,
                "native_stable": stable,
                "cached": cached,
                "allowed_error": tolerance,
                "cache_passed": abs(cached - expected) <= tolerance,
            }
        )
    report = {
        "scope": "Synthetic native regression fixture; exact dense body blocks stored in FP32, not a live Colibri cache capture",
        "cases": records,
    }
    Path("/tmp/colibri_cached_internal_cancellation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if args.require_accurate_cache:
        assert all(row["cache_passed"] for row in records), (
            "Dense common-root cancellation loses physical hinge response"
        )


if __name__ == "__main__":
    main()
