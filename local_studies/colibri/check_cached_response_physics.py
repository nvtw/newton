"""Audit a captured FP32 response cache against the unchanged native CUDA solve."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri.cached_component_response import (
    CachedComponentData,
    generalized_cached_delta,
    serial_native_pair,
)
from newton._src.solvers.phoenx.articulations.maximal_contact_response import (
    MaximalContactResponseData,
    apply_maximal_contact_impulse_thread,
)
from newton._src.solvers.phoenx.articulations.maximal_projector import MaximalTreeProjectorData, _sync_tree


@wp.kernel(enable_backward=False)
def compare_responses(
    tree: MaximalTreeProjectorData,
    response: MaximalContactResponseData,
    cache: CachedComponentData,
    forces: wp.array2d[wp.spatial_vectorf],
    native: wp.array2d[wp.spatial_vectorf],
    cached: wp.array2d[wp.spatial_vectorf],
    native_joint: wp.array2d[float],
    cached_joint: wp.array2d[float],
    generalized: wp.bool,
    serial: wp.bool,
):
    lane = wp.tid()
    for trial in range(forces.shape[0]):
        if lane < 2:
            response.impulse[0, lane] = forces[trial, lane]
        _sync_tree()
        apply_maximal_contact_impulse_thread(0, lane, tree, response)
        if lane < 2:
            native[trial, lane] = response.velocity[0, lane]
            native_joint[trial, lane] = response.joint_velocity[0, lane]
            delta = wp.spatial_vectorf(0.0)
            joint = wp.float32(0.0)
            for source in range(2):
                for axis in range(6):
                    coefficient = forces[trial, source][axis]
                    delta += coefficient * cache.velocity[0, source * 6 + axis, lane]
                    joint += coefficient * cache.joint[0, source * 6 + axis, lane]
            if generalized:
                delta, joint = generalized_cached_delta(tree, response, cache, 0, lane)
            if serial:
                root, child, joint_speed = serial_native_pair(tree, response, 0)
                delta = root
                joint = wp.float32(0.0)
                if lane == 1:
                    delta = child
                    joint = joint_speed
            cached[trial, lane] = delta
            cached_joint[trial, lane] = joint
        _sync_tree()


def load_struct(kind, data, prefix, device):
    """Restore allocated struct arrays using their original declared Warp dtypes."""
    obj = kind()
    for name, variable in kind.vars.items():
        key = prefix + name
        if key in data:
            setattr(obj, name, wp.array(data[key], dtype=variable.type.dtype, device=device))
    return obj


def main():
    """Compare random external point impulses and check independent physical ledgers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="/tmp/colibri_cached_component_frozen1")
    parser.add_argument("--generalized", action="store_true")
    parser.add_argument("--serial", action="store_true")
    args = parser.parse_args()
    assert not (args.serial and args.generalized)
    data = np.load(args.prefix + ".cached_response.npz")
    direct = np.load(args.prefix + ".native_state.npz")
    device = "cuda:0"
    tree = load_struct(MaximalTreeProjectorData, data, "tree__", device)
    response = load_struct(MaximalContactResponseData, data, "response__", device)
    cache = CachedComponentData()
    cache.velocity = wp.array(data["velocity"], dtype=wp.spatial_vectorf, device=device)
    cache.joint = wp.array(data["joint"], dtype=float, device=device)
    cache.valid = wp.array(data["valid"], dtype=int, device=device)
    assert int(data["tree__body_count"][0]) == 2
    count = 128
    rng = np.random.default_rng(7726)
    forces = np.zeros((count, 2, 6), dtype=np.float32)
    for trial in range(count):
        body = trial % 2
        impulse = rng.normal(size=3) * 10 ** rng.uniform(-6, -3)
        lever = rng.normal(size=3) * 0.1
        forces[trial, body, :3] = impulse
        forces[trial, body, 3:] = np.cross(lever, impulse)
    f = wp.array(forces, dtype=wp.spatial_vectorf, device=device)
    outputs = [wp.zeros((count, 2), dtype=wp.spatial_vectorf, device=device) for _ in range(2)]
    joints = [wp.zeros((count, 2), dtype=float, device=device) for _ in range(2)]
    wp.launch(
        compare_responses,
        dim=64,
        block_dim=64,
        inputs=[tree, response, cache, f, *outputs, *joints, args.generalized, args.serial],
        device=device,
    )
    native, cached = (x.numpy().astype(float) for x in outputs)
    native_joint, cached_joint = (x.numpy().astype(float) for x in joints)
    h = data["velocity"][0].reshape(12, 12).T.astype(float)
    body_ids = data["tree__body_slot"][0, :2]
    positions = data["bodies__position"][body_ids].astype(float)
    mass = np.zeros((12, 12))
    for body in range(2):
        offset = 6 * body
        mass[offset : offset + 3, offset : offset + 3] = np.eye(3) * data["tree__body_mass"][0, body]
        s = data["tree__body_inertia"][0, body].astype(float)
        mass[offset + 3 : offset + 6, offset + 3 : offset + 6] = [
            [s[0], s[3], s[4]],
            [s[3], s[1], s[5]],
            [s[4], s[5], s[2]],
        ]
    j = np.concatenate([direct["direct_row_wrench0"][0], direct["direct_row_wrench1"][0]], axis=1).astype(float)
    dynamic = direct["direct_row_dynamic"].astype(bool)
    result = {
        "scope": "128 external point impulses, frozen Colibri factor; FP32 native/cache arithmetic, FP64 independent audit"
    }
    reference_scale = np.max(np.abs(native), axis=(1, 2))
    difference = np.max(np.abs(cached - native), axis=(1, 2))
    result["max_response_relative_error"] = float(np.max(difference / np.maximum(reference_scale, 1e-30)))
    result["max_joint_speed_difference"] = float(np.max(abs(native_joint - cached_joint)))
    result["matrix_relative_asymmetry"] = float(np.max(abs(h - h.T)) / np.max(abs(h)))
    result["symmetric_part_min_eigenvalue"] = float(np.min(np.linalg.eigvalsh(0.5 * (h + h.T))))
    for label, velocities in (("native", native), ("cache", cached)):
        dv = velocities.reshape(count, 12)
        reaction = dv @ mass.T - forces.reshape(count, 12)
        reaction = reaction.reshape(count, 2, 6)
        linear = reaction[:, :, :3].sum(axis=1)
        angular = (reaction[:, :, 3:] + np.cross(positions[None], reaction[:, :, :3])).sum(axis=1)
        hard = dv @ j[~dynamic].T
        work = np.einsum("ni,ni->n", forces.reshape(count, 12), dv)
        result[label] = {
            "linear_reaction_error_Ns": float(np.max(abs(linear))),
            "angular_reaction_error_Nms": float(np.max(abs(angular))),
            "hard_joint_delta_residual": float(np.max(abs(hard))),
            "minimum_response_work": float(np.min(work)),
        }
    result["generalized_reconstruction"] = args.generalized
    result["serial_native_algebra"] = args.serial
    suffix = (
        ".serial_native_physics"
        if args.serial
        else (".cache_generalized_physics" if args.generalized else ".cache_physics")
    )
    np.savez_compressed(
        args.prefix + suffix + ".npz",
        forces=forces,
        native=native,
        cached=cached,
        native_joint=native_joint,
        cached_joint=cached_joint,
        mass=mass,
        jacobian=j,
        dynamic=dynamic,
        positions=positions,
        mobility=h,
    )
    Path(args.prefix + suffix + ".json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    # Match the existing native FP32 response; no rank projection or symmetrizing the cache.
    assert result["max_response_relative_error"] < 2e-5
    assert result["matrix_relative_asymmetry"] < 2e-6
    assert result["cache"]["minimum_response_work"] >= 0
    for key in ("linear_reaction_error_Ns", "angular_reaction_error_Nms", "hard_joint_delta_residual"):
        assert result["cache"][key] <= 5 * result["native"][key] + 1e-10, (key, result)


if __name__ == "__main__":
    main()
