"""Capture actual point-row inputs and compare a local cooperative candidate.

GPU execution requires an explicitly allocated shared-GPU window. Eager capture
must first match an uninstrumented controller trajectory; ablated or altered
boundary inputs are correctness tests only, never live physics evidence.
"""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri.g1_cooperative_point_rows import make_candidate
from newton.examples.robot.example_robot_policy import Example
from newton.viewer import ViewerNull


def clone(value, arrays):
    """Clone all reachable buffers while retaining shared array identity."""
    if isinstance(value, wp.array):
        key = (value.ptr, value.shape, value.strides, str(value.dtype))
        if key not in arrays:
            arrays[key] = wp.clone(value)
        return arrays[key]
    if hasattr(value, "_cls"):
        result = value._cls()
        for name in value._cls.vars:
            setattr(result, name, clone(getattr(value, name), arrays))
        return result
    return value


def equal(a, b, label):
    """Reject dtype, shape, signed-zero, or any other byte difference."""
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), label


def save_arrays(value, path, saved):
    """Save nested body/contact buffers as well as direct arguments."""
    if isinstance(value, wp.array):
        saved[path] = value.numpy()
    elif hasattr(value, "_cls"):
        for name in value._cls.vars:
            save_arrays(getattr(value, name), path + "." + name, saved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/g1_cooperative_point_rows.json"))
    parser.add_argument("--timing-repeats", type=int, default=0)
    parser.add_argument("--scalar-lane-values", action="store_true")
    args = parser.parse_args()
    assets = Path("/home/twidmer/.cache/newton/newton-assets_unitree_g1_2c175d66_f8fb7abc/unitree_g1")
    assert assets.is_dir()
    fingerprint_files = sorted(Path("newton").rglob("*.py")) + sorted(p for p in assets.rglob("*") if p.is_file())
    source_before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in fingerprint_files}
    wp.init()
    config = Example.create_parser().parse_args(["--robot", "g1_29dof", "--solver", "phoenx"])
    reference = Example(ViewerNull(), config)
    history = []
    for _ in range(21):
        reference.step()
        history.append((reference.state_0.body_q.numpy(), reference.state_0.body_qd.numpy()))
    old = np.load("/tmp/g1_cache_only_candidate.npz")
    equal(np.asarray([x[0] for x in history[:20]]), old["body_q"], "current reference q")
    equal(np.asarray([x[1] for x in history[:20]]), old["body_qd"], "current reference qd")
    example = Example(ViewerNull(), config)
    for frame in range(20):
        example.step()
        equal(example.state_0.body_q.numpy(), history[frame][0], f"capture prefix q {frame}")
        equal(example.state_0.body_qd.numpy(), history[frame][1], f"capture prefix qd {frame}")
    block = example.solver._reduced_articulation.contact_block_system
    scalar = block.build_rows_kernel
    candidate, generated = make_candidate(scalar_lane_values=args.scalar_lane_values)
    launch_original = wp.launch
    captures = []
    seen = set()

    def launch(*positional, **kwargs):
        kernel = kwargs.get("kernel", positional[0] if positional else None)
        if kernel is scalar and len(captures) < 4:
            inputs = kwargs["inputs"] + kwargs.get("outputs", [])
            rows = inputs[3].numpy()
            page = int(inputs[11].numpy()[0])
            prepare = bool(inputs[12])
            key = (page, prepare, tuple(rows.reshape(-1)))
            if np.max(rows) > 0 and key not in seen:
                seen.add(key)
                arrays = {}
                values = [clone(v, arrays) for v in inputs]
                captures.append((values, kwargs["dim"], kwargs.get("block_dim", 48)))
        return launch_original(*positional, **kwargs)

    example.graph = None
    wp.launch = launch
    try:
        example.step()
    finally:
        wp.launch = launch_original
    equal(example.state_0.body_q.numpy(), history[20][0], "eager capture q")
    equal(example.state_0.body_qd.numpy(), history[20][1], "eager capture qd")
    assert captures, "No active packed point-row capture"
    result = {
        "status": "running",
        "generated_source": str(generated),
        "trajectory_byte_gate": True,
        "cases": [],
        "metadata": [],
    }
    args.output.write_text(json.dumps(result, indent=2))
    for capture_id, (values, dim, block_dim) in enumerate(captures):
        packed_page = int(values[11].numpy()[0])
        metadata = {
            "capture": capture_id,
            "dim": list(dim),
            "scalar_block_dim": block_dim,
            "row_count": values[3].numpy().tolist(),
            "point_count": values[2].numpy().tolist(),
            "page": packed_page,
            "max_pages": values[10].numpy().tolist(),
            "prepare": bool(values[12]),
            "row_body": values[4].numpy().tolist(),
            "row_body_pair": values[6].numpy().tolist(),
            "articulation_start": values[0].reduced.articulation_start.numpy().tolist(),
            "articulation_end": values[0].reduced.articulation_end.numpy().tolist(),
            "joint_qd_start": values[0].reduced.joint_qd_start.numpy().tolist(),
        }
        result["metadata"].append(metadata)
        saved = {}
        for index, value in enumerate(values):
            save_arrays(value, f"arg_{index}", saved)
        np.savez_compressed(args.output.with_suffix(f".capture{capture_id}.npz"), **saved)
        variants = [("actual", None), ("disabled", None), ("not_prepare", None)]
        variants += [(f"cap_{cap}", cap) for cap in (0, 1, 2, 3, 6, 7, 8, 31, 32, 33, 95, 96)]
        variants += [
            (name, None) for name in ("synthetic32", "synthetic33", "synthetic96", "pair", "stale_pair", "stale_body")
        ]
        for variant, cap in variants:
            arrays_a, arrays_b = {}, {}
            a, b = [clone(v, arrays_a) for v in values], [clone(v, arrays_b) for v in values]
            if cap is not None:
                counts = np.minimum(values[3].numpy(), cap)
                a[3].assign(counts)
                b[3].assign(counts)
            if variant == "disabled":
                a[1].zero_()
                b[1].zero_()
            if variant == "not_prepare":
                a[12] = b[12] = wp.bool(False)
            if variant.startswith("synthetic"):
                count = int(variant.removeprefix("synthetic"))
                for target in (a, b):
                    # Duplicate real axes into distinct contact slots: storage
                    # boundary coverage only, not a real manifold trajectory.
                    for index in (4, 5, 6, 7):
                        data = target[index].numpy()
                        for row in range(3, 96):
                            data[:, row] = data[:, row % 3]
                        target[index].assign(data)
                    contact_ids = target[8].numpy()
                    source_contact = int(contact_ids[0, 0])
                    contact_ids[:] = np.arange(contact_ids.shape[1])
                    target[8].assign(contact_ids)
                    for field in target[9]._cls.vars:
                        array = getattr(target[9], field)
                        data = array.numpy()
                        data[:, :32] = data[:, source_contact : source_contact + 1]
                        array.assign(data)
                    counts = target[3].numpy()
                    counts[:] = count
                    target[3].assign(counts)
                    counts = target[2].numpy()
                    counts[:] = (count + 2) // 3
                    target[2].assign(counts)
            if variant in ("pair", "stale_pair", "stale_body"):
                for target in (a, b):
                    if variant == "pair":
                        target[6].assign(target[4].numpy())
                        target[7].assign(-target[5].numpy())
                    elif variant == "stale_pair":
                        previous = target[14].numpy()
                        previous[:] = int(target[4].numpy()[0, 0])
                        target[14].assign(previous)
                    else:
                        previous = target[13].numpy()
                        previous[:] = 0
                        target[13].assign(previous)
            launch_original(scalar, dim=dim, inputs=a, block_dim=block_dim, device=example.model.device)
            launch_original(candidate, dim=(dim[0], dim[1] * 8), inputs=b, block_dim=128, device=example.model.device)
            for key, array in arrays_a.items():
                equal(array.numpy(), arrays_b[key].numpy(), f"capture{capture_id}/{variant}/{key}")
            result["cases"].append(
                {"capture": capture_id, "variant": variant, "arrays": len(arrays_a), "byte_equal": True}
            )
            args.output.write_text(json.dumps(result, indent=2))
        # Timed actual-input replays restore every reachable input/scratch array.
        # Events exclude reset copies, and each measurement starts from the same
        # frozen state. These are kernel timings, not controller speed claims.
        if args.timing_repeats:
            timing = {}
            for label, kernel, launch_dim, launch_block in (
                ("scalar", scalar, dim, block_dim),
                ("cooperative", candidate, (dim[0], dim[1] * 8), 128),
            ):
                initial, work = {}, {}
                for value in values:
                    clone(value, initial)
                working_values = [clone(v, work) for v in values]
                start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
                with wp.ScopedCapture(device=example.model.device) as capture:
                    for key, array in initial.items():
                        wp.copy(work[key], array)
                    wp.record_event(start, external=True)
                    launch_original(
                        kernel,
                        dim=launch_dim,
                        inputs=working_values,
                        block_dim=launch_block,
                        device=example.model.device,
                    )
                    wp.record_event(end, external=True)
                # Captured event nodes exclude host enqueue gaps and copies.
                samples = []
                for repeat in range(args.timing_repeats + 10):
                    wp.capture_launch(capture.graph)
                    wp.synchronize_event(end)
                    elapsed = wp.get_event_elapsed_time(start, end)
                    if repeat >= 10:
                        samples.append(elapsed)
                hooks = kernel.module.execs[(example.model.device.context, launch_block)].get_kernel_hooks(kernel)
                driver = ctypes.CDLL("libcuda.so.1")
                resources = {}
                for attribute, name in ((0, "max_threads"), (1, "shared_bytes"), (3, "local_bytes"), (4, "registers")):
                    value = ctypes.c_int()
                    status = driver.cuFuncGetAttribute(
                        ctypes.byref(value), ctypes.c_int(attribute), ctypes.c_void_p(hooks.forward)
                    )
                    assert status == 0, status
                    resources[name] = value.value
                timing[label] = {"median_ms": float(np.median(samples)), "samples_ms": samples, "resources": resources}
            metadata["timing"] = timing
    source_after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in fingerprint_files}
    assert source_before == source_after, "Production source changed during replay"
    result.update(status="passed", source_hashes=source_after)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "status": result["status"],
                "captures": len(captures),
                "cases": len(result["cases"]),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
