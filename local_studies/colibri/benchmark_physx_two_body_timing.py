"""Isolated completed-step timing for the previously audited awake two-body USD.

Initialization, collider cooking, property reads, warmup, trajectory checks and
output are outside timing. This preserves the source SDK drive law; matching
authored parameters does not make it identical to the PhoenX discrete drive.
"""

import argparse
import ctypes
import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import ovphysx
import ovstage
from ovphysx import PhysX, PhysXConfig, TensorType


def same_bytes(actual, expected, label):
    """Require exact state/property equality including dtype and signed zero."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, label
    assert actual.tobytes() == expected.tobytes(), label


def main():
    """Time 300 completed frame calls and certify against the saved accuracy run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, default=Path("/tmp/colibri_physx_two_body_awake_plane1mm.usda"))
    parser.add_argument("--reference", type=Path, default=Path("/tmp/colibri_physx_two_body_awake_plane1mm60.npz"))
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true", help="Capture measured frames with the CUDA profiler API")
    parser.add_argument(
        "--contact-report", action="store_true", help="Enable diagnostic contact reporting; retain state parity gates"
    )
    args = parser.parse_args()
    assert args.frames > 0 and args.warmup >= 0
    reference = np.load(args.reference)
    reference_report = json.loads(args.reference.with_suffix(".json").read_text())
    source_hash = hashlib.sha256(args.usd.read_bytes()).hexdigest()
    assert source_hash == reference_report["source_sha256"], "Different USD from accuracy control"
    assert reference_report["position_iterations"] == 30
    assert reference_report["velocity_iterations"] == 1
    assert reference_report["outer_dt"] == 1 / 120
    assert str(ovphysx.__version__) == str(reference_report["ovphysx_version"])
    assert len(reference["q"]) > args.warmup + args.frames
    paths = ["/World/Colibri/FrameAsm/FrameGround", "/World/Colibri/FrameAsm/Frame"]
    same_bytes(np.array(paths), reference["paths"], "body paths")
    physics = PhysX(config=PhysXConfig(cooked_collider_cache_dir="/tmp/colibri_physx_cooked"))
    stage = ovstage.Stage("colibri-two-body-timing")
    bindings = []
    attached = False
    diagnostic_layer = None
    try:
        stage_path = args.usd
        if args.contact_report:
            diagnostic_layer = tempfile.TemporaryDirectory(prefix="colibri_physx_contact_report_")
            stage_path = Path(diagnostic_layer.name) / "report.usda"
            stage_path.write_text(
                "#usda 1.0\n(\n subLayers = [@" + str(args.usd.resolve()) + "@]\n)\n"
                'over "World"\n{\nover "Colibri"\n{\nover "FrameAsm"\n{\n'
                'over "FrameGround" (\n prepend apiSchemas = ["PhysxContactReportAPI"]\n)\n'
                "{\n float physxContactReport:threshold = 0\n}\n"
                'over "Frame" (\n prepend apiSchemas = ["PhysxContactReportAPI"]\n)\n'
                "{\n float physxContactReport:threshold = 0\n}\n}\n}\n}\n"
            )
        ovstage.population.open_usd(stage, str(stage_path), ordinal=1, domains=ovstage.PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        physics.attach_ovstage(stage, read_ordinal=1)
        attached = True
        physics.step_sync(1 / 120)

        def bind(kind):
            binding = physics.create_tensor_binding(
                prim_paths=paths, tensor_type=getattr(TensorType, "RIGID_BODY_" + kind), raise_if_empty=True
            )
            bindings.append(binding)
            return binding, np.zeros(binding.shape, dtype=np.dtype(str(binding.dtype)))

        poses, q = bind("POSE")
        velocities, qd = bind("VELOCITY")
        assert poses.count == 2

        def read_state(index):
            poses.read(q)
            velocities.read(qd)
            qs, vs = q.astype(float), qd.astype(float)
            qs[:, :3] *= 0.01
            vs[:, :3] *= 0.01
            assert np.isfinite(qs).all() and np.isfinite(vs).all()
            same_bytes(qs, reference["q"][index], f"pose frame {index}")
            same_bytes(vs, reference["qd"][index], f"velocity frame {index}")
            return qs.copy(), vs.copy()

        initial_q, initial_qd = read_state(0)
        properties = {}
        for name in ("MASS", "INERTIA", "COM_POSE", "CONTACT_OFFSET", "REST_OFFSET", "SHAPE_FRICTION_AND_RESTITUTION"):
            binding, value = bind(name)
            binding.read(value)
            same_bytes(value, reference[name.lower()], name)
            properties[name.lower()] = value.copy()
        if args.warmup:
            physics.step_n_sync(2 * args.warmup, 1 / 120)
        warm_q, warm_qd = read_state(args.warmup)
        times = []
        cuda = ctypes.CDLL("libcuda.so.1") if args.profile else None
        if cuda is not None:
            assert cuda.cuProfilerStart() == 0
        try:
            for _ in range(args.frames):
                start = time.perf_counter()
                physics.step_n_sync(2, 1 / 120)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
        finally:
            if cuda is not None:
                assert cuda.cuProfilerStop() == 0
        final_q, final_qd = read_state(args.warmup + args.frames)
        contact_report = None
        if args.contact_report:
            contact_report = physics.get_contact_report(include_friction_anchors=True, copy=True)
            args.output.with_suffix(".contacts.json").write_text(json.dumps(contact_report, indent=2, default=str))
        assert hashlib.sha256(args.usd.read_bytes()).hexdigest() == source_hash
        mean = float(np.mean(times))
        report = {
            "status": "passed_byte_gates",
            "scope": __doc__,
            "solver": "PhysX",
            "ovphysx_version": str(ovphysx.__version__),
            "source": str(args.usd),
            "source_sha256": source_hash,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "reference": str(args.reference),
            "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            "body_count": 2,
            "initialization_steps": 1,
            "warmup_frames": args.warmup,
            "measured_frames": args.frames,
            "outer_steps_per_frame": 2,
            "outer_dt": 1 / 120,
            "position_iterations": 30,
            "velocity_iterations": 1,
            "iteration_verification": "Existing audited source/schema; no runtime iteration getter",
            "sleep": "Existing awake overlay, source hash unchanged",
            "gravity_and_materials": "Existing source and six property arrays unchanged",
            "byte_gates": ["q/qd initial", "q/qd postwarmup", "q/qd final", *properties],
            "mean_frame_ms": mean * 1000,
            "p95_frame_ms": float(np.percentile(times, 95)) * 1000,
            "fps": 1 / mean,
            "timing_scope": "Completed step_n_sync(2,1/120) only; no readbacks/render/setup/warmup/output",
            "gpu_isolation": "Must be recorded by the coordinating caller",
            "profile_capture": args.profile,
            "contact_reporting_enabled": args.contact_report,
            "final_contact_counts": None
            if contact_report is None
            else {key: contact_report[key] for key in ("num_headers", "num_points", "num_anchors")},
            "cross_solver_limitation": "Matched temporal budget/source properties, different discrete drive/contact laws",
        }
        np.savez_compressed(
            args.output.with_suffix(".npz"),
            initial_q=initial_q,
            initial_qd=initial_qd,
            warm_q=warm_q,
            warm_qd=warm_qd,
            q=final_q,
            qd=final_qd,
            frame_times=np.asarray(times),
            **properties,
        )
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        for binding in reversed(bindings):
            binding.destroy()
        if attached:
            physics.detach_ovstage()
        stage.destroy()
        physics.release()
        if diagnostic_layer is not None:
            diagnostic_layer.cleanup()


if __name__ == "__main__":
    main()
