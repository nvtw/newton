"""Compare the Warp translation against a compiled extract of PhysX source.

The original row arithmetic is extracted from the local checkout, not rewritten
as a second Python oracle. Only vector dot products become scalar inputs.
This checks the row update; it does not validate the surrounding simulator.
"""

import argparse
import ctypes
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri.test_physx_normal_reference import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    source_path = Path("/home/twidmer/Documents/git/PhysX/physx/source/gpusolver/src/CUDA/solverBlockTGS.cuh")
    source = source_path.read_text()
    start = source.index("const float sep = PxMax")
    end = source.index("accumDeltaF += deltaF;", start)
    original = source[start:end]
    body = original.replace("b0AngDelta.dot(raXn)", "angular_motion0").replace(
        "b1AngDelta.dot(rbXn)", "angular_motion1"
    )
    names = (
        "separation",
        "deltaV",
        "angular_motion0",
        "angular_motion1",
        "minPen",
        "targetVel",
        "elapsedTime",
        "recipResponse",
        "biasCoefficient",
        "maxPenBias",
        "normalVel",
        "appliedForce",
        "maxImpulse",
    )
    declarations = "\n".join(f"const float {name}=input[{i}];" for i, name in enumerate(names))
    # Retain the original licensing notice in the generated source extract.
    license_notice = source[: source.index("#ifndef")]
    cpp = license_notice + "\n#include <cmath>\nusing PxReal=float;\n"
    cpp += "float PxMax(float a,float b){return fmaxf(a,b);}\n"
    cpp += 'extern "C" void normal_row(const float* input,float* output){\n'
    cpp += declarations + "\nconst float velMultiplier=recipResponse;\n" + body
    cpp += "\noutput[0]=newForce; output[1]=deltaF;\n}\n"
    directory = Path("/tmp/colibri_physx_normal_source_parity")
    directory.mkdir(exist_ok=True)
    cpp_path = directory / "normal_row.cpp"
    library_path = directory / "normal_row.so"
    cpp_path.write_text(cpp)
    subprocess.run(
        ["c++", "-O2", "-shared", "-fPIC", "-ffp-contract=off", str(cpp_path), "-o", str(library_path)],
        check=True,
    )
    library = ctypes.CDLL(str(library_path))
    pointer = ctypes.POINTER(ctypes.c_float)
    library.normal_row.argtypes = [pointer, pointer]
    library.normal_row.restype = None
    rng = np.random.default_rng(317)
    rows = np.zeros((4096, 13), dtype=np.float32)
    rows[:, :4] = rng.uniform(-0.001, 0.001, (len(rows), 4))
    rows[:, 4] = -100
    rows[:, 5] = rng.uniform(-0.1, 0.1, len(rows))
    rows[:, 6] = rng.uniform(0, 1 / 120, len(rows))
    rows[:, 7] = 10 ** rng.uniform(-3, 3, len(rows))
    rows[:, 8] = -rng.uniform(120, 3600, len(rows))
    rows[:, 9] = -2
    rows[:, 10] = rng.uniform(-1, 1, len(rows))
    rows[:, 11] = rng.uniform(0, 1, len(rows))
    rows[:, 12] = rng.uniform(1, 2, len(rows))
    expected = np.zeros((len(rows), 2), dtype=np.float32)
    for row, result in zip(rows, expected, strict=True):
        library.normal_row(row.ctypes.data_as(pointer), result.ctypes.data_as(pointer))
    output = wp.zeros(len(rows), dtype=wp.vec2f, device=args.device)
    wp.launch(evaluate, dim=len(rows), inputs=[wp.array(rows, device=args.device), output], device=args.device)
    actual = output.numpy().copy()
    cpu_expected = expected.copy()
    if args.device != "cpu":
        # Use the same GPU arithmetic policy, including fused operations.
        declarations_cuda = "\n".join(f"const float {name}=input.data[13*i+{j}];" for j, name in enumerate(names))
        snippet = "using PxReal=float;\n" + declarations_cuda + "\nconst float velMultiplier=recipResponse;\n"
        snippet += body.replace("PxMax(", "fmaxf(") + "\noutput.data[i]=wp::vec2f(newForce,deltaF);\n"
        python_source = "\n".join("# " + line for line in license_notice.splitlines())
        python_source += "\nimport warp as wp\n@wp.func_native(" + repr(snippet) + ")\n"
        python_source += "def row(input: wp.array[wp.float32], output: wp.array[wp.vec2f], i: int): ...\n"
        python_source += "@wp.kernel\ndef evaluate(rows: wp.array[wp.float32], out: wp.array[wp.vec2f]):\n"
        python_source += "    row(rows, out, wp.tid())\n"
        module_path = directory / "source_cuda.py"
        module_path.write_text(python_source)
        name = "colibri_physx_normal_source_cuda"
        spec = importlib.util.spec_from_file_location(name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        native_output = wp.zeros(len(rows), dtype=wp.vec2f, device=args.device)
        wp.launch(
            module.evaluate,
            dim=len(rows),
            inputs=[wp.array(rows.ravel(), device=args.device), native_output],
            device=args.device,
        )
        expected = native_output.numpy().copy()
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    report = {
        "source": str(source_path),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "extracted_expression_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "cases": len(rows),
        "max_absolute_impulse_difference": float(np.max(np.abs(actual - expected))),
        "byte_identical": actual.tobytes() == expected.tobytes(),
        "host_no_contraction_max_difference": float(np.max(np.abs(actual - cpu_expected))),
        "device": args.device,
        "scope": "FP32 scalar hard-normal row versus compiled original C++ source extract; no live stability claim",
    }
    (directory / ("result_" + args.device.replace(":", "_") + ".json")).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
