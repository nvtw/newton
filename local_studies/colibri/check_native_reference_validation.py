"""CPU-only checks of the actual reference gate, without importing GPU modules."""

import ast
import hashlib
import tempfile
from pathlib import Path

import numpy as np


def main():
    source = Path(__file__).with_name("check_native_velocity_iterations.py")
    tree = ast.parse(source.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "validate_reference"
    )
    namespace = {"Path": Path, "np": np, "hashlib": hashlib}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    validate = namespace["validate_reference"]
    count = 0
    with tempfile.TemporaryDirectory(prefix="native_reference_gate_") as directory:
        reference = Path(directory) / "reference.npz"
        actual = Path(directory) / "actual.npz"
        base = {
            key: np.ones((2, 3), dtype=np.float32)
            for key in ("q", "qd", "initial_q", "q_history", "qd_history", "history_times", "labels")
        }
        np.savez(reference, **base)
        np.savez(actual, **base)
        assert validate(actual, reference, "trajectory")["byte_identical"]
        count += 1
        assert validate(actual, reference, "snapshot")["byte_identical"]
        count += 1
        for key in ("q", "qd", "q_history", "qd_history"):
            changed = {k: v.copy() for k, v in base.items()}
            changed[key].flat[0] += 1
            np.savez(actual, **changed)
            try:
                validate(actual, reference, "trajectory")
            except AssertionError as error:
                assert key in str(error)
            else:
                raise AssertionError("Changed trajectory accepted")
            count += 1
        np.savez(actual, q=base["q"])
        try:
            validate(actual, reference, "snapshot")
        except AssertionError:
            count += 1
        else:
            raise AssertionError("Missing snapshot fields accepted")
        np.savez(reference, q=base["q"])
        np.savez(actual, q=base["q"])
        try:
            validate(actual, reference, "trajectory")
        except AssertionError:
            count += 1
        else:
            raise AssertionError("History-free reference accepted")
        try:
            validate(reference, reference, "snapshot")
        except ValueError:
            count += 1
        else:
            raise AssertionError("Self-reference accepted")
        np.savez(reference, x=np.array([0.0], dtype=np.float32))
        np.savez(actual, x=np.array([-0.0], dtype=np.float32))
        try:
            validate(actual, reference, "snapshot")
        except AssertionError:
            count += 1
        else:
            raise AssertionError("Signed-zero byte difference accepted")
    print(f"{count} reference validation checks passed; no GPU imports or launches")


if __name__ == "__main__":
    main()
