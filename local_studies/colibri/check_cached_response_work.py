"""Independently audit captured contact-response work, including drive reactions.

This CPU audit uses FP64 only to measure error in saved FP32 solver outputs.
For an impulse response with zero hard-constraint velocity, the kinetic
quadratic form plus implicit-drive quadratic form equals impulse work.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def audit(response_path, factor_path):
    """Check the original physical mass and compliant-drive work identity."""
    response = np.load(response_path)
    factor = np.load(factor_path)
    generalized_mass = factor["tree__generalized_mass"][0, :2].astype(float)
    forces = response["forces"].reshape(-1, 12).astype(float)
    mass = response["mass"].astype(float)
    result = {}
    for label, key in (("native", "native"), ("candidate", "cached")):
        velocity = response[key].reshape(-1, 12).astype(float)
        joint = response[key + "_joint"].astype(float)
        kinetic_form = np.einsum("ni,ij,nj->n", velocity, mass, velocity)
        drive_form = np.sum(generalized_mass * joint**2, axis=1)
        work = np.einsum("ni,ni->n", forces, velocity)
        error = kinetic_form + drive_form - work
        scale = np.maximum(np.abs(work), kinetic_form + np.abs(drive_form))
        relative = np.abs(error) / np.maximum(scale, 1e-30)
        result[label] = {
            "maximum_work_error_J": float(np.max(np.abs(error))),
            "maximum_relative_work_error": float(np.max(relative)),
            "minimum_drive_quadratic_form_J": float(np.min(drive_form)),
        }
        assert np.all(np.isfinite(relative)), label
        assert np.all(drive_form >= 0), label
        assert np.max(relative) < 2e-5, (label, result[label])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="/tmp/colibri_cached_component_frozen1")
    parser.add_argument("--suffix", default=".serial_native_physics")
    args = parser.parse_args()
    result = audit(args.prefix + args.suffix + ".npz", args.prefix + ".cached_response.npz")
    Path(args.prefix + args.suffix + ".work.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
