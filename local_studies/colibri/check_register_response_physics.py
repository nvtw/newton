"""Reuse the independent physical gate with the actual register-local helper."""

import shutil
import sys
from pathlib import Path

import warp as wp

from local_studies.colibri import check_cached_response_physics as reference
from local_studies.colibri.register_component_response import serial_pair_local
from newton._src.solvers.phoenx.articulations.maximal_contact_response import MaximalContactResponseData
from newton._src.solvers.phoenx.articulations.maximal_projector import MaximalTreeProjectorData


@wp.func
def register_response(tree: MaximalTreeProjectorData, response: MaximalContactResponseData, articulation: int):
    return serial_pair_local(
        response.mobility[articulation, 0],
        tree.articulated[articulation, 1],
        tree.motion[articulation, 1],
        tree.shift[articulation, 1],
        tree.inverse_d[articulation, 1],
        response.impulse[articulation, 0],
        response.impulse[articulation, 1],
    )


def main():
    original = "/tmp/colibri_cached_component_frozen1"
    output = "/tmp/colibri_register_response_gate"
    for suffix in (".cached_response.npz", ".native_state.npz"):
        shutil.copyfile(Path(original + suffix), Path(output + suffix))
    reference.serial_native_pair = register_response
    sys.argv = [__file__, "--prefix", output, "--serial"]
    reference.main()


if __name__ == "__main__":
    main()
