"""Run the native contact control from the independent static equilibrium.

Install the native source overlays before importing the scene builder. Only
initial poses change; the base remains dynamic and contact history starts empty.
"""

import hashlib
import json
import runpy
import sys
from pathlib import Path


def main():
    """Compose existing controls without changing solver settings or sources."""
    arguments = sys.argv[1:]
    output = Path(arguments[arguments.index("--output") + 1])
    directory = Path(__file__).parent
    files = [directory / name for name in ("native_conditioned_two_body.py", "start_analytical_equilibrium.py")]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    original = runpy.run_module
    calls = []

    def run_module(name, *args, **kwargs):
        if name == "local_studies.colibri.staged_base_mechanism" and not calls:
            calls.append(name)
            sys.argv = ["start_analytical_equilibrium", name, *sys.argv[1:]]
            return original("local_studies.colibri.start_analytical_equilibrium", *args, **kwargs)
        return original(name, *args, **kwargs)

    runpy.run_module = run_module
    sys.argv = ["native_conditioned_two_body", *arguments]
    try:
        original("local_studies.colibri.native_conditioned_two_body", run_name="__main__")
    finally:
        runpy.run_module = original
        unchanged = before == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        output.with_suffix(".equilibrium_wrapper.json").write_text(
            json.dumps({"source_sha256": before, "unchanged": unchanged, "intercepted": calls}, indent=2)
        )
        assert unchanged and len(calls) == 1


if __name__ == "__main__":
    main()
