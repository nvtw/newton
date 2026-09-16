"""Isolated control: preserve material references across small positive gaps.

Contact witnesses, collision matching, and force projection stay unchanged.
The existing 2 mm material correlation limit also bounds normal separation.
"""

import hashlib
import json
import runpy
import sys
from pathlib import Path

from local_studies.colibri import friction_break_state


def install():
    original_sources = friction_break_state.sources

    def sources():
        original, modified = original_sources()
        name = "constraint_contact_cloth"
        old = "if solver_gap > wp.float32(0.0) or drift_sq > slip_threshold * slip_threshold:"
        new = "if solver_gap > slip_threshold or drift_sq > slip_threshold * slip_threshold:"
        assert modified[name].count(old) == 1
        modified[name] = modified[name].replace(old, new)
        return original, modified

    friction_break_state.sources = sources
    return friction_break_state.install()


def main():
    original, modified, directory = install()
    module, *arguments = sys.argv[1:]
    sys.argv = [module, *arguments]
    try:
        runpy.run_module(module, run_name="__main__")
    finally:
        for name, source in original.items():
            path = friction_break_state.ROOT / "newton/_src/solvers/phoenx/constraints" / f"{name}.py"
            assert path.read_text() == source
        if "--output" in arguments:
            output = Path(arguments[arguments.index("--output") + 1])
            output.with_suffix(".friction_gap.json").write_text(
                json.dumps(
                    {
                        "production_unchanged": True,
                        "overlay": str(directory),
                        "hashes": {
                            name: hashlib.sha256(source.encode()).hexdigest() for name, source in modified.items()
                        },
                        "arguments": arguments,
                        "scope": "Actual projection break flag plus 2 mm normal material-reference correlation; matching unchanged",
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
