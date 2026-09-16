"""Local combined material-history experiment; production sources stay untouched."""

# ruff: noqa: PLC0415 -- source overlays must precede Newton imports.

import hashlib
import json
import runpy
import sys
from pathlib import Path

from local_studies.colibri import friction_break_state, friction_gap_history


def install():
    """Install source overlays before Newton imports and matching before capture."""
    previous_sources = friction_break_state.sources

    def sources():
        original, modified = previous_sources()
        modified["contact_ingest"] = friction_break_state.clear_fresh_positive_warmstart(modified["contact_ingest"])
        return original, modified

    friction_break_state.sources = sources
    original, modified, directory = friction_gap_history.install()

    import newton
    from local_studies.colibri import history_only_matching

    constructor = newton.CollisionPipeline.__init__
    installed = []

    def initialize(pipeline, *args, **kwargs):
        constructor(pipeline, *args, **kwargs)
        if pipeline._matching_sticky:
            installed.append(history_only_matching.install(pipeline))

    newton.CollisionPipeline.__init__ = initialize
    return original, modified, directory, constructor, installed


def main():
    original, modified, directory, constructor, installed = install()
    module, *arguments = sys.argv[1:]
    sys.argv = [module, *arguments]
    try:
        runpy.run_module(module, run_name="__main__")
    finally:
        import newton

        newton.CollisionPipeline.__init__ = constructor
        for name, source in original.items():
            path = friction_break_state.ROOT / "newton/_src/solvers/phoenx/constraints" / f"{name}.py"
            assert path.read_text() == source
        if "--output" in arguments:
            output = Path(arguments[arguments.index("--output") + 1])
            output.with_suffix(".combined_history.json").write_text(
                json.dumps(
                    {
                        "production_unchanged": True,
                        "overlay": str(directory),
                        "hashes": {
                            name: hashlib.sha256(source.encode()).hexdigest() for name, source in modified.items()
                        },
                        "arguments": arguments,
                        "sticky_pipelines_installed": len(installed),
                        "geometry_map_preallocated_before_capture": True,
                        "positive_fresh_gap_clears_all_carried_impulses": True,
                        "scope": "Actual break flag; existing material correlation; identity-only positive-gap matching; fresh geometry",
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
