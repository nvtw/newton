# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Corrected native Colibri control with identity-only unclaimed-contact retry."""

from pathlib import Path


def main():
    """Install retry after baseline history matching and record measured recovery."""
    path = Path(__file__).with_name("native_conditioned_two_body.py")
    source = path.read_text()
    marker = "    from newton.solvers import SolverPhoenX"
    assert source.count(marker) == 1
    source = source.replace(
        marker,
        marker
        + """
    from local_studies.colibri.history_match_retry import install as install_retry
    retry_instances = []
    retry_constructor = newton.CollisionPipeline.__init__
    def initialize_retry(pipeline, *args, **kwargs):
        retry_constructor(pipeline, *args, **kwargs)
        if pipeline._matching_sticky:
            retry_instances.append(install_retry(pipeline))
    newton.CollisionPipeline.__init__ = initialize_retry
""",
    )
    constructor_marker = "        constructed.append(solver)"
    assert source.count(constructor_marker) == 1
    source = source.replace(
        constructor_marker,
        constructor_marker
        + "\n        from local_studies.colibri.history_transfer_probe import install as install_transfer_probe\n        phase_audits.append(install_transfer_probe(solver, output))",
    )
    marker = '        assert report["production_unchanged"]'
    assert source.count(marker) == 1
    source = source.replace(
        marker,
        """        assert len(retry_instances) == 1
        retry = retry_instances[0]
        values = retry.stats.numpy()
        events = retry.events.numpy()
        event_count = int(retry.event_count.numpy()[0])
        events = events[np.arange(max(0, event_count - len(events)), event_count) % len(events)]
        saved = {name: array.numpy() for name, array in retry.snapshot.items()}
        saved["first_match"] = retry.original.numpy()
        saved["events"] = events
        np.savez_compressed(output.with_suffix(".retry_match.npz"), **saved)
        output.with_suffix(".retry_match.json").write_text(json.dumps({
            "stats_fields": ["initial_matched", "initial_unmatched", "recovered", "rounds", "duplicate_original_owners"],
            "stats": values.tolist(), "events": event_count,
            "scope": "Existing matched identities frozen; fresh geometry map unchanged; exact existing eligibility; deterministic retry until no remaining eligible claims",
        }, indent=2))
        assert values[4] == 0, "Original matching duplicated predecessor ownership"
"""
        + marker,
    )
    exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})


if __name__ == "__main__":
    main()
