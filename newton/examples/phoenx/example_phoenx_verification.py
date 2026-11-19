# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run a Newton Verification benchmark with its authored scene, controls and criteria.

Install the optional evaluation suite from its source checkout with
`uv pip install --python .venv -e /path/to/newton-eval-main`. Select a rigid, articulation,
or manipulation benchmark supported by the suite's CUDA PhoenX candidate.

Success requires completing the benchmark's prescribed physical duration (or
its successful early termination) and meeting its promoted PASS or STRICT tier.

Run commands:
    python -m newton.examples phoenx_verification --scenario rigid.resting_shapes
    python -m newton.examples phoenx_verification --scenario articulation.pendulum --viewer null --test
"""

from __future__ import annotations

import warp as wp

import newton.examples

try:
    from newton_verification.api import (
        BuildContext,
        EvaluationEvidence,
        ProblemInfo,
        QualityTier,
        RunContext,
        benchmark_interactions,
    )
    from newton_verification.artifacts.archive import RunArchiveWriter
    from newton_verification.artifacts.evidence_bridge import EvidenceArchiveBridge
    from newton_verification.capabilities import broad_capability_issue
    from newton_verification.evaluation.criterion_source import evaluate_with_authored_criteria
    from newton_verification.execution.engine import _declare_new_diagnostics, _reset, _snapshots, _synchronize
    from newton_verification.execution.loop_closures import apply_loop_closure_layout
    from newton_verification.registry import create_registries
    from newton_verification.tuning.resolution import resolve_candidate_configuration
except ModuleNotFoundError as exc:
    if exc.name != "newton_verification":
        raise
    create_registries = None


def _registries():
    if create_registries is None:
        raise ModuleNotFoundError(
            "Install Newton Verification with 'uv pip install --python .venv -e /path/to/newton-eval-main'."
        )
    return create_registries(discover_plugins=False, audit=False)


class Example:
    def __init__(self, viewer, args):
        benchmarks, candidates = _registries()
        self.viewer = viewer
        self.benchmark = bench = benchmarks.get(args.scenario)
        candidate = candidates.get("phoenx")
        device = str(wp.get_device(args.device))
        issue = broad_capability_issue(bench.metadata, bench.settings, candidate.metadata, device, "tuned")
        if issue:
            raise ValueError(f"{args.scenario} is unsupported on {device}: {issue.message}")

        self.frame_dt = bench.settings.frame_dt
        self.duration_frames = round(bench.settings.duration / self.frame_dt)
        self.sim_time = 0.0
        self.termination = None
        self.collect = args.test
        self.quiet = getattr(args, "quiet", False)
        problem = ProblemInfo(
            id=bench.metadata.id,
            version=bench.metadata.version,
            category=bench.metadata.category,
            settings=bench.settings,
            tags=bench.metadata.tags,
            interactions=benchmark_interactions(bench, {}),
        )
        config = resolve_candidate_configuration(candidate, problem, "tuned").config
        built = bench.build(BuildContext(device=device, precision=bench.settings.precision, seed=args.seed))
        self.model = built.model
        apply_loop_closure_layout(self.model, candidate.metadata.loop_closure_layout)
        self.runtime = candidate.build(self.model, problem, config)
        for name in ("substep_controls", "substep_forces", "substep_post_solve"):
            callback = getattr(built, name, None)
            if callback is not None:
                getattr(self.runtime, f"set_{name}")(callback)
        self.run = RunContext(built=built, candidate_runtime=self.runtime, state=built.state_0)
        bench.initialize(self.run)
        if candidate.metadata.graph_capture and wp.get_device(device).is_cuda:
            # Mirror qualification warmup and capture. Each phase starts from
            # the authored state, with contact history reset by the candidate.
            snapshots = _snapshots(built)
            for advance in (self.runtime.step, self.runtime.capture):
                bench.before_step(self.run, 0.0)
                advance(built.state_0, built.state_1, built.control, self.frame_dt)
                _synchronize(self.runtime)
                _reset(self.runtime, built, snapshots)
                self.run.state = built.state_0
            self.run.data.clear()
            bench.initialize(self.run)
        viewer.set_model(self.model)

        if self.collect:
            # Use the suite's in-memory series and reductions. No archive is written.
            self.writer = RunArchiveWriter("run.rrd")
            self.bridge = EvidenceArchiveBridge(bench.evidence_schema)
            self.bridge.declare_archive_evidence(self.writer)
            self.diagnostics = {}

    def step(self):
        if self.termination is not None or self.run.frame_index >= self.duration_frames:
            return
        self.benchmark.before_step(self.run, self.sim_time)
        built = self.run.built
        result = self.runtime.step(built.state_0, built.state_1, built.control, self.frame_dt)
        self.run.state = result.current_state
        self.run.frame_index += 1
        self.sim_time = self.run.frame_index * self.frame_dt
        observation = self.benchmark.observe_step(self.run, self.sim_time)
        self.termination = result.terminate or observation.terminate
        if self.collect:
            payload = self.bridge.split_observation(observation)
            diagnostics = _declare_new_diagnostics(self.writer, self.diagnostics, payload.diagnostics)
            self.writer.log_frame(
                self.run.frame_index, self.sim_time, evidence=payload.evidence, diagnostics=diagnostics
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.benchmark.log_state(self.viewer, self.run, self.sim_time)
        self.viewer.end_frame()

    def test_final(self):
        assert self.termination is None or self.termination.success, self.termination
        assert self.termination is not None or self.run.frame_index == self.duration_frames, (
            f"Completed {self.run.frame_index} of {self.duration_frames} required frames"
        )
        metrics = self.writer.scalar_summaries()
        if hasattr(self.benchmark, "case_metrics"):
            metrics.update(self.benchmark.case_metrics(self.run.built))
        metrics.update(frames_completed=self.run.frame_index, simulated_time=self.sim_time)
        evaluation = evaluate_with_authored_criteria(self.benchmark, EvaluationEvidence(metrics))
        assert evaluation.tier in (QualityTier.PASS, QualityTier.STRICT), (
            f"{self.benchmark.metadata.id}: {evaluation.tier.value}: "
            + ", ".join(
                f"{c.id} (observed {c.observed}, threshold {c.threshold})"
                for c in evaluation.criteria
                if c.role != "diagnostic" and c.passes(QualityTier.PASS) is False
            )
        )
        if not self.quiet:
            print(f"{self.benchmark.metadata.id}: {evaluation.tier.value}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--scenario", default="rigid.resting_shapes", help="Newton Verification benchmark ID.")
        parser.add_argument("--seed", type=int, default=0)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    # A short explicit --num-frames can be used for interactive previews.
    selected = parser.parse_args()
    benchmarks, _ = _registries()
    settings = benchmarks.get(selected.scenario).settings
    parser.set_defaults(num_frames=round(settings.duration / settings.frame_dt))
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
