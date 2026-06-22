# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused PADMM/DVI benchmark matrix for Kamino."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import warp as wp

from ...core.builder import ModelBuilderKamino
from ...utils import logger as msg
from ...utils.device import get_device_spec_info
from ...utils.sim import Simulator
from .configs import make_dvi_padmm_benchmark_configs
from .metrics import BenchmarkMetrics
from .problems import BenchmarkProblemNameToConfigFn, make_benchmark_problems
from .render import render_problem_dimensions_table, render_solver_configs_table
from .runner import run_single_benchmark

_CONTACT_STATES = {"contact": True, "no-contact": False}
_CUDA_GRAPH_MODES = {"on": True, "off": False}
_CONFIG_ALIASES = {
    "padmm-accurate": "PADMM accurate",
    "padmm-fast": "PADMM fast",
    "dvi": "DVI",
}


@dataclass(frozen=True)
class MatrixScenario:
    """Single row of the PADMM/DVI benchmark matrix."""

    name: str
    problem: str
    num_worlds: int
    ground: bool
    use_cuda_graph: bool


def make_matrix_scenarios(
    problem: str,
    world_counts: list[int],
    contact_states: list[str],
    cuda_graph_modes: list[str],
) -> list[MatrixScenario]:
    """Create the benchmark scenarios for the requested parameter grid."""
    if problem not in BenchmarkProblemNameToConfigFn:
        raise ValueError(f"Unsupported problem '{problem}'.")
    if any(num_worlds <= 0 for num_worlds in world_counts):
        raise ValueError("World counts must be positive.")

    scenarios: list[MatrixScenario] = []
    for num_worlds in world_counts:
        for contact_state in contact_states:
            if contact_state not in _CONTACT_STATES:
                raise ValueError(f"Unsupported contact state '{contact_state}'.")
            for graph_mode in cuda_graph_modes:
                if graph_mode not in _CUDA_GRAPH_MODES:
                    raise ValueError(f"Unsupported CUDA graph mode '{graph_mode}'.")
                name = f"{problem}/{contact_state}/worlds={num_worlds}/graph={graph_mode}"
                scenarios.append(
                    MatrixScenario(
                        name=name,
                        problem=problem,
                        num_worlds=num_worlds,
                        ground=_CONTACT_STATES[contact_state],
                        use_cuda_graph=_CUDA_GRAPH_MODES[graph_mode],
                    )
                )
    return scenarios


def make_selected_solver_configs(aliases: list[str]) -> dict[str, Any]:
    """Return the focused solver config set filtered by CLI aliases."""
    configs = make_dvi_padmm_benchmark_configs()
    selected: dict[str, Any] = {}
    for alias in aliases:
        if alias not in _CONFIG_ALIASES:
            raise ValueError(f"Unsupported solver config alias '{alias}'.")
        name = _CONFIG_ALIASES[alias]
        selected[name] = configs[name]
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kamino PADMM/DVI benchmark matrix")
    parser.add_argument("--device", type=str, help="Warp device, e.g. 'cuda:0' or 'cpu'.")
    parser.add_argument("--problem", choices=BenchmarkProblemNameToConfigFn.keys(), default="dr_legs")
    parser.add_argument("--world-counts", nargs="+", type=int, default=[1, 16, 64, 256])
    parser.add_argument(
        "--contact-states",
        nargs="+",
        choices=_CONTACT_STATES.keys(),
        default=["contact", "no-contact"],
    )
    parser.add_argument(
        "--cuda-graph-modes",
        nargs="+",
        choices=_CUDA_GRAPH_MODES.keys(),
        default=["off", "on"],
    )
    parser.add_argument(
        "--solver-configs",
        nargs="+",
        choices=_CONFIG_ALIASES.keys(),
        default=list(_CONFIG_ALIASES.keys()),
    )
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--gravity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["total", "stepstats", "convergence", "accuracy"], default="convergence")
    parser.add_argument("--dvi-block-iterations", type=int)
    parser.add_argument("--dvi-contact-iterations", type=int)
    parser.add_argument("--dvi-contact-jacobi-omega", type=float)
    parser.add_argument("--dvi-contact-jacobi-relaxation", type=float)
    parser.add_argument("--dvi-contact-block-preconditioner", action=argparse.BooleanOptionalAction)
    parser.add_argument("--clear-cache", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def apply_dvi_overrides(configs: dict[str, Any], args: argparse.Namespace) -> None:
    """Apply CLI DVI tuning overrides to selected DVI solver configs."""
    for config in configs.values():
        if config.dynamics_solver != "dvi":
            continue
        if args.dvi_block_iterations is not None:
            config.dvi.block_iterations = args.dvi_block_iterations
        if args.dvi_contact_iterations is not None:
            config.dvi.contact_iterations = args.dvi_contact_iterations
        if args.dvi_contact_jacobi_omega is not None:
            config.dvi.contact_jacobi_omega = args.dvi_contact_jacobi_omega
        if args.dvi_contact_jacobi_relaxation is not None:
            config.dvi.contact_jacobi_relaxation = args.dvi_contact_jacobi_relaxation
        if args.dvi_contact_block_preconditioner is not None:
            config.dvi.contact_block_preconditioner = args.dvi_contact_block_preconditioner
        config.dvi.validate()


def run_matrix(args: argparse.Namespace) -> BenchmarkMetrics:
    """Run the requested PADMM/DVI benchmark matrix and return collected metrics."""
    if args.device:
        device = wp.get_device(args.device)
        wp.set_device(device)
    else:
        device = wp.get_preferred_device()

    msg.notif("[Device]: %s", get_device_spec_info(device))
    can_use_cuda_graph = device.is_cuda and wp.is_mempool_enabled(device)

    scenarios = make_matrix_scenarios(
        problem=args.problem,
        world_counts=args.world_counts,
        contact_states=args.contact_states,
        cuda_graph_modes=args.cuda_graph_modes,
    )
    configs = make_selected_solver_configs(args.solver_configs)
    apply_dvi_overrides(configs, args)

    collect_step_metrics = args.mode in {"stepstats", "convergence", "accuracy"}
    collect_solver_metrics = args.mode in {"convergence", "accuracy"}
    collect_physics_metrics = args.mode == "accuracy"
    metrics = BenchmarkMetrics(
        problems=[scenario.name for scenario in scenarios],
        configs=configs,
        num_steps=args.num_steps,
        step_metrics=collect_step_metrics,
        solver_metrics=collect_solver_metrics,
        physics_metrics=collect_physics_metrics,
    )

    render_solver_configs_table(configs=configs, groups=["solver", "sparse", "linear", "padmm", "dvi"], to_console=True)

    runner_args = SimpleNamespace(num_steps=args.num_steps, seed=args.seed, viewer=False)
    for problem_idx, scenario in enumerate(scenarios):
        problem_config = make_benchmark_problems(
            names=[scenario.problem],
            num_worlds=scenario.num_worlds,
            gravity=args.gravity,
            ground=scenario.ground,
        )[scenario.problem]
        builder, control, camera = problem_config
        if not isinstance(builder, ModelBuilderKamino):
            builder = builder()

        use_cuda_graph = scenario.use_cuda_graph and can_use_cuda_graph
        if scenario.use_cuda_graph and not use_cuda_graph:
            msg.warning("CUDA graphs requested for '%s' but are not available on this device.", scenario.name)

        for config_idx, (config_name, config) in enumerate(configs.items()):
            msg.notif("Running '%s' with '%s'", scenario.name, config_name)
            sim_config = Simulator.Config(dt=args.dt, solver=config)
            sim_config.solver.use_fk_solver = False
            run_single_benchmark(
                problem_idx=problem_idx,
                config_idx=config_idx,
                metrics=metrics,
                args=runner_args,
                builder=builder,
                configs=sim_config,
                control=control,
                camera=camera,
                device=device,
                use_cuda_graph=use_cuda_graph,
                print_device_info=False,
            )

    render_problem_dimensions_table(metrics._problem_dims, to_console=True)
    metrics.compute_stats()
    metrics.render_total_metrics_table()
    if metrics.step_time is not None:
        metrics.render_step_time_table()
    if metrics.solver_metrics is not None:
        metrics.render_padmm_metrics_table()
    if metrics.physics_metrics is not None:
        metrics.render_physics_metrics_table()
    return metrics


def main() -> None:
    args = parse_args()
    if args.clear_cache:
        wp.clear_kernel_cache()
        wp.clear_lto_cache()
    msg.set_log_level(msg.LogLevel.INFO)
    run_matrix(args)


if __name__ == "__main__":
    main()
