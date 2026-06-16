# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ast

from ...solver_kamino_impl import SolverKaminoImpl

###
# Module interface
###

__all__ = [
    "make_benchmark_configs",
    "make_dvi_padmm_benchmark_configs",
    "make_solver_config_default",
    "make_solver_config_dense_dvi_dr_legs",
    "make_solver_config_dense_jacobian_llt_accurate",
    "make_solver_config_dense_jacobian_llt_fast",
    "make_solver_config_sparse_delassus_cr_accurate",
    "make_solver_config_sparse_delassus_cr_fast",
    "make_solver_config_sparse_jacobian_llt_accurate",
    "make_solver_config_sparse_jacobian_llt_fast",
]


###
# Solver configurations
###


def make_solver_config_default() -> tuple[str, SolverKaminoImpl.Config]:
    # ------------------------------------------------------------------------------
    name = "Default"
    # ------------------------------------------------------------------------------
    config = SolverKaminoImpl.Config()
    # ------------------------------------------------------------------------------
    # Constraint stabilization
    config.constraints.alpha = 0.1
    # ------------------------------------------------------------------------------
    # Jacobian representation
    config.sparse_jacobian = False
    config.sparse_dynamics = False
    # ------------------------------------------------------------------------------
    # Linear system solver
    config.dynamics.linear_solver_type = "LLTB"
    config.dynamics.linear_solver_kwargs = {}
    # ------------------------------------------------------------------------------
    # PADMM
    config.padmm.max_iterations = 200
    config.padmm.primal_tolerance = 1e-6
    config.padmm.dual_tolerance = 1e-6
    config.padmm.compl_tolerance = 1e-6
    config.padmm.restart_tolerance = 0.999
    config.padmm.eta = 1e-5
    config.padmm.rho_0 = 1.0
    config.padmm.rho_min = 1e-5
    config.padmm.penalty_update_method = "fixed"
    config.padmm.penalty_update_freq = 1
    config.padmm.use_acceleration = True
    config.padmm.use_graph_conditionals = False
    # ------------------------------------------------------------------------------
    # Warm-starting
    config.padmm.warmstart_mode = "containers"
    config.padmm.contact_warmstart_method = "geom_pair_net_force"
    # ------------------------------------------------------------------------------
    return name, config


def make_solver_config_dense_jacobian_llt_accurate() -> tuple[str, SolverKaminoImpl.Config]:
    # ------------------------------------------------------------------------------
    name = "Dense Jacobian LLT accurate"
    # ------------------------------------------------------------------------------
    config = SolverKaminoImpl.Config()
    # ------------------------------------------------------------------------------
    # Constraint stabilization
    config.constraints.alpha = 0.1
    # ------------------------------------------------------------------------------
    # Jacobian representation
    config.sparse_dynamics = False
    config.sparse_jacobian = False
    # ------------------------------------------------------------------------------
    # Linear system solver
    config.dynamics.linear_solver_type = "LLTB"
    config.dynamics.linear_solver_kwargs = {}
    # ------------------------------------------------------------------------------
    # PADMM
    config.padmm.max_iterations = 200
    config.padmm.primal_tolerance = 1e-6
    config.padmm.dual_tolerance = 1e-6
    config.padmm.compl_tolerance = 1e-6
    config.padmm.restart_tolerance = 0.999
    config.padmm.eta = 1e-5
    config.padmm.rho_0 = 0.1
    config.padmm.rho_min = 1e-5
    config.padmm.penalty_update_method = "fixed"
    config.padmm.penalty_update_freq = 1
    config.padmm.use_acceleration = True
    config.padmm.use_graph_conditionals = False
    # ------------------------------------------------------------------------------
    # Warm-starting
    config.padmm.warmstart_mode = "containers"
    config.padmm.contact_warmstart_method = "geom_pair_net_force"
    # ------------------------------------------------------------------------------
    return name, config


def make_solver_config_dense_jacobian_llt_fast() -> tuple[str, SolverKaminoImpl.Config]:
    # ------------------------------------------------------------------------------
    name = "Dense Jacobian LLT fast"
    # ------------------------------------------------------------------------------
    config = SolverKaminoImpl.Config()
    # ------------------------------------------------------------------------------
    # Constraint stabilization
    config.constraints.alpha = 0.1
    # ------------------------------------------------------------------------------
    # Jacobian representation
    config.sparse_dynamics = False
    config.sparse_jacobian = False
    # ------------------------------------------------------------------------------
    # Linear system solver
    config.dynamics.linear_solver_type = "LLTB"
    config.dynamics.linear_solver_kwargs = {}
    # ------------------------------------------------------------------------------
    # PADMM
    config.padmm.max_iterations = 100
    config.padmm.primal_tolerance = 1e-4
    config.padmm.dual_tolerance = 1e-4
    config.padmm.compl_tolerance = 1e-4
    config.padmm.restart_tolerance = 0.999
    config.padmm.eta = 1e-5
    config.padmm.rho_0 = 0.02
    config.padmm.rho_min = 1e-5
    config.padmm.penalty_update_method = "fixed"
    config.padmm.penalty_update_freq = 1
    config.padmm.use_acceleration = True
    config.padmm.use_graph_conditionals = False
    # ------------------------------------------------------------------------------
    # Warm-starting
    config.padmm.warmstart_mode = "containers"
    config.padmm.contact_warmstart_method = "geom_pair_net_force"
    # ------------------------------------------------------------------------------
    return name, config


def make_solver_config_sparse_jacobian_llt_accurate() -> tuple[str, SolverKaminoImpl.Config]:
    # ------------------------------------------------------------------------------
    name = "Sparse Jacobian LLT accurate"
    # ------------------------------------------------------------------------------
    config = SolverKaminoImpl.Config()
    # ------------------------------------------------------------------------------
    # Constraint stabilization
    config.constraints.alpha = 0.1
    # ------------------------------------------------------------------------------
    # Jacobian representation
    config.sparse_dynamics = False
    config.sparse_jacobian = True
    # ------------------------------------------------------------------------------
    # Linear system solver
    config.dynamics.linear_solver_type = "LLTB"
    config.dynamics.linear_solver_kwargs = {}
    # ------------------------------------------------------------------------------
    # PADMM
    config.padmm.max_iterations = 200
    config.padmm.primal_tolerance = 1e-6
    config.padmm.dual_tolerance = 1e-6
    config.padmm.compl_tolerance = 1e-6
    config.padmm.restart_tolerance = 0.999
    config.padmm.eta = 1e-5
    config.padmm.rho_0 = 0.1
    config.padmm.rho_min = 1e-5
    config.padmm.penalty_update_method = "fixed"
    config.padmm.penalty_update_freq = 1
    config.padmm.use_acceleration = True
    config.padmm.use_graph_conditionals = False
    # ------------------------------------------------------------------------------
    # Warm-starting
    config.padmm.warmstart_mode = "containers"
    config.padmm.contact_warmstart_method = "geom_pair_net_force"
    # ------------------------------------------------------------------------------
    return name, config


def make_solver_config_sparse_jacobian_llt_fast() -> tuple[str, SolverKaminoImpl.Config]:
    # ------------------------------------------------------------------------------
    name = "Sparse Jacobian LLT fast"
    # ------------------------------------------------------------------------------
    config = SolverKaminoImpl.Config()
    # ------------------------------------------------------------------------------
    # Constraint stabilization
    config.constraints.alpha = 0.1
    # ------------------------------------------------------------------------------
    # Jacobian representation
    config.sparse_dynamics = False
    config.sparse_jacobian = True
    # ------------------------------------------------------------------------------
    # Linear system solver
    config.dynamics.linear_solver_type = "LLTB"
    config.dynamics.linear_solver_kwargs = {}
    # ------------------------------------------------------------------------------
    # PADMM
    config.padmm.max_iterations = 100
    config.padmm.primal_tolerance = 1e-4
    config.padmm.dual_tolerance = 1e-4
    config.padmm.compl_tolerance = 1e-4
    config.padmm.restart_tolerance = 0.999
    config.padmm.eta = 1e-5
    config.padmm.rho_0 = 0.02
    config.padmm.rho_min = 1e-5
    config.padmm.penalty_update_method = "fixed"
    config.padmm.penalty_update_freq = 1
    config.padmm.use_acceleration = True
    config.padmm.use_graph_conditionals = False
    # ------------------------------------------------------------------------------
    # Warm-starting
    config.padmm.warmstart_mode = "containers"
    config.padmm.contact_warmstart_method = "geom_pair_net_force"
    # ------------------------------------------------------------------------------
    return name, config


def make_solver_config_sparse_delassus_cr_accurate() -> tuple[str, SolverKaminoImpl.Config]:
    # ------------------------------------------------------------------------------
    name = "Sparse Delassus CR accurate"
    # ------------------------------------------------------------------------------
    config = SolverKaminoImpl.Config()
    # ------------------------------------------------------------------------------
    # Constraint stabilization
    config.constraints.alpha = 0.1
    # ------------------------------------------------------------------------------
    # Jacobian representation
    config.sparse_dynamics = True
    config.sparse_jacobian = True
    # ------------------------------------------------------------------------------
    # Linear system solver
    config.dynamics.linear_solver_type = "CR"
    config.dynamics.linear_solver_kwargs = {"maxiter": 30}
    # ------------------------------------------------------------------------------
    # PADMM
    config.padmm.max_iterations = 200
    config.padmm.primal_tolerance = 1e-6
    config.padmm.dual_tolerance = 1e-6
    config.padmm.compl_tolerance = 1e-6
    config.padmm.restart_tolerance = 0.999
    config.padmm.eta = 1e-5
    config.padmm.rho_0 = 0.1
    config.padmm.rho_min = 1e-5
    config.padmm.penalty_update_method = "fixed"
    config.padmm.penalty_update_freq = 1
    config.padmm.use_acceleration = True
    config.padmm.use_graph_conditionals = False
    # ------------------------------------------------------------------------------
    # Warm-starting
    config.padmm.warmstart_mode = "containers"
    config.padmm.contact_warmstart_method = "geom_pair_net_force"
    # ------------------------------------------------------------------------------
    return name, config


def make_solver_config_sparse_delassus_cr_fast() -> tuple[str, SolverKaminoImpl.Config]:
    # ------------------------------------------------------------------------------
    name = "Sparse Delassus CR fast"
    # ------------------------------------------------------------------------------
    config = SolverKaminoImpl.Config()
    # ------------------------------------------------------------------------------
    # Jacobian representation
    config.sparse_dynamics = True
    config.sparse_jacobian = True
    # ------------------------------------------------------------------------------
    # Constraint stabilization
    config.constraints.alpha = 0.1
    # ------------------------------------------------------------------------------
    # Linear system solver
    config.dynamics.linear_solver_type = "CR"
    config.dynamics.linear_solver_kwargs = {"maxiter": 9}
    # ------------------------------------------------------------------------------
    # PADMM
    config.padmm.max_iterations = 100
    config.padmm.primal_tolerance = 1e-4
    config.padmm.dual_tolerance = 1e-4
    config.padmm.compl_tolerance = 1e-4
    config.padmm.restart_tolerance = 0.999
    config.padmm.eta = 1e-5
    config.padmm.rho_0 = 0.02
    config.padmm.rho_min = 1e-5
    config.padmm.penalty_update_method = "fixed"
    config.padmm.penalty_update_freq = 1
    config.padmm.use_acceleration = True
    config.padmm.use_graph_conditionals = False
    # ------------------------------------------------------------------------------
    # Warm-starting
    config.padmm.warmstart_mode = "containers"
    config.padmm.contact_warmstart_method = "geom_pair_net_force"
    # ------------------------------------------------------------------------------
    return name, config


def make_solver_config_dense_dvi_dr_legs() -> tuple[str, SolverKaminoImpl.Config]:
    # ------------------------------------------------------------------------------
    name = "Dense DVI Dr Legs"
    # ------------------------------------------------------------------------------
    config = SolverKaminoImpl.Config()
    # ------------------------------------------------------------------------------
    # Dynamics solver
    config.dynamics_solver = "dvi"
    config.integrator = "moreau"
    # ------------------------------------------------------------------------------
    # Constraint stabilization
    config.constraints.alpha = 0.1
    config.constraints.beta = 0.011
    config.constraints.gamma = 0.05
    # ------------------------------------------------------------------------------
    # Jacobian representation
    config.sparse_dynamics = False
    config.sparse_jacobian = False
    # ------------------------------------------------------------------------------
    # Linear system solver
    config.dynamics.linear_solver_type = "LLTB"
    config.dynamics.linear_solver_kwargs = {}
    # ------------------------------------------------------------------------------
    # DVI
    config.dvi.max_iterations = 200
    config.dvi.tolerance = 1e-4
    config.dvi.regularization = 1e-5
    config.dvi.omega = 1.0
    # ------------------------------------------------------------------------------
    # Warm-starting
    config.dvi.warmstart_mode = "containers"
    config.dvi.contact_warmstart_method = "geom_pair_net_force"
    # ------------------------------------------------------------------------------
    return name, config


###
# Utilities
###


def make_benchmark_configs(include_default: bool = True) -> dict[str, SolverKaminoImpl.Config]:
    if include_default:
        generators = [make_solver_config_default]
    else:
        generators = []
    generators.extend(
        [
            make_solver_config_dense_jacobian_llt_accurate,
            make_solver_config_dense_jacobian_llt_fast,
            make_solver_config_sparse_jacobian_llt_accurate,
            make_solver_config_sparse_jacobian_llt_fast,
            make_solver_config_sparse_delassus_cr_accurate,
            make_solver_config_sparse_delassus_cr_fast,
            make_solver_config_dense_dvi_dr_legs,
        ]
    )
    solver_configs: dict[str, SolverKaminoImpl.Config] = {}
    for gen in generators:
        name, config = gen()
        solver_configs[name] = config
    return solver_configs


def make_dvi_padmm_benchmark_configs() -> dict[str, SolverKaminoImpl.Config]:
    """Return the focused dense PADMM/DVI solver set used by DVI benchmarks."""
    configs: dict[str, SolverKaminoImpl.Config] = {}

    _, config = make_solver_config_dense_jacobian_llt_accurate()
    configs["PADMM accurate"] = config

    _, config = make_solver_config_dense_jacobian_llt_fast()
    configs["PADMM fast"] = config

    _, config = make_solver_config_dense_dvi_dr_legs()
    configs["DVI"] = config

    return configs


###
# Functions
###


def save_solver_configs_to_hdf5(configs: dict[str, SolverKaminoImpl.Config], datafile):
    for config_name, config in configs.items():
        scope = f"Solver/{config_name}"
        # ------------------------------------------------------------------------------
        datafile[f"{scope}/dynamics_solver"] = config.dynamics_solver
        datafile[f"{scope}/integrator"] = config.integrator
        datafile[f"{scope}/sparse_jacobian"] = config.sparse_jacobian
        datafile[f"{scope}/sparse_dynamics"] = config.sparse_dynamics
        # ------------------------------------------------------------------------------
        datafile[f"{scope}/constraints/alpha"] = config.constraints.alpha
        datafile[f"{scope}/constraints/beta"] = config.constraints.beta
        datafile[f"{scope}/constraints/gamma"] = config.constraints.gamma
        datafile[f"{scope}/constraints/delta"] = config.constraints.delta
        # ------------------------------------------------------------------------------
        datafile[f"{scope}/dynamics/preconditioning"] = config.dynamics.preconditioning
        datafile[f"{scope}/dynamics/linear_solver/type"] = str(config.dynamics.linear_solver_type)
        datafile[f"{scope}/dynamics/linear_solver/args"] = f"{config.dynamics.linear_solver_kwargs}"
        # ------------------------------------------------------------------------------
        datafile[f"{scope}/padmm/max_iterations"] = config.padmm.max_iterations
        datafile[f"{scope}/padmm/primal_tolerance"] = config.padmm.primal_tolerance
        datafile[f"{scope}/padmm/dual_tolerance"] = config.padmm.dual_tolerance
        datafile[f"{scope}/padmm/compl_tolerance"] = config.padmm.compl_tolerance
        datafile[f"{scope}/padmm/restart_tolerance"] = config.padmm.restart_tolerance
        datafile[f"{scope}/padmm/eta"] = config.padmm.eta
        datafile[f"{scope}/padmm/rho_0"] = config.padmm.rho_0
        datafile[f"{scope}/padmm/rho_min"] = config.padmm.rho_min
        datafile[f"{scope}/padmm/a_0"] = config.padmm.a_0
        datafile[f"{scope}/padmm/alpha"] = config.padmm.alpha
        datafile[f"{scope}/padmm/tau"] = config.padmm.tau
        datafile[f"{scope}/padmm/penalty_update_method"] = config.padmm.penalty_update_method
        datafile[f"{scope}/padmm/penalty_update_freq"] = config.padmm.penalty_update_freq
        datafile[f"{scope}/padmm/linear_solver_tolerance"] = config.padmm.linear_solver_tolerance
        datafile[f"{scope}/padmm/linear_solver_tolerance_ratio"] = config.padmm.linear_solver_tolerance_ratio
        datafile[f"{scope}/padmm/use_acceleration"] = config.padmm.use_acceleration
        datafile[f"{scope}/padmm/use_graph_conditionals"] = config.padmm.use_graph_conditionals
        # ------------------------------------------------------------------------------
        datafile[f"{scope}/dvi/max_iterations"] = config.dvi.max_iterations
        datafile[f"{scope}/dvi/tolerance"] = config.dvi.tolerance
        datafile[f"{scope}/dvi/regularization"] = config.dvi.regularization
        datafile[f"{scope}/dvi/omega"] = config.dvi.omega
        datafile[f"{scope}/dvi/warmstart_mode"] = config.dvi.warmstart_mode
        datafile[f"{scope}/dvi/contact_warmstart_method"] = config.dvi.contact_warmstart_method
        # ------------------------------------------------------------------------------
        datafile[f"{scope}/warmstarting/warmstart_mode"] = config.padmm.warmstart_mode
        datafile[f"{scope}/warmstarting/contact_warmstart_method"] = config.padmm.contact_warmstart_method


def _read_hdf5_string(datafile, path: str) -> str:
    value = datafile[path][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_solver_configs_to_hdf5(datafile) -> dict[str, SolverKaminoImpl.Config]:
    configs = {}
    for config_name in datafile["Solver"].keys():
        config = SolverKaminoImpl.Config()
        # ------------------------------------------------------------------------------
        scope = f"Solver/{config_name}"
        if f"{scope}/dynamics_solver" in datafile:
            config.dynamics_solver = _read_hdf5_string(datafile, f"{scope}/dynamics_solver")
        if f"{scope}/integrator" in datafile:
            config.integrator = _read_hdf5_string(datafile, f"{scope}/integrator")
        config.sparse_jacobian = bool(datafile[f"{scope}/sparse_jacobian"][()])
        config.sparse_dynamics = bool(datafile[f"{scope}/sparse_dynamics"][()])
        # ------------------------------------------------------------------------------
        config.constraints.alpha = float(datafile[f"{scope}/constraints/alpha"][()])
        config.constraints.beta = float(datafile[f"{scope}/constraints/beta"][()])
        config.constraints.gamma = float(datafile[f"{scope}/constraints/gamma"][()])
        config.constraints.delta = float(datafile[f"{scope}/constraints/delta"][()])
        # ------------------------------------------------------------------------------
        config.dynamics.preconditioning = bool(datafile[f"{scope}/dynamics/preconditioning"][()])
        config.dynamics.linear_solver_type = _read_hdf5_string(datafile, f"{scope}/dynamics/linear_solver/type")
        config.dynamics.linear_solver_kwargs = ast.literal_eval(
            _read_hdf5_string(datafile, f"{scope}/dynamics/linear_solver/args")
        )
        # ------------------------------------------------------------------------------
        config.padmm.max_iterations = int(datafile[f"{scope}/padmm/max_iterations"][()])
        config.padmm.primal_tolerance = float(datafile[f"{scope}/padmm/primal_tolerance"][()])
        config.padmm.dual_tolerance = float(datafile[f"{scope}/padmm/dual_tolerance"][()])
        config.padmm.compl_tolerance = float(datafile[f"{scope}/padmm/compl_tolerance"][()])
        config.padmm.restart_tolerance = float(datafile[f"{scope}/padmm/restart_tolerance"][()])
        config.padmm.eta = float(datafile[f"{scope}/padmm/eta"][()])
        config.padmm.rho_0 = float(datafile[f"{scope}/padmm/rho_0"][()])
        config.padmm.rho_min = float(datafile[f"{scope}/padmm/rho_min"][()])
        config.padmm.a_0 = float(datafile[f"{scope}/padmm/a_0"][()])
        config.padmm.alpha = float(datafile[f"{scope}/padmm/alpha"][()])
        config.padmm.tau = float(datafile[f"{scope}/padmm/tau"][()])
        config.padmm.penalty_update_method = _read_hdf5_string(datafile, f"{scope}/padmm/penalty_update_method")
        config.padmm.penalty_update_freq = int(datafile[f"{scope}/padmm/penalty_update_freq"][()])
        config.padmm.linear_solver_tolerance = float(datafile[f"{scope}/padmm/linear_solver_tolerance"][()])
        config.padmm.linear_solver_tolerance_ratio = float(datafile[f"{scope}/padmm/linear_solver_tolerance_ratio"][()])
        config.padmm.use_acceleration = bool(datafile[f"{scope}/padmm/use_acceleration"][()])
        config.padmm.use_graph_conditionals = bool(datafile[f"{scope}/padmm/use_graph_conditionals"][()])
        # ------------------------------------------------------------------------------
        config.padmm.warmstart_mode = _read_hdf5_string(datafile, f"{scope}/warmstarting/warmstart_mode")
        config.padmm.contact_warmstart_method = _read_hdf5_string(
            datafile, f"{scope}/warmstarting/contact_warmstart_method"
        )
        if f"{scope}/dvi" in datafile:
            config.dvi.max_iterations = int(datafile[f"{scope}/dvi/max_iterations"][()])
            config.dvi.tolerance = float(datafile[f"{scope}/dvi/tolerance"][()])
            config.dvi.regularization = float(datafile[f"{scope}/dvi/regularization"][()])
            config.dvi.omega = float(datafile[f"{scope}/dvi/omega"][()])
            config.dvi.warmstart_mode = _read_hdf5_string(datafile, f"{scope}/dvi/warmstart_mode")
            config.dvi.contact_warmstart_method = _read_hdf5_string(datafile, f"{scope}/dvi/contact_warmstart_method")
        # ------------------------------------------------------------------------------
        configs[config_name] = config
    return configs
