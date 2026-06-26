# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Population Based Training (PBT) for Newton RL policies."""

from __future__ import annotations

import dataclasses
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .ppo import ConfigPPO, TrainerPPO
from .training import ConfigTrainAnymalPPO, ConfigTrainG1PPO, ResultTrainAnymalPPO, ResultTrainG1PPO


# ---------------------------------------------------------------------------
# Hyperparameter specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HparamSpec:
    """Describes one perturbable hyperparameter.

    Args:
        field: Field name in :class:`ConfigPPO`.
        prior: Sampling distribution — ``"log_uniform"``, ``"uniform"``, or
            ``"categorical"``.
        low: Lower bound for ``log_uniform`` or ``uniform`` priors.
        high: Upper bound for ``log_uniform`` or ``uniform`` priors.
        values: Discrete choices for the ``categorical`` prior.
    """

    field: str
    prior: str = "log_uniform"
    low: float = 1e-5
    high: float = 1e-1
    values: tuple[float, ...] | None = None


# ---------------------------------------------------------------------------
# PBT configuration
# ---------------------------------------------------------------------------


@dataclass
class ConfigPBT:
    """Configuration for :func:`population_based_train`.

    Args:
        population_size: Number of concurrent workers.
        exploit_interval: Training iterations each worker runs per cycle.
        total_cycles: Number of exploit/explore cycles.
        exploit_fraction: Bottom fraction of workers to replace.
        fitness_window: Rolling average window for fitness estimation.
        fitness_metric: Attribute name on per-iteration stats objects used as
            the scalar fitness signal.
        perturbation_factors: Multiplicative range ``(low, high)`` for classic
            perturbation; sampled uniformly in log space.
        resample_probability: Probability of resampling a hyper-parameter from
            its prior rather than perturbing the current value.
        exploit_strategy: ``"truncation"`` selects the top fraction uniformly;
            ``"pb2"`` uses a GP-UCB surrogate for the next hyper-parameter
            vector.
        fresh_optimizer_on_exploit: When ``True``, workers that inherit a
            checkpoint reset their optimizer by loading policy weights only.
        link_lr_fields: Fields that must be perturbed together by the same
            factor so actor/critic learning rates stay in sync.
        seed: RNG seed for sampling and perturbation.
        log_interval: Print a summary every ``log_interval`` cycles.
    """

    population_size: int = 4
    exploit_interval: int = 50
    total_cycles: int = 10
    exploit_fraction: float = 0.25
    fitness_window: int = 10
    fitness_metric: str = "mean_tracking_perf"
    perturbation_factors: tuple[float, float] = (0.8, 1.2)
    resample_probability: float = 0.25
    exploit_strategy: str = "truncation"
    fresh_optimizer_on_exploit: bool = True
    link_lr_fields: tuple[str, ...] = ("actor_lr", "critic_lr")
    seed: int = 0
    log_interval: int = 1


# ---------------------------------------------------------------------------
# Worker and result state
# ---------------------------------------------------------------------------


@dataclass
class WorkerState:
    """Mutable state for a single PBT worker.

    Args:
        worker_id: Unique worker index.
        ppo_config: Current :class:`ConfigPPO` for this worker.
        checkpoint_path: Most recent checkpoint written by this worker, or
            ``None`` if no cycle has completed yet.
        history: Accumulated per-iteration stats objects from all completed
            cycles.
        iteration: Global iteration counter (incremented after each cycle).
        fitness_history: Per-cycle scalar fitness values.
        generation: How many times this worker has inherited a checkpoint from
            a top worker.
        parent_id: Worker ID that last donated a checkpoint, or ``None``.
        hparam_history: Sequence of hyper-parameter dicts, one per cycle.
    """

    worker_id: int
    ppo_config: ConfigPPO
    checkpoint_path: str | None
    history: list[Any] = field(default_factory=list)
    iteration: int = 0
    fitness_history: list[float] = field(default_factory=list)
    generation: int = 0
    parent_id: int | None = None
    hparam_history: list[dict[str, float]] = field(default_factory=list)

    @property
    def fitness(self) -> float:
        """Mean of the most recent fitness values, or ``-inf`` when empty."""
        if not self.fitness_history:
            return float("-inf")
        return float(np.mean(self.fitness_history))


@dataclass
class GenerationResult:
    """Summary of a single PBT exploit/explore cycle.

    Args:
        generation: Zero-based cycle index.
        worker_fitnesses: Mapping from worker_id to scalar fitness.
        exploited_pairs: List of ``(bottom_id, top_id)`` replacements.
        best_worker_id: Worker with the highest fitness this cycle.
        best_fitness: Fitness of the best worker.
        worker_hparams: Mapping from worker_id to hyper-parameter dict.
    """

    generation: int
    worker_fitnesses: dict[int, float]
    exploited_pairs: list[tuple[int, int]]
    best_worker_id: int
    best_fitness: float
    worker_hparams: dict[int, dict[str, float]]


@dataclass
class ResultPBT:
    """Final result of a :func:`population_based_train` run.

    Args:
        best_worker_id: Worker that achieved the best overall fitness.
        best_checkpoint: Path to the best checkpoint seen during training.
        best_fitness: Fitness value at the best checkpoint.
        workers: Final state of all workers.
        generations: Per-cycle summaries.
        output_dir: Root directory used for checkpoints and logs.
    """

    best_worker_id: int
    best_checkpoint: str
    best_fitness: float
    workers: list[WorkerState]
    generations: list[GenerationResult]
    output_dir: str


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _sample_value(spec: HparamSpec, rng: np.random.Generator) -> float:
    """Draw one sample from *spec*'s prior distribution."""
    if spec.prior == "categorical":
        if spec.values is None:
            raise ValueError(f"HparamSpec '{spec.field}' with categorical prior must provide values")
        idx = int(rng.integers(0, len(spec.values)))
        return float(spec.values[idx])
    if spec.prior == "log_uniform":
        log_low = math.log(spec.low)
        log_high = math.log(spec.high)
        return float(math.exp(rng.uniform(log_low, log_high)))
    if spec.prior == "uniform":
        return float(rng.uniform(spec.low, spec.high))
    raise ValueError(f"Unknown prior '{spec.prior}' for field '{spec.field}'")


def _sample_hparams(specs: list[HparamSpec], rng: np.random.Generator) -> dict[str, float]:
    """Sample all specs independently from their priors."""
    return {spec.field: _sample_value(spec, rng) for spec in specs}


def _apply_hparams(config: ConfigPPO, hparams: dict[str, float]) -> ConfigPPO:
    """Return a copy of *config* with *hparams* applied."""
    return dataclasses.replace(config, **hparams)


# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------


def _perturb_classic(
    current: ConfigPPO,
    specs: list[HparamSpec],
    factors: tuple[float, float],
    resample_prob: float,
    link_lr_fields: tuple[str, ...],
    rng: np.random.Generator,
) -> dict[str, float]:
    """Classic PBT perturbation: multiply by a random factor or resample.

    Linked learning-rate fields share a single factor draw so the actor/critic
    ratio is preserved.
    """
    result: dict[str, float] = {}
    linked = set(link_lr_fields)
    lr_factor: float | None = None

    for spec in specs:
        current_val = float(getattr(current, spec.field))
        if float(rng.random()) < resample_prob:
            result[spec.field] = _sample_value(spec, rng)
            continue
        if spec.field in linked:
            if lr_factor is None:
                log_lo = math.log(factors[0])
                log_hi = math.log(factors[1])
                lr_factor = float(math.exp(rng.uniform(log_lo, log_hi)))
            factor = lr_factor
        else:
            log_lo = math.log(factors[0])
            log_hi = math.log(factors[1])
            factor = float(math.exp(rng.uniform(log_lo, log_hi)))
        new_val = current_val * factor
        if spec.prior in ("log_uniform", "uniform"):
            new_val = float(np.clip(new_val, spec.low, spec.high))
        result[spec.field] = new_val
    return result


# ---------------------------------------------------------------------------
# GP-UCB for PB2-style perturbation
# ---------------------------------------------------------------------------


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, length_scale: float) -> np.ndarray:
    """Squared-exponential kernel over normalised hyper-parameter vectors."""
    diffs = X[:, None, :] - Y[None, :, :]
    sq_dist = np.sum(diffs**2, axis=-1) / (length_scale**2 + 1e-12)
    return np.exp(-0.5 * sq_dist)


def _perturb_pb2(
    all_observations: list[tuple[dict[str, float], float]],
    specs: list[HparamSpec],
    n_candidates: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """PB2-style GP-UCB acquisition over the observed population history.

    Falls back to resampling from the prior when fewer than 3 observations are
    available — not enough to fit a meaningful surrogate.
    """
    if len(all_observations) < 3:
        return _sample_hparams(specs, rng)

    fields = [spec.field for spec in specs]
    spec_map = {spec.field: spec for spec in specs}

    def _normalise(val: float, spec: HparamSpec) -> float:
        if spec.prior == "log_uniform":
            lo, hi = math.log(spec.low), math.log(spec.high)
            return (math.log(max(val, 1e-30)) - lo) / (hi - lo + 1e-12)
        if spec.prior == "uniform":
            return (val - spec.low) / (spec.high - spec.low + 1e-12)
        return 0.0

    X = np.array(
        [[_normalise(hp[f], spec_map[f]) for f in fields] for hp, _ in all_observations], dtype=np.float64
    )
    y = np.array([fitness for _, fitness in all_observations], dtype=np.float64)
    y = (y - y.mean()) / (y.std() + 1e-8)

    noise = 1e-3
    length_scale = 0.5
    K = _rbf_kernel(X, X, length_scale) + noise * np.eye(len(X))
    try:
        alpha = np.linalg.solve(K, y)
    except np.linalg.LinAlgError:
        return _sample_hparams(specs, rng)

    candidates_norm = rng.uniform(0.0, 1.0, size=(n_candidates, len(fields)))
    K_star = _rbf_kernel(candidates_norm, X, length_scale)
    mu = K_star @ alpha
    k_star_star = np.ones(n_candidates)
    v = np.linalg.solve(K, K_star.T)
    sigma2 = np.clip(k_star_star - np.sum(K_star * v.T, axis=1), 0.0, None)
    sigma = np.sqrt(sigma2)
    beta = 2.0
    ucb = mu + beta * sigma
    best_idx = int(np.argmax(ucb))
    best_norm = candidates_norm[best_idx]

    result: dict[str, float] = {}
    for i, f in enumerate(fields):
        spec = spec_map[f]
        nv = float(np.clip(best_norm[i], 0.0, 1.0))
        if spec.prior == "log_uniform":
            lo, hi = math.log(spec.low), math.log(spec.high)
            result[f] = float(math.exp(lo + nv * (hi - lo)))
        elif spec.prior == "uniform":
            result[f] = float(spec.low + nv * (spec.high - spec.low))
        elif spec.prior == "categorical":
            if spec.values is None:
                raise ValueError(f"HparamSpec '{f}' categorical prior missing values")
            idx = min(int(nv * len(spec.values)), len(spec.values) - 1)
            result[f] = float(spec.values[idx])
        else:
            result[f] = _sample_value(spec, rng)
    return result


# ---------------------------------------------------------------------------
# Fitness computation
# ---------------------------------------------------------------------------


def _compute_fitness(history: list[Any], fitness_window: int, fitness_metric: str) -> float:
    """Compute scalar fitness from the tail of a stats history."""
    if not history:
        return float("-inf")
    tail = history[-max(1, fitness_window):]
    values = []
    for stat in tail:
        v = getattr(stat, fitness_metric, None)
        if v is not None:
            values.append(float(v))
    if not values:
        return float("-inf")
    return float(np.mean(values))


# ---------------------------------------------------------------------------
# Exploit / explore
# ---------------------------------------------------------------------------


def _current_hparams(config: ConfigPPO, specs: list[HparamSpec]) -> dict[str, float]:
    """Extract only the tracked fields from *config*."""
    return {spec.field: float(getattr(config, spec.field)) for spec in specs}


def _exploit_explore(
    workers: list[WorkerState],
    specs: list[HparamSpec],
    pbt_config: ConfigPBT,
    rng: np.random.Generator,
    output_dir: Path,
    generation: int,
) -> tuple[list[WorkerState], GenerationResult, set[int]]:
    """Truncation selection: bottom workers inherit from the top.

    Returns the updated worker list, a :class:`GenerationResult`, and the set
    of worker IDs that were replaced (so the caller can set
    ``resume_policy_only=True`` for their next cycle).
    """
    n = len(workers)
    fitnesses = {w.worker_id: w.fitness for w in workers}

    sorted_ids = sorted(fitnesses, key=lambda wid: fitnesses[wid], reverse=True)
    n_top = max(1, int(math.ceil(n * pbt_config.exploit_fraction)))
    n_bottom = max(1, int(math.floor(n * pbt_config.exploit_fraction)))
    top_ids = set(sorted_ids[:n_top])
    bottom_ids = sorted_ids[-n_bottom:]

    worker_map = {w.worker_id: w for w in workers}
    exploited_pairs: list[tuple[int, int]] = []
    replaced_ids: set[int] = set()

    all_obs: list[tuple[dict[str, float], float]] = []
    if pbt_config.exploit_strategy == "pb2":
        for w in workers:
            for hp, fit in zip(w.hparam_history, w.fitness_history):
                all_obs.append((hp, fit))

    for bot_id in bottom_ids:
        if bot_id in top_ids:
            continue
        top_id = int(rng.choice(sorted(top_ids)))
        top_worker = worker_map[top_id]
        bot_worker = worker_map[bot_id]

        if pbt_config.exploit_strategy == "pb2":
            new_hparams = _perturb_pb2(all_obs, specs, n_candidates=64, rng=rng)
        else:
            new_hparams = _perturb_classic(
                top_worker.ppo_config,
                specs,
                pbt_config.perturbation_factors,
                pbt_config.resample_probability,
                pbt_config.link_lr_fields,
                rng,
            )

        new_config = _apply_hparams(top_worker.ppo_config, new_hparams)
        bot_worker.ppo_config = new_config
        bot_worker.checkpoint_path = top_worker.checkpoint_path
        bot_worker.parent_id = top_id
        bot_worker.generation += 1
        bot_worker.hparam_history.append(new_hparams)
        exploited_pairs.append((bot_id, top_id))
        replaced_ids.add(bot_id)

    best_id = sorted_ids[0]
    gen_result = GenerationResult(
        generation=generation,
        worker_fitnesses=fitnesses,
        exploited_pairs=exploited_pairs,
        best_worker_id=best_id,
        best_fitness=fitnesses[best_id],
        worker_hparams={w.worker_id: _current_hparams(w.ppo_config, specs) for w in workers},
    )
    return workers, gen_result, replaced_ids


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _print_cycle_summary(
    workers: list[WorkerState],
    gen_result: GenerationResult,
    specs: list[HparamSpec],
) -> None:
    primary = specs[0].field if specs else "fitness"
    parts = []
    for w in sorted(workers, key=lambda x: x.worker_id):
        fit = gen_result.worker_fitnesses.get(w.worker_id, float("-inf"))
        lr_val = gen_result.worker_hparams.get(w.worker_id, {}).get(primary, float("nan"))
        parts.append(f"w{w.worker_id} perf={fit:.3f} {primary}={lr_val:.2e}")
    print(f"[PBT gen={gen_result.generation:03d}] {' '.join(parts)}")
    if gen_result.exploited_pairs:
        exploit_strs = []
        for bot_id, top_id in gen_result.exploited_pairs:
            new_val = gen_result.worker_hparams.get(bot_id, {}).get(primary, float("nan"))
            exploit_strs.append(f"w{bot_id}→w{top_id} (new_{primary}={new_val:.2e})")
        best_id = gen_result.best_worker_id
        best_fit = gen_result.best_fitness
        print(
            f"[PBT gen={gen_result.generation:03d}] exploit: {', '.join(exploit_strs)}"
            f"  best=w{best_id} perf={best_fit:.3f}"
        )


def _write_pbt_log(path: Path, generations: list[GenerationResult], workers: list[WorkerState]) -> None:
    """Persist a JSON log of all generations for offline analysis."""
    log: dict[str, Any] = {
        "generations": [
            {
                "generation": g.generation,
                "worker_fitnesses": g.worker_fitnesses,
                "exploited_pairs": g.exploited_pairs,
                "best_worker_id": g.best_worker_id,
                "best_fitness": g.best_fitness,
                "worker_hparams": g.worker_hparams,
            }
            for g in generations
        ],
        "workers": [
            {
                "worker_id": w.worker_id,
                "iteration": w.iteration,
                "fitness_history": w.fitness_history,
                "generation": w.generation,
                "parent_id": w.parent_id,
                "hparam_history": w.hparam_history,
            }
            for w in workers
        ],
    }
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def _apply_ppo_hparams_to_trainer(config: ConfigPPO, trainer: Any, specs: list[HparamSpec]) -> None:
    """Apply PPO config fields to a live trainer in-place (no graph rebuild)."""
    for spec in specs:
        val = float(getattr(config, spec.field))
        if spec.field == "actor_lr":
            trainer.set_actor_lr(val)
        elif spec.field == "critic_lr":
            trainer.set_critic_lr(val)
        elif spec.field == "entropy_coeff":
            trainer.set_entropy_coeff(val)
        elif spec.field == "mirror_loss_coeff":
            trainer.set_mirror_loss_coeff(val)
        # For all fields: also update trainer.config so the value is visible
        # to code that reads it as a Python float (e.g. reward_clip in GAE,
        # clip_ratio in the next graph capture).
        setattr(trainer.config, spec.field, val)



def population_based_train(
    train_fn: Callable[..., Any],
    make_config_fn: Callable[[ConfigPPO, str | None, bool, int, int, str], Any],
    get_history_fn: Callable[[Any], list[Any]],
    pbt_config: ConfigPBT,
    hparam_specs: list[HparamSpec],
    initial_ppo_config: ConfigPPO,
    *,
    output_dir: str | Path,
    continue_fn: Callable[[Any, Any], Any] | None = None,
    get_trainer_fn: Callable[[Any], Any] | None = None,
) -> ResultPBT:
    """Run Population Based Training over a set of parallel workers.

    Each cycle trains every worker for ``pbt_config.exploit_interval``
    iterations, then truncation-selects the bottom fraction to inherit
    checkpoints and perturbed hyper-parameters from the top fraction.

    When *continue_fn* and *get_trainer_fn* are provided, live workers are
    reused across cycles (no env or CUDA graph rebuild).  Hyper-parameter
    changes are applied in-place via the trainer's setter methods.

    Args:
        train_fn: Function such as :func:`train_g1_ppo` that accepts a config
            and returns a result with ``.history``.
        make_config_fn: Callable that constructs the task-specific train
            config.  Signature:
            ``(ppo_config, resume_checkpoint, resume_policy_only, seed,
            iterations, checkpoint_path) -> config``.
        get_history_fn: Extracts the per-iteration stats list from a train
            result.
        pbt_config: PBT hyper-parameters.
        hparam_specs: Which fields of :class:`ConfigPPO` to optimise.
        initial_ppo_config: Starting PPO config; worker 0 keeps it unchanged,
            workers 1..N-1 get sampled perturbations.
        output_dir: Directory for checkpoints, logs, and the best checkpoint.
        continue_fn: Optional callable ``(result, cycle_config) -> result``
            that continues training a live worker without rebuilding the env.
        get_trainer_fn: Optional callable ``(result) -> TrainerPPO`` that
            extracts the trainer from a result for in-place hparam updates.

    Returns:
        :class:`ResultPBT` with all worker states and generation summaries.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(pbt_config.seed)

    # Sample initial hparams for each worker
    worker_ppo_configs: list[ConfigPPO] = []
    for wid in range(pbt_config.population_size):
        if wid == 0:
            cfg = initial_ppo_config
        else:
            hparams = _sample_hparams(hparam_specs, rng)
            cfg = _apply_hparams(initial_ppo_config, hparams)
        worker_ppo_configs.append(cfg)

    workers: list[WorkerState] = [
        WorkerState(
            worker_id=wid,
            ppo_config=worker_ppo_configs[wid],
            checkpoint_path=None,
            hparam_history=[_current_hparams(worker_ppo_configs[wid], hparam_specs)],
        )
        for wid in range(pbt_config.population_size)
    ]

    # Live results: None until first cycle initialises the worker
    live_results: list[Any | None] = [None] * pbt_config.population_size

    generations: list[GenerationResult] = []
    best_fitness = float("-inf")
    best_checkpoint = str(out / "best.npz")
    best_worker_id = 0
    replace_with_policy_only: set[int] = set()

    for cycle in range(pbt_config.total_cycles):
        for worker in workers:
            ckpt_out = str(out / f"w{worker.worker_id}_gen{cycle:03d}.npz")
            wid = worker.worker_id

            if live_results[wid] is None:
                # First time: full initialisation (builds env + CUDA graph)
                resume_policy_only = False
                worker_seed = pbt_config.seed + wid * 1_000_000
                train_config = make_config_fn(
                    worker.ppo_config,
                    None,
                    resume_policy_only,
                    worker_seed,
                    pbt_config.exploit_interval,
                    ckpt_out,
                )
                live_results[wid] = train_fn(train_config)
            else:
                result = live_results[wid]
                trainer = get_trainer_fn(result) if get_trainer_fn is not None else None
                if trainer is not None and wid in replace_with_policy_only:
                    # Find the top donor result and copy weights on GPU
                    donor_id = worker.parent_id
                    if donor_id is not None and live_results[donor_id] is not None:
                        donor_trainer = get_trainer_fn(live_results[donor_id])
                        if donor_trainer is not None:
                            trainer.copy_weights_from(donor_trainer)
                    # Apply new hparams to trainer in-place
                    _apply_ppo_hparams_to_trainer(worker.ppo_config, trainer, hparam_specs)
                if continue_fn is not None:
                    # Build a cycle config with updated ppo_config and iteration count
                    cycle_config = make_config_fn(
                        worker.ppo_config,
                        None,  # no checkpoint needed — live state
                        False,
                        pbt_config.seed + wid * 1_000_000 + cycle * 1_000,
                        pbt_config.exploit_interval,
                        ckpt_out,
                    )
                    live_results[wid] = continue_fn(live_results[wid], cycle_config)
                else:
                    # Fallback: checkpoint-based (for tasks without continue_fn)
                    resume_policy_only = wid in replace_with_policy_only
                    worker_seed = pbt_config.seed + wid * 1_000_000 + cycle * 1_000
                    train_config = make_config_fn(
                        worker.ppo_config,
                        worker.checkpoint_path,
                        resume_policy_only,
                        worker_seed,
                        pbt_config.exploit_interval,
                        ckpt_out,
                    )
                    live_results[wid] = train_fn(train_config)

            hist = get_history_fn(live_results[wid])
            worker.history.extend(hist)
            worker.iteration += pbt_config.exploit_interval

            # Save checkpoint (for logging / best tracking)
            if get_trainer_fn is not None:
                t = get_trainer_fn(live_results[wid])
                if t is not None:
                    t.save_checkpoint(ckpt_out, iteration=worker.iteration)
            worker.checkpoint_path = ckpt_out

            cycle_fitness = _compute_fitness(hist, pbt_config.fitness_window, pbt_config.fitness_metric)
            worker.fitness_history.append(cycle_fitness)

        workers, gen_result, replaced_ids = _exploit_explore(
            workers, hparam_specs, pbt_config, rng, out, cycle
        )
        generations.append(gen_result)

        if pbt_config.fresh_optimizer_on_exploit:
            replace_with_policy_only = replaced_ids
        else:
            replace_with_policy_only = set()

        if gen_result.best_fitness > best_fitness:
            best_fitness = gen_result.best_fitness
            best_worker_id = gen_result.best_worker_id
            src = workers[gen_result.best_worker_id].checkpoint_path
            if src is not None and Path(src).exists():
                shutil.copy2(src, best_checkpoint)

        if pbt_config.log_interval > 0 and cycle % pbt_config.log_interval == 0:
            _print_cycle_summary(workers, gen_result, hparam_specs)

        _write_pbt_log(out / "pbt_log.json", generations, workers)

    return ResultPBT(
        best_worker_id=best_worker_id,
        best_checkpoint=best_checkpoint,
        best_fitness=best_fitness,
        workers=workers,
        generations=generations,
        output_dir=str(out),
    )


# ---------------------------------------------------------------------------
# Default hyperparameter specs
# ---------------------------------------------------------------------------


def default_g1_hparam_specs() -> list[HparamSpec]:
    """Default perturbable hyper-parameters for the G1 walking task."""
    return [
        HparamSpec("actor_lr", "log_uniform", 1e-4, 5e-2),
        HparamSpec("entropy_coeff", "log_uniform", 1e-6, 1e-2),
        HparamSpec("mirror_loss_coeff", "uniform", 0.0, 0.5),
        HparamSpec("max_grad_norm", "log_uniform", 0.1, 1.0),
        HparamSpec("reward_clip", "uniform", 0.5, 5.0),
    ]


def default_anymal_hparam_specs() -> list[HparamSpec]:
    """Default perturbable hyper-parameters for the Anymal walking task."""
    return [
        HparamSpec("actor_lr", "log_uniform", 1e-4, 5e-3),
        HparamSpec("entropy_coeff", "log_uniform", 1e-6, 1e-2),
        HparamSpec("max_grad_norm", "log_uniform", 0.1, 1.0),
    ]


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def population_based_train_g1(
    base_config: ConfigTrainG1PPO,
    pbt_config: ConfigPBT | None = None,
    hparam_specs: list[HparamSpec] | None = None,
    *,
    output_dir: str | Path,
) -> ResultPBT:
    """Run PBT for the G1 walking task.

    Args:
        base_config: Template :class:`ConfigTrainG1PPO`; all fields except
            ``ppo_config``, ``resume_checkpoint``, ``resume_policy_only``,
            ``seed``, ``iterations``, and ``checkpoint_path`` are inherited by
            every worker.
        pbt_config: PBT hyper-parameters; defaults to :class:`ConfigPBT`.
        hparam_specs: Which PPO fields to search; defaults to
            :func:`default_g1_hparam_specs`.
        output_dir: Directory for checkpoints and logs.

    Returns:
        :class:`ResultPBT` with all worker states.
    """
    from .training import _train_g1_ppo_cycle, train_g1_ppo

    pbt = pbt_config or ConfigPBT()
    specs = hparam_specs or default_g1_hparam_specs()
    initial_ppo = base_config.ppo_config

    if initial_ppo is None:
        from .g1_recipe import default_g1_ppo_config

        initial_ppo = default_g1_ppo_config()

    def make_config(
        ppo_config: ConfigPPO,
        resume_checkpoint: str | None,
        resume_policy_only: bool,
        seed: int,
        iterations: int,
        checkpoint_path: str,
    ) -> ConfigTrainG1PPO:
        return dataclasses.replace(
            base_config,
            ppo_config=ppo_config,
            resume_checkpoint=resume_checkpoint,
            resume_policy_only=resume_policy_only,
            seed=seed,
            iterations=iterations,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=0,
        )

    def get_history(result: ResultTrainG1PPO) -> list[Any]:
        return result.history

    def continue_training(result: ResultTrainG1PPO, cycle_cfg: ConfigTrainG1PPO) -> ResultTrainG1PPO:
        return _train_g1_ppo_cycle(result, cycle_cfg)

    def get_trainer(result: ResultTrainG1PPO) -> TrainerPPO:
        return result.trainer

    return population_based_train(
        train_fn=train_g1_ppo,
        make_config_fn=make_config,
        get_history_fn=get_history,
        pbt_config=pbt,
        hparam_specs=specs,
        initial_ppo_config=initial_ppo,
        output_dir=output_dir,
        continue_fn=continue_training,
        get_trainer_fn=get_trainer,
    )


def population_based_train_anymal(
    base_config: ConfigTrainAnymalPPO,
    pbt_config: ConfigPBT | None = None,
    hparam_specs: list[HparamSpec] | None = None,
    *,
    output_dir: str | Path,
) -> ResultPBT:
    """Run PBT for the Anymal walking task.

    Args:
        base_config: Template :class:`ConfigTrainAnymalPPO`; all fields except
            ``ppo_config``, ``resume_checkpoint``, ``resume_policy_only``,
            ``seed``, ``iterations``, and ``checkpoint_path`` are inherited by
            every worker.
        pbt_config: PBT hyper-parameters; defaults to :class:`ConfigPBT`.
        hparam_specs: Which PPO fields to search; defaults to
            :func:`default_anymal_hparam_specs`.
        output_dir: Directory for checkpoints and logs.

    Returns:
        :class:`ResultPBT` with all worker states.
    """
    from .training import _default_ppo_config, train_anymal_ppo

    pbt = pbt_config or ConfigPBT()
    specs = hparam_specs or default_anymal_hparam_specs()
    initial_ppo = base_config.ppo_config

    if initial_ppo is None:
        initial_ppo = _default_ppo_config()

    def make_config(
        ppo_config: ConfigPPO,
        resume_checkpoint: str | None,
        resume_policy_only: bool,
        seed: int,
        iterations: int,
        checkpoint_path: str,
    ) -> ConfigTrainAnymalPPO:
        return dataclasses.replace(
            base_config,
            ppo_config=ppo_config,
            resume_checkpoint=resume_checkpoint,
            resume_policy_only=resume_policy_only,
            seed=seed,
            iterations=iterations,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=0,
        )

    def get_history(result: ResultTrainAnymalPPO) -> list[Any]:
        return result.history

    return population_based_train(
        train_fn=train_anymal_ppo,
        make_config_fn=make_config,
        get_history_fn=get_history,
        pbt_config=pbt,
        hparam_specs=specs,
        initial_ppo_config=initial_ppo,
        output_dir=output_dir,
    )
