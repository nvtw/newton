# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


def resize_ppo_checkpoint_inputs(
    checkpoint_path: str | Path,
    output_path: str | Path,
    new_obs_dim: int,
    *,
    input_map: Sequence[int | None] | None = None,
    fill_value: float = 0.0,
) -> Path:
    """Resize observation inputs in a saved MLP PPO checkpoint.

    The first-layer rows of the actor, critic, and optimizer state are
    remapped. Existing hidden layers, output layers, log standard deviations,
    and PPO hyperparameters are left unchanged. New inputs are initialized with
    ``fill_value`` in the weights and zero optimizer moments. A ``fill_value``
    of ``0.0`` preserves the old policy exactly when the new observation columns
    are zero.

    Args:
        checkpoint_path: Source ``.npz`` PPO checkpoint.
        output_path: Destination ``.npz`` checkpoint.
        new_obs_dim: New observation dimension.
        input_map: Optional map from each new observation column to an old
            observation column. Use ``None`` for a newly inserted input. When
            omitted, existing prefix columns are preserved and tail columns are
            inserted or dropped.
        fill_value: Weight value for newly inserted input rows.

    Returns:
        Path to the written checkpoint.

    Raises:
        ValueError: If the checkpoint is not an MLP PPO checkpoint or the map
            does not match the saved observation dimension.
    """

    data = _load_npz(checkpoint_path)
    old_obs_dim = _scalar_int(data, "obs_dim")
    target_dim = int(new_obs_dim)
    if target_dim <= 0:
        raise ValueError("new_obs_dim must be positive")
    if input_map is None:
        keep = min(old_obs_dim, target_dim)
        remap: list[int | None] = [i if i < keep else None for i in range(target_dim)]
    else:
        remap = list(input_map)
        if len(remap) != target_dim:
            raise ValueError(f"input_map length {len(remap)} does not match new_obs_dim {target_dim}")
    _remap_checkpoint_inputs(data, remap, fill_value=float(fill_value))
    data["obs_dim"] = np.asarray(target_dim, dtype=np.int64)
    return _write_npz(output_path, data)


def insert_ppo_checkpoint_inputs(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    index: int,
    count: int = 1,
    fill_value: float = 0.0,
) -> Path:
    """Insert zero-initialized observation inputs into a saved MLP PPO checkpoint.

    Args:
        checkpoint_path: Source ``.npz`` PPO checkpoint.
        output_path: Destination ``.npz`` checkpoint.
        index: New inputs are inserted before this old observation index.
        count: Number of inputs to insert.
        fill_value: Weight value for inserted input rows.

    Returns:
        Path to the written checkpoint.
    """

    data = _load_npz(checkpoint_path)
    old_obs_dim = _scalar_int(data, "obs_dim")
    insert_at = int(index)
    insert_count = int(count)
    if insert_count <= 0:
        raise ValueError("count must be positive")
    if insert_at < 0 or insert_at > old_obs_dim:
        raise ValueError(f"index must be in [0, {old_obs_dim}], got {insert_at}")
    input_map: list[int | None] = list(range(insert_at))
    input_map.extend([None] * insert_count)
    input_map.extend(range(insert_at, old_obs_dim))
    _remap_checkpoint_inputs(data, input_map, fill_value=float(fill_value))
    data["obs_dim"] = np.asarray(old_obs_dim + insert_count, dtype=np.int64)
    return _write_npz(output_path, data)


def drop_ppo_checkpoint_inputs(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    indices: Sequence[int],
) -> Path:
    """Drop observation inputs from a saved MLP PPO checkpoint.

    Args:
        checkpoint_path: Source ``.npz`` PPO checkpoint.
        output_path: Destination ``.npz`` checkpoint.
        indices: Old observation column indices to remove.

    Returns:
        Path to the written checkpoint.
    """

    data = _load_npz(checkpoint_path)
    old_obs_dim = _scalar_int(data, "obs_dim")
    drop = {int(index) for index in indices}
    if not drop:
        raise ValueError("indices must contain at least one input column")
    bad = [index for index in sorted(drop) if index < 0 or index >= old_obs_dim]
    if bad:
        raise ValueError(f"drop indices out of range for obs_dim {old_obs_dim}: {bad}")
    input_map = [old for old in range(old_obs_dim) if old not in drop]
    _remap_checkpoint_inputs(data, input_map, fill_value=0.0)
    data["obs_dim"] = np.asarray(len(input_map), dtype=np.int64)
    return _write_npz(output_path, data)


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as source:
        return {key: np.asarray(source[key]).copy() for key in source.files}


def _write_npz(path: str | Path, data: dict[str, np.ndarray]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **data)
    return output


def _scalar_int(data: dict[str, np.ndarray], key: str) -> int:
    if key not in data:
        raise ValueError(f"Checkpoint is missing required key {key!r}")
    return int(np.asarray(data[key]).item())


def _scalar_str(data: dict[str, np.ndarray], key: str, default: str) -> str:
    if key not in data:
        return default
    return str(np.asarray(data[key]).item())


def _remap_checkpoint_inputs(
    data: dict[str, np.ndarray], input_map: Sequence[int | None], *, fill_value: float
) -> None:
    old_obs_dim = _scalar_int(data, "obs_dim")
    _validate_input_map(input_map, old_obs_dim)
    policy_network = _scalar_str(data, "policy_network", _scalar_str(data, "config_policy_network", "mlp"))
    if policy_network != "mlp":
        raise ValueError(f"PPO input surgery currently supports MLP checkpoints only, got {policy_network!r}")

    _remap_mlp_first_layer(data, "actor", input_map, old_obs_dim, fill_value=fill_value)
    _remap_optimizer_first_state(data, "actor_optimizer", input_map, old_obs_dim)
    if "critic_weight_0" in data:
        _remap_mlp_first_layer(data, "critic", input_map, old_obs_dim, fill_value=fill_value)
        _remap_optimizer_first_state(data, "critic_optimizer", input_map, old_obs_dim)


def _validate_input_map(input_map: Sequence[int | None], old_obs_dim: int) -> None:
    if not input_map:
        raise ValueError("input_map must contain at least one output column")
    for new_index, old_index in enumerate(input_map):
        if old_index is None:
            continue
        old = int(old_index)
        if old < 0 or old >= old_obs_dim:
            raise ValueError(f"input_map[{new_index}]={old} is out of range for source obs_dim {old_obs_dim}")


def _remap_mlp_first_layer(
    data: dict[str, np.ndarray],
    prefix: str,
    input_map: Sequence[int | None],
    old_obs_dim: int,
    *,
    fill_value: float,
) -> None:
    network_type = _scalar_str(data, f"{prefix}_network_type", "mlp")
    if network_type != "mlp":
        raise ValueError(f"{prefix} network input surgery supports MLP only, got {network_type!r}")
    key = f"{prefix}_weight_0"
    if key not in data:
        raise ValueError(f"Checkpoint is missing required key {key!r}")
    data[key] = _remap_input_rows(data[key], input_map, old_obs_dim, fill_value=fill_value, key=key)


def _remap_optimizer_first_state(
    data: dict[str, np.ndarray], prefix: str, input_map: Sequence[int | None], old_obs_dim: int
) -> None:
    for suffix in ("m_0", "v_0"):
        key = f"{prefix}_{suffix}"
        if key in data:
            data[key] = _remap_input_rows(data[key], input_map, old_obs_dim, fill_value=0.0, key=key)


def _remap_input_rows(
    array: np.ndarray,
    input_map: Sequence[int | None],
    old_obs_dim: int,
    *,
    fill_value: float,
    key: str,
) -> np.ndarray:
    source = np.asarray(array)
    if source.ndim != 2 or int(source.shape[0]) != int(old_obs_dim):
        raise ValueError(f"{key} must have shape [obs_dim, width], got {source.shape} for obs_dim {old_obs_dim}")
    target = np.full((len(input_map), int(source.shape[1])), fill_value, dtype=source.dtype)
    for new_index, old_index in enumerate(input_map):
        if old_index is not None:
            target[new_index, :] = source[int(old_index), :]
    return target
