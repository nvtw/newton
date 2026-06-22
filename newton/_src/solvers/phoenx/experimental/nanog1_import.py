# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Import nanoG1 PufferNet binaries into PhoenX PPO checkpoints.

This module is intentionally experimental. It lets PhoenX use the shipped
nanoG1 policy as a teacher or warm-start without adding PyTorch to the core RL
stack. The binary parser follows PufferLib src/puffernet.h exactly: tensors are
read from a float32 stream and the read index is advanced to the next 8-float
boundary after each tensor. The shipped nanoG1 binary relies on the same padded
read behavior as the C inference path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.rl_training import g1_recipe


@dataclass(frozen=True)
class PufferNetWeights:
    """PufferNet tensors converted to PhoenX row-major layout."""

    encoder_weight: np.ndarray
    decoder_weight: np.ndarray
    log_std: np.ndarray
    recurrent_weights: tuple[np.ndarray, ...]
    raw_float_count: int
    aligned_float_count: int


def _align8(index: int) -> int:
    return (int(index) + 7) & ~7


def _read_aligned(raw: np.ndarray, index: int, count: int) -> tuple[np.ndarray, int]:
    values = np.zeros(int(count), dtype=np.float32)
    available = max(0, min(int(count), int(raw.shape[0]) - int(index)))
    if available > 0:
        values[:available] = raw[index : index + available]
    index = _align8(index + int(count))
    return values, index


def load_puffernet_weights(
    path: str | Path,
    *,
    input_dim: int,
    hidden_size: int,
    action_dim: int,
    num_layers: int,
) -> PufferNetWeights:
    """Load a PufferNet float binary and transpose tensors for PhoenX.

    Args:
        path: Input PufferNet binary written as float32 values.
        input_dim: Observation dimension.
        hidden_size: Recurrent hidden width.
        action_dim: Continuous action dimension.
        num_layers: Number of MinGRU layers.

    Returns:
        Parsed tensors in PhoenX layout: encoder ``[input, hidden]``, decoder
        ``[hidden, action + value]``, and recurrent ``[hidden, 3 * hidden]``.
    """

    raw = np.fromfile(path, dtype=np.float32)
    index = 0
    encoder_flat, index = _read_aligned(raw, index, int(hidden_size) * int(input_dim))
    decoder_flat, index = _read_aligned(raw, index, (int(action_dim) + 1) * int(hidden_size))
    log_std, index = _read_aligned(raw, index, int(action_dim))
    recurrent = []
    for _ in range(int(num_layers)):
        flat, index = _read_aligned(raw, index, 3 * int(hidden_size) * int(hidden_size))
        recurrent.append(flat.reshape(3 * int(hidden_size), int(hidden_size)).T.copy())

    return PufferNetWeights(
        encoder_weight=encoder_flat.reshape(int(hidden_size), int(input_dim)).T.copy(),
        decoder_weight=decoder_flat.reshape(int(action_dim) + 1, int(hidden_size)).T.copy(),
        log_std=log_std.copy(),
        recurrent_weights=tuple(recurrent),
        raw_float_count=int(raw.shape[0]),
        aligned_float_count=int(index),
    )


def assign_puffernet_weights(trainer: rl.TrainerPPO, weights: PufferNetWeights) -> None:
    """Assign parsed PufferNet weights to a shared-value PufferMinGRU trainer."""

    net = trainer.actor.net
    if not isinstance(net, rl.PufferMinGRUNet):
        raise TypeError("trainer actor must use PufferMinGRUNet")
    if not trainer.config.shared_value_network:
        raise ValueError("nanoG1 PufferNet import requires shared_value_network")
    if weights.encoder_weight.shape != tuple(net.encoder_weight.shape):
        raise ValueError("encoder shape does not match trainer")
    if weights.decoder_weight.shape != tuple(net.decoder_weight.shape):
        raise ValueError("decoder shape does not match trainer")
    if weights.log_std.shape != tuple(trainer.actor.log_std.shape):
        raise ValueError("log_std shape does not match trainer")
    if len(weights.recurrent_weights) != len(net.recurrent_weights):
        raise ValueError("recurrent layer count does not match trainer")

    net.encoder_weight.assign(weights.encoder_weight)
    net.decoder_weight.assign(weights.decoder_weight)
    trainer.actor.log_std.assign(weights.log_std)
    for src, dst in zip(weights.recurrent_weights, net.recurrent_weights, strict=True):
        if src.shape != tuple(dst.shape):
            raise ValueError("recurrent shape does not match trainer")
        dst.assign(src)
    trainer.reset_rollout_state()


def make_nanog1_trainer(*, device: wp.context.Devicelike = None, seed: int = g1_recipe.SEED) -> rl.TrainerPPO:
    """Create a PhoenX PPO trainer with the nanoG1 PufferNet shape."""

    ppo_config = g1_recipe.default_g1_ppo_config(
        shared_value_network=True,
        policy_network="puffer_mingru",
        manual_actor_backward=True,
        manual_critic_backward=True,
    )
    return rl.TrainerPPO(
        obs_dim=rl.OBS_DIM_G1,
        action_dim=rl.ACTION_DIM_G1,
        hidden_layers=(128, 128, 128),
        config=ppo_config,
        device=device,
        seed=int(seed),
        squash_actions=g1_recipe.SQUASH_ACTIONS,
        activation=g1_recipe.ACTIVATION,
        log_std_init=g1_recipe.LOG_STD_INIT,
        mirror_map=rl.g1_mirror_map_ppo(),
    )


def import_nanog1_checkpoint(
    weights_path: str | Path,
    checkpoint_path: str | Path,
    *,
    device: wp.context.Devicelike = None,
    seed: int = g1_recipe.SEED,
) -> rl.TrainerPPO:
    """Import a nanoG1 binary and save a normal PhoenX PPO checkpoint."""

    trainer = make_nanog1_trainer(device=device, seed=seed)
    weights = load_puffernet_weights(
        weights_path,
        input_dim=rl.OBS_DIM_G1,
        hidden_size=128,
        action_dim=rl.ACTION_DIM_G1,
        num_layers=3,
    )
    assign_puffernet_weights(trainer, weights)
    trainer.save_checkpoint(checkpoint_path, iteration=0)
    return trainer


def puffernet_numpy_forward(
    weights: PufferNetWeights,
    obs: np.ndarray,
    *,
    state: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Reference single-step PufferNet forward for importer tests."""

    x = np.asarray(obs, dtype=np.float32) @ weights.encoder_weight
    if state is None:
        state = [np.zeros_like(x) for _ in weights.recurrent_weights]
    next_state: list[np.ndarray] = []
    for layer, weight in enumerate(weights.recurrent_weights):
        combined = x @ weight
        hidden_size = x.shape[1]
        hidden = combined[:, :hidden_size]
        gate = combined[:, hidden_size : 2 * hidden_size]
        highway = combined[:, 2 * hidden_size :]
        candidate = np.where(hidden >= 0.0, hidden + np.float32(0.5), _sigmoid(hidden))
        gate_sig = _sigmoid(gate)
        recurrent = state[layer] + gate_sig * (candidate - state[layer])
        highway_sig = _sigmoid(highway)
        x = highway_sig * recurrent + (np.float32(1.0) - highway_sig) * x
        next_state.append(recurrent.astype(np.float32))
    return x @ weights.decoder_weight, next_state


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-x, dtype=np.float32))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="/home/twidmer/Documents/git/nanoG1/assets/nanoG1.bin")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=g1_recipe.SEED)
    parser.add_argument("--json-indent", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    trainer = import_nanog1_checkpoint(args.weights, args.checkpoint, device=args.device, seed=int(args.seed))
    weights = load_puffernet_weights(
        args.weights,
        input_dim=rl.OBS_DIM_G1,
        hidden_size=128,
        action_dim=rl.ACTION_DIM_G1,
        num_layers=3,
    )
    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "weights": str(args.weights),
        "raw_float_count": weights.raw_float_count,
        "aligned_float_count": weights.aligned_float_count,
        "config": asdict(trainer.config),
    }
    print(json.dumps(result, indent=int(args.json_indent), sort_keys=True))


if __name__ == "__main__":
    main()
