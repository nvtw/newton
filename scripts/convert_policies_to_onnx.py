# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""One-shot converter: TorchScript ``.pt`` policies in newton-assets -> ONNX.

This script is intended to be run **once** (e.g. by Newton maintainers) so the
converted ``.onnx`` files can be committed to the ``newton-assets`` repository.
End users do not need to run it -- examples load the ``.onnx`` directly.

Each policy is loaded with :func:`torch.jit.load` (no graph rewriting), an
example observation is fabricated of the correct dimension, and
``torch.onnx.export`` writes a fresh ``.onnx`` next to the original.

Usage::

    # Activate an env that has torch + onnx installed.
    python scripts/convert_policies_to_onnx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import newton.utils

POLICY_INFO = {
    # asset_dir, relative .pt path, observation dim
    "anybotics_anymal_c": [
        ("rl_policies/anymal_walking_policy_physx.pt", 48),
        ("rl_policies/mjw_anymal.pt", 48),
        ("rl_policies/physx_anymal.pt", 48),
    ],
    "unitree_g1": [
        ("rl_policies/mjw_g1_29DOF.pt", 141),  # 12 + 3*43
        ("rl_policies/mjw_g1_23DOF.pt", 123),  # 12 + 3*37
        ("rl_policies/physx_g1_23DOF.pt", 123),
    ],
    "unitree_go2": [
        ("rl_policies/mjw_go2.pt", 48),
        ("rl_policies/physx_go2.pt", 48),
    ],
}


def _convert_one(pt_path: Path, obs_dim: int) -> Path:
    """Load .pt, export to .onnx with same name (extension swapped)."""
    import torch

    onnx_path = pt_path.with_suffix(".onnx")
    print(f"  loading {pt_path.name} ...")
    policy = torch.jit.load(str(pt_path), map_location="cpu")
    policy.eval()

    dummy = torch.zeros((1, obs_dim), dtype=torch.float32)

    # Probe the output shape for sanity
    with torch.no_grad():
        out = policy(dummy)
    if not isinstance(out, torch.Tensor):
        raise RuntimeError(f"{pt_path}: policy returned non-tensor (got {type(out)})")
    print(f"    obs_dim={obs_dim}, action_dim={int(out.shape[-1])}")

    torch.onnx.export(
        policy,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch_size"}, "action": {0: "batch_size"}},
        dynamo=False,
    )
    print(f"    wrote {onnx_path}")
    return onnx_path


def main() -> int:
    failures: list[tuple[Path, BaseException]] = []
    converted: list[Path] = []

    for asset_dir, policies in POLICY_INFO.items():
        print(f"[{asset_dir}]")
        asset_root = Path(newton.utils.download_asset(asset_dir))
        for rel_path, obs_dim in policies:
            pt_path = asset_root / rel_path
            if not pt_path.exists():
                print(f"  !! missing: {pt_path}")
                continue

            try:
                converted.append(_convert_one(pt_path, obs_dim))
            except Exception as exc:
                print(f"  !! FAILED: {pt_path}: {exc}")
                failures.append((pt_path, exc))

    print()
    print(f"converted {len(converted)} policy file(s); {len(failures)} failure(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
