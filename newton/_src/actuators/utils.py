# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

import onnx

from ..utils.onnx_runtime import OnnxRuntime


def load_metadata(path: str) -> dict[str, Any]:
    """Load only the metadata dict from an ONNX checkpoint.

    Metadata is read from the model's ``metadata_props`` field.  A single
    property named ``metadata`` with a JSON-encoded value is preferred;
    otherwise individual ``key/value`` properties are returned verbatim.

    Args:
        path: File path to the ``.onnx`` model.

    Returns:
        Metadata mapping (empty dict when no metadata is stored).
    """
    model = onnx.load(path, load_external_data=False)
    return _extract_metadata(model)


def load_checkpoint(path: str, device: str | None = None, batch_size: int = 1):
    """Load a neural-network checkpoint as ``(runtime, metadata)``.

    Args:
        path: File path to the ``.onnx`` model.
        device: Warp device string (e.g. ``"cuda:0"``).  ``None`` uses the
            current default device.
        batch_size: Fixed batch dimension used to pre-allocate intermediate
            buffers.

    Returns:
        ``(runtime, metadata)`` where *runtime* is an
        :class:`~newton.utils.OnnxRuntime` ready for inference and
        *metadata* is a configuration dict.
    """
    metadata = load_metadata(path)
    runtime = OnnxRuntime(path, device=device, batch_size=batch_size)
    return runtime, metadata


def _extract_metadata(model) -> dict[str, Any]:
    props = {p.key: p.value for p in model.metadata_props}
    if "metadata" in props:
        try:
            return json.loads(props["metadata"])
        except json.JSONDecodeError:
            pass
    parsed: dict[str, Any] = {}
    for k, v in props.items():
        try:
            parsed[k] = json.loads(v)
        except json.JSONDecodeError:
            parsed[k] = v
    return parsed
