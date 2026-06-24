# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import os
import warnings
from typing import Any

import warp as wp


def _require_onnx():
    """Lazy import of the ``onnx`` package with a friendly error message."""
    try:
        import onnx  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only on missing dep
        raise ImportError(
            "Loading neural-controller ONNX checkpoints requires the optional `onnx` package. "
            "Install it with `pip install newton[onnx]`."
        ) from exc
    return onnx


def _require_warp_nn_runtime():
    """Lazy import of the Warp-NN ONNX runtime."""
    try:
        from warp_nn.runtime import OnnxRuntime  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only on missing dep
        raise ImportError(
            "Loading neural-controller ONNX checkpoints requires Warp-NN's ONNX runtime. "
            "Install it with `pip install newton[onnx]`."
        ) from exc
    return OnnxRuntime


def _looks_like_torch_checkpoint(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in (".pt", ".pth")


def load_metadata(path: str) -> dict[str, Any]:
    """Load only the metadata dict from a checkpoint.

    ONNX metadata is read from the model's ``metadata_props`` field.  A single
    property named ``metadata`` with a JSON-encoded value is preferred;
    otherwise individual ``key/value`` properties are returned verbatim.

    Args:
        path: File path to the checkpoint.

    Returns:
        Metadata mapping (empty dict when no metadata is stored).
    """
    if _looks_like_torch_checkpoint(path):
        return _load_torch_metadata(path)
    onnx = _require_onnx()
    model = onnx.load(path, load_external_data=False)
    return _extract_metadata(model, path)


def load_checkpoint(
    path: str,
    device: str | wp.Device | None = None,
    batch_size: int = 1,
    input_batch_axes: int | dict[str, int] | None = None,
):
    """Load a neural-network checkpoint as ``(runtime, metadata)``.

    Both ONNX (``.onnx``) and TorchScript (``.pt`` / ``.pth``) checkpoints
    are accepted.  TorchScript loading is deprecated: it emits a
    :class:`DeprecationWarning` and will be removed in a future release.
    Convert legacy ``.pt`` policies to ``.onnx`` once with
    ``torch.onnx.export(...)``.

    Args:
        path: File path to the checkpoint.  ``.onnx`` is preferred;
            ``.pt`` / ``.pth`` is accepted for backward compatibility.
        device: Warp device string (e.g. ``"cuda:0"``).  ``None`` uses the
            current default device.
        batch_size: Fixed batch dimension used to pre-allocate intermediate
            buffers.
        input_batch_axes: Optional ONNX graph-input batch-axis override passed
            to :class:`warp_nn.runtime.OnnxRuntime`.

    Returns:
        ``(runtime, metadata)`` where *runtime* is callable as
        ``runtime({input_name: warp_array})`` and exposes ``input_names`` /
        ``output_names`` lists, and *metadata* is a configuration dict.
    """
    if _looks_like_torch_checkpoint(path):
        return _load_torch_checkpoint(path, device=device)

    metadata = load_metadata(path)
    OnnxRuntime = _require_warp_nn_runtime()
    runtime = OnnxRuntime(path, device=device, batch_size=batch_size, input_batch_axes=input_batch_axes)
    return runtime, metadata


def _extract_metadata(model, path: str) -> dict[str, Any]:
    props = {p.key: p.value for p in model.metadata_props}
    if "metadata" in props:
        try:
            metadata = json.loads(props["metadata"])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in ONNX metadata property 'metadata' for '{path}'") from exc
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Invalid ONNX metadata property 'metadata' for '{path}': expected a JSON object, "
                f"got {type(metadata).__name__}"
            )
        return metadata
    parsed: dict[str, Any] = {}
    for k, v in props.items():
        try:
            parsed[k] = json.loads(v)
        except json.JSONDecodeError:
            parsed[k] = v
    return parsed


def _parse_metadata_scale(
    metadata: dict[str, Any],
    key: str,
    model_path: str,
    default: float = 1.0,
    fallback_key: str | None = None,
) -> float:
    value_key = key
    value = default
    if key in metadata:
        value = metadata[key]
    elif fallback_key is not None and fallback_key in metadata:
        value_key = fallback_key
        value = metadata[fallback_key]

    try:
        scale = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid metadata value for '{value_key}' in '{model_path}': expected a finite non-zero number, got {value!r}"
        ) from exc
    if not math.isfinite(scale) or scale == 0.0:
        raise ValueError(
            f"Invalid metadata value for '{value_key}' in '{model_path}': expected a finite non-zero number, got {value!r}"
        )
    return scale


def _runtime_shape(runtime, name: str) -> tuple[int, ...]:
    """Return a runtime tensor shape while isolating Warp-NN private access."""
    shapes = getattr(runtime, "_shapes", None)
    if shapes is None:
        raise AttributeError(
            f"{type(runtime).__name__} does not expose tensor shapes; update Warp-NN to provide shape metadata"
        )
    if name not in shapes:
        raise ValueError(f"{type(runtime).__name__} has no shape for tensor '{name}'; available tensors: {sorted(shapes)}")
    return tuple(shapes[name])


_TORCH_DEPRECATION_MSG = (
    "Loading neural-controller checkpoints from TorchScript .pt/.pth files is deprecated "
    "and will be removed in a future release. Convert your checkpoint to ONNX once "
    "(see torch.onnx.export) and load the .onnx file instead."
)


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Loading legacy .pt/.pth checkpoints requires PyTorch. "
            "Install it (e.g. `pip install newton[torch-cu12]`) or convert the "
            "checkpoint to ONNX (`torch.onnx.export`) and load the .onnx file."
        ) from exc
    return torch


def _load_torch_raw(path: str) -> tuple[Any, dict[str, Any]]:
    """Load a legacy ``.pt`` / ``.pth`` checkpoint."""
    torch = _require_torch()

    extra_files: dict[str, str] = {"metadata.json": ""}
    try:
        net = torch.jit.load(path, map_location="cpu", _extra_files=extra_files)
        meta = json.loads(extra_files["metadata.json"]) if extra_files["metadata.json"] else {}
        return net, meta
    except Exception:
        pass

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        meta = checkpoint.get("metadata", {})
        if "model" not in checkpoint:
            raise ValueError(f"Cannot load checkpoint at '{path}'; dict checkpoint has no 'model' key")
        return checkpoint["model"], meta

    raise ValueError(f"Cannot load checkpoint at '{path}'; expected a TorchScript archive or a dict with a 'model' key")


def _load_torch_metadata(path: str) -> dict[str, Any]:
    warnings.warn(_TORCH_DEPRECATION_MSG, DeprecationWarning, stacklevel=3)
    _, metadata = _load_torch_raw(path)
    return metadata


def _load_torch_checkpoint(path: str, device: str | wp.Device | None = None):
    """Wrap a TorchScript / dict checkpoint in an ONNX-runtime-compatible adapter."""
    warnings.warn(_TORCH_DEPRECATION_MSG, DeprecationWarning, stacklevel=3)
    model, metadata = _load_torch_raw(path)
    if hasattr(model, "eval"):
        model = model.eval()
    return _TorchModuleAdapter(model, device=device), metadata


def _load_legacy_lstm_torch_checkpoint(path: str, device: str | wp.Device | None = None):
    """Load a legacy ``.pt`` LSTM checkpoint and wrap it for the ONNX-shaped controller.

    Args:
        path: File path to the legacy checkpoint.
        device: Warp device where output arrays should be allocated.

    Returns:
        ``(adapter, metadata)`` with metadata keys required by
        :class:`ControllerNeuralLSTM`.
    """
    warnings.warn(_TORCH_DEPRECATION_MSG, DeprecationWarning, stacklevel=3)
    model, metadata = _load_torch_raw(path)
    if hasattr(model, "eval"):
        model = model.eval()
    if not hasattr(model, "lstm"):
        raise ValueError(
            f"Legacy .pt LSTM checkpoint at '{path}' must expose a 'lstm' attribute (torch.nn.LSTM); "
            "re-export to ONNX or supply a compatible checkpoint."
        )
    lstm = model.lstm
    if not hasattr(lstm, "num_layers"):
        raise ValueError("Legacy .pt LSTM checkpoint: network.lstm must be a torch.nn.LSTM (missing num_layers)")
    if not getattr(lstm, "batch_first", False):
        raise ValueError("Legacy .pt LSTM checkpoint: network.lstm.batch_first must be True")
    if getattr(lstm, "input_size", None) != 2:
        raise ValueError(
            f"Legacy .pt LSTM checkpoint: network.lstm.input_size must be 2 (pos_error, vel_error); "
            f"got {lstm.input_size}"
        )
    if getattr(lstm, "bidirectional", False):
        raise ValueError("Legacy .pt LSTM checkpoint: network.lstm must not be bidirectional")
    if getattr(lstm, "proj_size", 0) != 0:
        raise ValueError(f"Legacy .pt LSTM checkpoint: network.lstm.proj_size must be 0; got {lstm.proj_size}")

    legacy_meta = dict(metadata) if metadata else {}
    legacy_meta.setdefault("input_name", "observation")
    legacy_meta.setdefault("hidden_in_name", "hidden_in")
    legacy_meta.setdefault("cell_in_name", "cell_in")
    legacy_meta.setdefault("output_name", "effort")
    legacy_meta.setdefault("hidden_out_name", "hidden_out")
    legacy_meta.setdefault("cell_out_name", "cell_out")
    legacy_meta["num_layers"] = int(lstm.num_layers)
    legacy_meta["hidden_size"] = int(lstm.hidden_size)
    return _LegacyLstmTorchAdapter(model, legacy_meta, device=device), legacy_meta


class _LegacyLstmTorchAdapter:
    """Legacy ``.pt`` LSTM adapter exposing the ONNX-runtime interface."""

    def __init__(self, model, metadata: dict[str, Any], device: str | wp.Device | None = None):
        torch = _require_torch()
        self._torch = torch
        self._model = model
        self._device = device
        self._input_name: str = metadata["input_name"]
        self._hidden_in_name: str = metadata["hidden_in_name"]
        self._cell_in_name: str = metadata["cell_in_name"]
        self._output_name: str = metadata["output_name"]
        self._hidden_out_name: str = metadata["hidden_out_name"]
        self._cell_out_name: str = metadata["cell_out_name"]
        self._num_layers = int(metadata["num_layers"])
        self._hidden_size = int(metadata["hidden_size"])
        self.input_names: list[str] = [self._input_name, self._hidden_in_name, self._cell_in_name]
        self.output_names: list[str] = [self._output_name, self._hidden_out_name, self._cell_out_name]
        self._torch_device = self._resolve_torch_device(device)
        self._model = self._model.to(self._torch_device)
        self._shapes: dict[str, tuple[int, ...]] = {}

    def to(self, device: str | wp.Device | None):
        self._device = device
        self._torch_device = self._resolve_torch_device(device)
        self._model = self._model.to(self._torch_device)
        return self

    def _resolve_torch_device(self, device: str | wp.Device | None):
        torch = self._torch
        device_str = str(device) if device is not None else "cpu"
        if device_str == "cpu":
            return torch.device("cpu")
        if device_str.startswith("cuda") and torch.cuda.is_available():
            return torch.device(device_str)
        return torch.device("cpu")

    def _to_torch(self, arr):
        torch = self._torch
        np_arr = arr.numpy() if hasattr(arr, "numpy") else arr
        return torch.as_tensor(np_arr, device=self._torch_device)

    def __call__(self, inputs):
        torch = self._torch
        if self._input_name not in inputs:
            raise KeyError(f"_LegacyLstmTorchAdapter: missing input '{self._input_name}'")
        if self._hidden_in_name not in inputs:
            raise KeyError(f"_LegacyLstmTorchAdapter: missing input '{self._hidden_in_name}'")
        if self._cell_in_name not in inputs:
            raise KeyError(f"_LegacyLstmTorchAdapter: missing input '{self._cell_in_name}'")

        x = self._to_torch(inputs[self._input_name])
        if x.dim() == 3 and x.shape[0] == 1:
            x = x.transpose(0, 1).contiguous()
        h = self._to_torch(inputs[self._hidden_in_name])
        c = self._to_torch(inputs[self._cell_in_name])

        with torch.inference_mode():
            effort, (h_new, c_new) = self._model(x, (h, c))
        if isinstance(effort, (tuple, list)):
            effort = effort[0]

        effort_wp = wp.array(effort.detach().cpu().numpy(), dtype=wp.float32, device=self._device)
        h_wp = wp.array(h_new.detach().cpu().numpy(), dtype=wp.float32, device=self._device)
        c_wp = wp.array(c_new.detach().cpu().numpy(), dtype=wp.float32, device=self._device)

        self._shapes[self._output_name] = tuple(effort_wp.shape)
        self._shapes[self._hidden_out_name] = tuple(h_wp.shape)
        self._shapes[self._cell_out_name] = tuple(c_wp.shape)

        return {
            self._output_name: effort_wp,
            self._hidden_out_name: h_wp,
            self._cell_out_name: c_wp,
        }


class _TorchModuleAdapter:
    """Adapter that exposes a Torch module via the ONNX-runtime interface."""

    def __init__(self, model, device: str | wp.Device | None = None):
        torch = _require_torch()
        self._torch = torch
        self._model = model
        self._device = device
        self.input_names: list[str] = ["observation"]
        self.output_names: list[str] = ["action"]
        self._shapes: dict[str, tuple[int, ...]] = {}

    def __call__(self, inputs):
        torch = self._torch
        if len(inputs) != 1:
            raise NotImplementedError(
                "_TorchModuleAdapter only supports single-input MLP-shaped policies "
                f"(got {len(inputs)} inputs: {sorted(inputs)}). Stateful controllers "
                "such as ControllerNeuralLSTM should be exported to ONNX."
            )
        in_name = self.input_names[0]
        if in_name not in inputs:
            raise KeyError(f"_TorchModuleAdapter: missing input '{in_name}'")
        arr = inputs[in_name]
        x_np = arr.numpy() if hasattr(arr, "numpy") else arr
        x = torch.as_tensor(x_np)
        with torch.no_grad():
            y = self._model(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        out_name = self.output_names[0]
        out = wp.array(y.detach().cpu().numpy(), dtype=wp.float32, device=self._device)
        self._shapes[out_name] = tuple(out.shape)
        return {out_name: out}
