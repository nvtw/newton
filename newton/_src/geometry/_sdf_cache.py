# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""On-disk cache for cooked texture SDFs.

The mesh SDF cooking pipeline produces a dictionary of plain numpy arrays
(see :func:`newton._src.geometry.sdf_texture.build_sparse_sdf_from_mesh`)
just before the data is uploaded to ``wp.Texture3D`` instances.  Reading
back from a Warp 3D texture is not supported, so this cache snapshots the
data at the pre-upload boundary; on a hit, the GPU upload path
(:func:`newton._src.geometry.sdf_texture.create_sparse_sdf_textures`) runs
unchanged.

Cache layout
------------

For each cached SDF, two files are written under the user-supplied
``cache_dir`` and share a basename derived from a content hash of the
mesh and build parameters:

* ``{hash}.sdf.npz`` — uncompressed (``np.savez``) bundle of all numpy
  arrays and scalar metadata that make up the cooked sparse SDF, plus a
  reserved ``__cache_format_version__`` 0-d ``int32`` array.  The
  embedded version is the authoritative invalidator; a mismatch is
  treated as a miss and the file is overwritten on the next cook.
* ``{hash}.sdf.json`` — sidecar manifest with a human-readable schema
  description, key inputs (mesh hashes + resolved build parameters), and
  the cache format version.  The sidecar is the *first* version check
  (cheap to read), but the embedded ``.npz`` value remains
  authoritative.

Both files are written atomically via ``os.replace`` from ``*.tmp``
companions; the ``.npz`` is replaced first so that a partial write where
the sidecar is missing is detected as a miss.

Cache key
---------

The hash includes the bytes that determine the cooked output:

* Mesh: ``vertices.astype(np.float32)``, ``indices.astype(np.int32)``,
  and ``is_solid``.
* Build parameters: ``narrow_band_range``, ``target_voxel_size``,
  effective ``max_resolution``, ``margin``, ``texture_format``,
  resolved ``sign_method`` (``parity`` or ``winding``),
  ``winding_threshold`` sign (``+0.5``/``-0.5``), and ``scale``.
* The :data:`CACHE_FORMAT_VERSION` constant.

Note that ``shape_margin`` is *not* part of the key: it is applied at
sample time and is not baked into the cooked numpy dictionary, so two
shape margins can share one cache entry.

Schema versioning
-----------------

Bump :data:`CACHE_FORMAT_VERSION` whenever the cooked dictionary's
shape, dtypes, or quantization conventions change.  Existing on-disk
caches are then transparently invalidated and recooked.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


CACHE_FORMAT_VERSION: int = 1
"""Version of the on-disk cooked-SDF cache format.

Bump when the dictionary returned by
:func:`newton._src.geometry.sdf_texture.build_sparse_sdf_from_mesh`
changes shape, dtypes, or quantization meaning.  Existing cache files
become invalid and are transparently re-cooked.
"""


_VERSION_KEY = "__cache_format_version__"
_NPZ_SUFFIX = ".sdf.npz"
_JSON_SUFFIX = ".sdf.json"
_KIND = "newton.texture_sdf"


SCHEMA: dict[str, Any] = {
    "kind": _KIND,
    "cache_format_version": CACHE_FORMAT_VERSION,
    "arrays": [
        {
            "name": "coarse_sdf",
            "dtype": "float32",
            "shape": "(bg_size_z, bg_size_y, bg_size_x)",
            "description": "Background/coarse SDF samples.",
        },
        {
            "name": "subgrid_data",
            "dtype": "float32 | uint16 | uint8 (per quantization_mode)",
            "shape": "(tex_size, tex_size, tex_size)",
            "description": "Packed narrow-band subgrid texture data.",
        },
        {
            "name": "subgrid_start_slots",
            "dtype": "uint32",
            "shape": "(w, h, d)",
            "description": "Indirection from coarse cell to subgrid slot.",
        },
        {
            "name": "subgrid_required",
            "dtype": "int32",
            "shape": "(w * h * d,)",
            "description": "1D occupancy flags for non-linear subgrids.",
        },
        {
            "name": "subgrid_occupied",
            "dtype": "int32",
            "shape": "(w * h * d,)",
            "description": "1D pre-linearization narrow-band occupancy.",
        },
    ],
    "scalars": {
        "coarse_dims": "tuple[int, int, int] -- (w, h, d)",
        "subgrid_tex_size": "int -- packed subgrid texture cube edge",
        "num_subgrids": "int",
        "min_extents": "float64[3] -- AABB lower corner [m]",
        "max_extents": "float64[3] -- AABB upper corner [m]",
        "cell_size": "float64[3] -- voxel size [m]",
        "subgrid_size": "int -- cells per subgrid",
        "quantization_mode": "int -- newton._src.geometry.sdf_texture.QuantizationMode",
        "subgrids_min_sdf_value": "float -- decode origin [m]",
        "subgrids_sdf_value_range": "float -- decode range [m]",
    },
}
"""Static description of the on-disk cache schema, mirrored into the
sidecar ``.json`` manifest as ``schema``.

This constant is informational; loading does not consult it at runtime.
"""


def _digest_array(arr: np.ndarray, dtype: np.dtype) -> str:
    """SHA-256 over ``arr`` after a deterministic dtype/contiguity cast."""

    buf = np.ascontiguousarray(arr, dtype=dtype).tobytes()
    return hashlib.sha256(buf).hexdigest()


def _canonical_key_inputs(
    *,
    vertices: np.ndarray,
    indices: np.ndarray,
    is_solid: bool,
    narrow_band_range: tuple[float, float],
    target_voxel_size: float | None,
    max_resolution: int | None,
    margin: float,
    texture_format: str,
    sign_method_resolved: str,
    winding_threshold: float,
    scale: tuple[float, float, float] | None,
) -> dict[str, Any]:
    """Build the JSON-serializable key inputs for hashing and the manifest."""

    vertices_sha = _digest_array(vertices, np.dtype(np.float32))
    indices_sha = _digest_array(indices, np.dtype(np.int32))

    return {
        "mesh": {
            "vertices_sha256": vertices_sha,
            "indices_sha256": indices_sha,
            "num_vertices": int(np.asarray(vertices).reshape(-1, 3).shape[0]),
            "num_triangles": int(np.asarray(indices).size // 3),
            "is_solid": bool(is_solid),
        },
        "build_params": {
            "narrow_band_range": [float(narrow_band_range[0]), float(narrow_band_range[1])],
            "target_voxel_size": (None if target_voxel_size is None else float(target_voxel_size)),
            "max_resolution": (None if max_resolution is None else int(max_resolution)),
            "margin": float(margin),
            "texture_format": str(texture_format),
            "sign_method_resolved": str(sign_method_resolved),
            # Use sign only: the magnitude is derived from mesh orientation
            # and identical (0.5) in absolute value for the parity path.
            "winding_threshold_sign": (1 if winding_threshold >= 0.0 else -1),
            "scale": (None if scale is None else [float(scale[0]), float(scale[1]), float(scale[2])]),
        },
    }


def hash_inputs(
    *,
    vertices: np.ndarray,
    indices: np.ndarray,
    is_solid: bool,
    narrow_band_range: tuple[float, float],
    target_voxel_size: float | None,
    max_resolution: int | None,
    margin: float,
    texture_format: str,
    sign_method_resolved: str,
    winding_threshold: float,
    scale: tuple[float, float, float] | None,
) -> tuple[str, dict[str, Any]]:
    """Compute the cache key for a texture-SDF cook.

    Args:
        vertices: Mesh vertex array, ``(N, 3)``.
        indices: Mesh triangle indices, ``(M * 3,)`` or ``(M, 3)``.
        is_solid: Whether the mesh is treated as solid.
        narrow_band_range: Signed narrow-band distance range [m].
        target_voxel_size: Target voxel size [m] or ``None``.
        max_resolution: Effective maximum grid dimension [voxel] or ``None``.
        margin: Extra AABB padding [m].
        texture_format: Subgrid storage format (``"float32"``, ``"uint16"``,
            ``"uint8"``).
        sign_method_resolved: Resolved sign strategy (``"parity"`` or
            ``"winding"``).
        winding_threshold: Winding-number threshold value; only its sign
            participates in the hash.
        scale: Pre-baked vertex scale or ``None``.

    Returns:
        Tuple ``(hash_hex, key_inputs)``.  ``hash_hex`` is a 32-character
        BLAKE2b digest used as the cache filename basename.
    """

    key_inputs = _canonical_key_inputs(
        vertices=vertices,
        indices=indices,
        is_solid=is_solid,
        narrow_band_range=narrow_band_range,
        target_voxel_size=target_voxel_size,
        max_resolution=max_resolution,
        margin=margin,
        texture_format=texture_format,
        sign_method_resolved=sign_method_resolved,
        winding_threshold=winding_threshold,
        scale=scale,
    )

    payload = {
        "kind": _KIND,
        "cache_format_version": CACHE_FORMAT_VERSION,
        "key_inputs": key_inputs,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h = hashlib.blake2b(encoded, digest_size=16).hexdigest()
    return h, key_inputs


def cache_paths(cache_dir: str | os.PathLike[str], hash_hex: str) -> tuple[Path, Path]:
    """Return the ``(npz_path, json_path)`` for a given cache key."""

    base = Path(cache_dir) / hash_hex
    return base.with_suffix(_NPZ_SUFFIX), base.with_suffix(_JSON_SUFFIX)


def _tmp_sibling(path: Path) -> Path:
    """Return a sibling temporary path that preserves all existing suffixes."""

    return path.parent / (path.name + ".tmp")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = _tmp_sibling(path)
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    # ``np.savez`` appends ``.npz`` to its target. Write to a sibling
    # path that already ends in ``.npz`` so the saved file's actual name
    # matches what we ``os.replace`` afterwards.
    tmp_npz = path.parent / (path.name + ".tmp.npz")
    np.savez(tmp_npz, **arrays)
    os.replace(tmp_npz, path)


def save_sparse_data(
    cache_dir: str | os.PathLike[str],
    hash_hex: str,
    sparse_data: Mapping[str, Any],
    *,
    key_inputs: Mapping[str, Any],
    newton_version: str,
) -> tuple[Path, Path]:
    """Persist a cooked SDF dict to the cache.

    Args:
        cache_dir: Destination directory.  Created if missing.
        hash_hex: Cache key from :func:`hash_inputs`.
        sparse_data: Dictionary returned by
            :func:`newton._src.geometry.sdf_texture.build_sparse_sdf_from_mesh`.
        key_inputs: Hash inputs (second element from :func:`hash_inputs`),
            stored in the sidecar manifest for diagnostics.
        newton_version: Newton package version string for provenance.

    Returns:
        ``(npz_path, json_path)`` written.

    Raises:
        OSError: On filesystem errors.  Callers should treat any failure
            as non-fatal and fall back to live cooking.
    """

    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = cache_paths(cache_dir_path, hash_hex)

    arrays: dict[str, np.ndarray] = {
        _VERSION_KEY: np.asarray(CACHE_FORMAT_VERSION, dtype=np.int32),
    }

    array_keys = (
        "coarse_sdf",
        "subgrid_data",
        "subgrid_start_slots",
        "subgrid_required",
        "subgrid_occupied",
    )
    for k in array_keys:
        arrays[k] = np.asarray(sparse_data[k])

    coarse_dims = tuple(int(v) for v in sparse_data["coarse_dims"])
    arrays["coarse_dims"] = np.asarray(coarse_dims, dtype=np.int32)
    arrays["subgrid_tex_size"] = np.asarray(int(sparse_data["subgrid_tex_size"]), dtype=np.int32)
    arrays["num_subgrids"] = np.asarray(int(sparse_data["num_subgrids"]), dtype=np.int32)
    arrays["min_extents"] = np.asarray(sparse_data["min_extents"], dtype=np.float64).reshape(3)
    arrays["max_extents"] = np.asarray(sparse_data["max_extents"], dtype=np.float64).reshape(3)
    arrays["cell_size"] = np.asarray(sparse_data["cell_size"], dtype=np.float64).reshape(3)
    arrays["subgrid_size"] = np.asarray(int(sparse_data["subgrid_size"]), dtype=np.int32)
    arrays["quantization_mode"] = np.asarray(int(sparse_data["quantization_mode"]), dtype=np.int32)
    arrays["subgrids_min_sdf_value"] = np.asarray(float(sparse_data["subgrids_min_sdf_value"]), dtype=np.float32)
    arrays["subgrids_sdf_value_range"] = np.asarray(
        float(sparse_data["subgrids_sdf_value_range"]), dtype=np.float32
    )

    _atomic_save_npz(npz_path, arrays)

    manifest = {
        "kind": _KIND,
        "cache_format_version": CACHE_FORMAT_VERSION,
        "newton_version": str(newton_version),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hash": hash_hex,
        "key_inputs": dict(key_inputs),
        "schema": SCHEMA,
        "arrays_present": [
            {"name": name, "dtype": str(arr.dtype), "shape": list(arr.shape)}
            for name, arr in arrays.items()
        ],
        "scalars": {
            "coarse_dims": list(coarse_dims),
            "subgrid_tex_size": int(sparse_data["subgrid_tex_size"]),
            "num_subgrids": int(sparse_data["num_subgrids"]),
            "min_extents": [float(v) for v in np.asarray(sparse_data["min_extents"]).reshape(3)],
            "max_extents": [float(v) for v in np.asarray(sparse_data["max_extents"]).reshape(3)],
            "cell_size": [float(v) for v in np.asarray(sparse_data["cell_size"]).reshape(3)],
            "subgrid_size": int(sparse_data["subgrid_size"]),
            "quantization_mode": int(sparse_data["quantization_mode"]),
            "subgrids_min_sdf_value": float(sparse_data["subgrids_min_sdf_value"]),
            "subgrids_sdf_value_range": float(sparse_data["subgrids_sdf_value_range"]),
        },
    }
    encoded = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    _atomic_write_bytes(json_path, encoded)

    return npz_path, json_path


def try_load_sparse_data(
    cache_dir: str | os.PathLike[str],
    hash_hex: str,
) -> dict[str, Any] | None:
    """Load a cooked SDF dict from the cache, or ``None`` on miss.

    This first checks the sidecar ``.json`` for ``cache_format_version``
    (cheap), then verifies the embedded ``__cache_format_version__`` in
    the ``.npz`` (authoritative).  Any IO error, missing file, parse
    failure, or version mismatch is logged and treated as a miss.

    Args:
        cache_dir: Directory holding the cache files.
        hash_hex: Cache key from :func:`hash_inputs`.

    Returns:
        The reconstructed ``sparse_data`` dict suitable for
        :func:`newton._src.geometry.sdf_texture.create_sparse_sdf_textures`,
        or ``None`` if the entry is missing or invalid.
    """

    npz_path, json_path = cache_paths(cache_dir, hash_hex)

    if not npz_path.exists() or not json_path.exists():
        return None

    try:
        with open(json_path, "rb") as f:
            manifest = json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning("SDF cache: failed to read sidecar %s: %s", json_path, exc)
        return None

    sidecar_version = manifest.get("cache_format_version")
    if sidecar_version != CACHE_FORMAT_VERSION:
        logger.info(
            "SDF cache: sidecar version %r != %d, treating as miss (%s)",
            sidecar_version,
            CACHE_FORMAT_VERSION,
            json_path,
        )
        return None

    try:
        with np.load(npz_path, allow_pickle=False) as npz:
            if _VERSION_KEY not in npz.files:
                logger.info(
                    "SDF cache: missing embedded version key, treating as miss (%s)",
                    npz_path,
                )
                return None
            embedded = int(np.asarray(npz[_VERSION_KEY]).item())
            if embedded != CACHE_FORMAT_VERSION:
                logger.info(
                    "SDF cache: embedded version %d != %d, treating as miss (%s)",
                    embedded,
                    CACHE_FORMAT_VERSION,
                    npz_path,
                )
                return None

            data: dict[str, Any] = {
                "coarse_sdf": np.asarray(npz["coarse_sdf"]),
                "subgrid_data": np.asarray(npz["subgrid_data"]),
                "subgrid_start_slots": np.asarray(npz["subgrid_start_slots"]),
                "subgrid_required": np.asarray(npz["subgrid_required"]),
                "subgrid_occupied": np.asarray(npz["subgrid_occupied"]),
                "coarse_dims": tuple(int(v) for v in np.asarray(npz["coarse_dims"]).reshape(-1)),
                "subgrid_tex_size": int(np.asarray(npz["subgrid_tex_size"]).item()),
                "num_subgrids": int(np.asarray(npz["num_subgrids"]).item()),
                "min_extents": np.asarray(npz["min_extents"], dtype=np.float64).reshape(3),
                "max_extents": np.asarray(npz["max_extents"], dtype=np.float64).reshape(3),
                "cell_size": np.asarray(npz["cell_size"], dtype=np.float64).reshape(3),
                "subgrid_size": int(np.asarray(npz["subgrid_size"]).item()),
                "quantization_mode": int(np.asarray(npz["quantization_mode"]).item()),
                "subgrids_min_sdf_value": float(np.asarray(npz["subgrids_min_sdf_value"]).item()),
                "subgrids_sdf_value_range": float(np.asarray(npz["subgrids_sdf_value_range"]).item()),
            }
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("SDF cache: failed to load %s: %s", npz_path, exc)
        return None

    return data


__all__ = [
    "CACHE_FORMAT_VERSION",
    "SCHEMA",
    "cache_paths",
    "hash_inputs",
    "save_sparse_data",
    "try_load_sparse_data",
]
