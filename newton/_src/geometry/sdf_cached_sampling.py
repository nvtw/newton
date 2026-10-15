# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SDF-SDF contacts by temporal caching and importance sampling.

Implements the discrete collision method of Giles and Andrews, *Real-Time
Collision Handling for Signed Distance Fields via Caching and Importance
Sampling* (Pacific Graphics 2026), on Newton's sparse texture SDFs. For each
shape pair whose shapes both carry a texture SDF, a handful of points is
minimized inside the overlap ``V`` of the two bounding boxes against the smooth
objective (Eq. 2-3 of the paper)

.. math::

    g(x) = \\tfrac12\\left(\\phi_A + \\phi_B + \\sqrt{(\\phi_A - \\phi_B)^2 + \\varepsilon}\\right)
           - \\alpha \\sum_m \\lVert x - x_m \\rVert

with the derivative-free Nelder-Mead method. The optimizer is seeded from

* a per-pair cache of the points kept in the previous collide call, carried with
  the shapes' rigid motion (Sec. 3.5.1), and
* stochastic samples drawn from precomputed seed points on either shape's
  surface that lie inside ``V`` (Sec. 3.5.2).

Each pair yields at most ``cache_size`` contacts, so the global contact reducer
is bypassed.

**GPU layout.** Per collide call: ``prepare`` (one thread per pair: search domain,
cache lookup and advection), ``select_seeds`` (32 threads per pair scan the seed
points and pick one per stochastic lane with an order-independent atomic minimum),
``optimize`` (one thread per candidate lane), ``finalize`` (one thread per pair:
de-duplication, selection, cache update, output) and a key sort that orders the
cache for the next call's binary search.

**Deviations from the paper.** The paper's method alone, used with PhoenX, does
not bring curved shapes to rest; these changes were each needed for resting
contact (see ``Config`` for the switches that restore the paper's behavior):

* Cached points are stored in both shapes' local frames and carried with the
  shapes' poses (the average of both, or only the moving shape if the other is
  static): Eq. 6 to first order, but exact for rigid motion.
* Only the kept (cached) points become contacts. A transient sample emitted at a
  new spot every call applies impulses at random places.
* The kept points are chosen by greedy farthest-point selection among the points
  in contact, so they span the support region instead of clustering at the
  deepest point.
* Stochastic samples prefer deep seed points away from the cached ones, and each
  point is optimized within a box around its seed, so lanes resolve distinct
  contact regions.
* Cached points move only if that improves the objective by a tolerance
  (hysteresis), and the contact normal blends both surface normals weighted by
  inverse curvature, so a feature resting on a face takes the face normal.
* Lengths are relative to the pair's characteristic size ``L`` (the smaller of
  the two shapes' largest bounding-box extents), since the paper's constants
  assume unit-size models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from ..core.types import Devicelike
from .contact_data import SHAPE_PAIR_HFIELD_BIT, ContactData
from .sdf_texture import TextureSDFData, texture_sample_sdf_grad, texture_sample_sdf_hw
from .types import GeoType

# Candidate flags.
_FLAG_VALID = wp.constant(1)
_FLAG_ACTIVE = wp.constant(2)
_FLAG_CACHED = wp.constant(4)
_FLAG_FROZEN = wp.constant(8)  # cached point kept without re-optimization

# Fraction of the search box used as the initial Nelder-Mead simplex (Sec. 3.7).
_SIMPLEX_FRACTION = wp.constant(0.1)

# Threads per pair scanning seed points.
_SEED_THREADS = 128

# Sentinel of the packed (score, seed) selection keys.
_NO_SEED = wp.constant(wp.int64(0x7FFFFFFFFFFFFFFF))

# Seed scores are quantized to this fraction of L before packing [unitless].
_SCORE_RESOLUTION = wp.constant(1.0e-6)

_NORMAL_MODES = {"deeper": 0, "bisector": 1, "curvature": 2}


class SDFCachedSampling:
    """Cached, importance-sampled SDF-SDF contact generation.

    Owns the per-pair temporal cache, the per-SDF seed points, and the kernels.
    Created by :class:`~newton.CollisionPipeline` when
    ``sdf_contact_method="cached_sampling"``; it replaces the edge-sampling SDF
    path for every non-hydroelastic shape pair in which both shapes have a texture
    SDF. Other pairs on the SDF path (BVH-only meshes, heightfields) keep edge
    sampling.

    .. experimental::
    """

    @dataclass
    class Config:
        """Parameters of cached, importance-sampled SDF contacts.

        Lengths marked *relative* are multiples of the pair's characteristic size,
        the smaller of the two shapes' largest local bounding-box extents.
        """

        cache_size: int = 4
        """Points kept, and emitted as contacts, per shape pair, ``n_cache`` [count]."""
        random_samples: int = 1
        """New stochastic samples per shape pair and collide call, ``n_rand`` [count]."""
        curvature_sampling: bool = True
        """Include high-curvature seed points (Sec. 3.5.2) in the stochastic samples."""
        surface_sampling: bool = True
        """Include evenly spaced surface seed points in the stochastic samples."""
        seed_selection: str = "depth"
        """How a stochastic sample picks among the seed points inside ``V``:
        ``"uniform"`` (the paper) picks one uniformly at random; ``"depth"`` prefers deep
        points (lowest objective with Gumbel noise of scale
        ``depth_selection_temperature``) that are farther than ``exploration_radius``
        from every cached point. Without a seed a sample is uniform in ``V``."""
        depth_selection_temperature: float = 0.02
        """Relative noise scale of ``"depth"`` seed selection [unitless]."""
        exploration_radius: float = 0.1
        """Relative radius around cached points that ``"depth"`` seed selection skips
        [unitless]."""
        skip_convex_samples: bool = True
        """Skip the stochastic samples of a convex pair whose full cache is in contact.
        Two convex shapes overlap in one convex region that the cached points track."""
        softmax_smoothing: float = 0.3
        """Relative width of the soft maximum, ``sqrt(epsilon) / L`` (Eq. 2) [unitless]."""
        repulsion_weight: float = 0.01
        """Repulsion ``alpha`` between manifold points (Eq. 3) [unitless]."""
        tolerance: float = 0.01
        """Relative optimizer tolerance ``tau`` [unitless]."""
        max_iterations: int = 50
        """Nelder-Mead iteration limit per point [count]."""
        search_radius: float = 0.1
        """Relative half-size of the box around its seed in which a point is optimized;
        zero searches all of ``V`` (the paper) [unitless]."""
        reoptimize_distance: float = 0.005
        """Relative motion of the shapes at a cached point, accumulated since it was
        last optimized, below which the point is kept without re-optimization; zero
        re-optimizes every call (the paper) [unitless]."""
        hysteresis: float = 0.005
        """Relative objective improvement a cached point needs to move away from its
        carried position; zero follows the paper [unitless]."""
        normal_mode: str = "curvature"
        """Contact normal at an optimized point: ``"deeper"`` takes the gradient of the
        more deeply penetrated field (Eq. 5); ``"bisector"`` blends both surface normals
        equally; ``"curvature"`` blends them weighted by inverse squared curvature,
        favoring the flatter surface. Both blends equal Eq. 5 at an exact minimizer."""
        merge_distance: float = 0.01
        """Relative distance below which a point duplicates an earlier one (the paper
        uses ``tau``) [unitless]."""
        selection: str = "spread"
        """Which points to keep: ``"rank"`` sorts by the objective against the manifold
        (Alg. 1, line 28); ``"spread"`` keeps the deepest point in contact, then greedily
        the points in contact farthest from those kept, then the rest by depth."""
        replacement_threshold: float = 0.01
        """Relative advantage of a cached point over a new sample in the selection
        [unitless]."""
        seed_candidates: int = 8000
        """Surface samples per SDF scanned for seed points [count]."""
        seed_curvature_threshold: float = 4.0
        """Relative Laplacian threshold ``kappa * L_shape`` of curvature seeds [unitless]."""
        seed_spacing: float = 0.08
        """Relative Poisson-disk spacing of curvature seeds [unitless]."""
        max_seeds_per_sdf: int = 256
        """Upper bound on curvature seeds per SDF [count]."""
        surface_seed_spacing: float = 0.05
        """Relative Poisson-disk spacing of surface seeds [unitless]."""
        max_surface_seeds_per_sdf: int = 512
        """Upper bound on surface seeds per SDF [count]."""

        def validate(self) -> None:
            """Raise :class:`ValueError` for out-of-range parameters."""
            if self.cache_size < 1 or self.cache_size > 16:
                raise ValueError("cache_size must be in [1, 16]")
            if self.random_samples < 0 or self.cache_size + self.random_samples > 24:
                raise ValueError("random_samples must be >= 0 and cache_size + random_samples <= 24")
            if self.max_iterations < 1:
                raise ValueError("max_iterations must be positive")
            if self.seed_selection not in ("uniform", "depth"):
                raise ValueError("seed_selection must be 'uniform' or 'depth'")
            if self.selection not in ("rank", "spread"):
                raise ValueError("selection must be 'rank' or 'spread'")
            if self.normal_mode not in _NORMAL_MODES:
                raise ValueError(f"normal_mode must be one of {sorted(_NORMAL_MODES)}")
            for name in (
                "softmax_smoothing",
                "tolerance",
                "merge_distance",
                "seed_spacing",
                "seed_curvature_threshold",
                "surface_seed_spacing",
            ):
                if getattr(self, name) <= 0.0:
                    raise ValueError(f"{name} must be positive")
            for name in (
                "repulsion_weight",
                "replacement_threshold",
                "search_radius",
                "hysteresis",
                "reoptimize_distance",
                "exploration_radius",
                "depth_selection_temperature",
            ):
                if getattr(self, name) < 0.0:
                    raise ValueError(f"{name} must be non-negative")

    def __init__(
        self,
        *,
        config: SDFCachedSampling.Config,
        max_pairs: int,
        shape_count: int,
        writer_func: Any,
        texture_sdf_data: wp.array[TextureSDFData] | None = None,
        shape_body: wp.array[wp.int32] | None = None,
        speculative: bool = False,
        device: Devicelike | None = None,
    ) -> None:
        """Allocate the pair cache and build the kernels.

        Args:
            config: Method parameters.
            max_pairs: Capacity of SDF pairs per collide call [count].
            shape_count: Number of shapes, sizes the pair keys [count].
            writer_func: Contact writer ``writer_func(ContactData, writer_data, -1)``.
            texture_sdf_data: Compact texture SDF table, used to precompute seed points.
                ``None`` disables seed points (stochastic samples are uniform in ``V``).
            shape_body: Body index per shape; a shape of body ``-1`` is static and does
                not carry cached points. ``None`` treats every shape as moving.
            speculative: Whether the narrow phase admits speculative contacts, in which
                case contacts carry the authored gap and the search grows by the
                speculative extension.
            device: Allocation device.
        """
        config.validate()
        self.config = config
        self.device = device
        self.max_pairs = max(int(max_pairs), 1)
        self.speculative = speculative
        self._key_shape_bits = max(1, max(int(shape_count) - 1, 0).bit_length())
        if 2 * self._key_shape_bits + 1 > 63:
            raise ValueError("too many shapes for cached SDF sampling pair keys")
        self._key_end_bit = 2 * self._key_shape_bits + 1
        K = config.cache_size
        T = config.cache_size + config.random_samples
        self.lanes = T
        cap = self.max_pairs

        with wp.ScopedDevice(device):
            uses_seeds = config.curvature_sampling or config.surface_sampling
            if uses_seeds and texture_sdf_data is not None and texture_sdf_data.shape[0] > 0:
                self.seed_points, self.seed_range = compute_sdf_seeds(texture_sdf_data, config, device)
            else:
                n_sdf = 0 if texture_sdf_data is None else texture_sdf_data.shape[0]
                self.seed_points = wp.zeros(1, dtype=wp.vec3)
                self.seed_range = wp.zeros(max(1, n_sdf), dtype=wp.vec4i)
            self.shape_body = shape_body if shape_body is not None else wp.zeros(1, dtype=wp.int32)

            # Pair split: [cached-sampling count, edge-sampling count].
            self.counts = wp.zeros(2, dtype=wp.int32)
            self.pairs = wp.zeros(cap, dtype=wp.vec2i)
            self.rest_pairs = wp.zeros(cap, dtype=wp.vec2i)

            # Per-pair scratch.
            self.pair_key = wp.zeros(cap, dtype=wp.int64)
            self.pair_prev_slot = wp.zeros(cap, dtype=wp.int32)
            self.pair_cached = wp.zeros(cap, dtype=wp.int32)
            self.pair_lower = wp.zeros(cap, dtype=wp.vec3)
            self.pair_upper = wp.zeros(cap, dtype=wp.vec3)
            self.pair_length = wp.zeros(cap, dtype=wp.float32)
            self.anchors = wp.zeros(cap * K, dtype=wp.vec3)

            # Per-candidate scratch.
            self.lane_seed = wp.zeros(cap * T, dtype=wp.int64)
            self.cand_x = wp.zeros(cap * T, dtype=wp.vec3)
            self.cand_pa = wp.zeros(cap * T, dtype=wp.vec3)
            self.cand_pb = wp.zeros(cap * T, dtype=wp.vec3)
            self.cand_n = wp.zeros(cap * T, dtype=wp.vec3)
            self.cand_sep = wp.zeros(cap * T, dtype=wp.float32)
            self.cand_soft = wp.zeros(cap * T, dtype=wp.float32)
            self.cand_flags = wp.zeros(cap * T, dtype=wp.int32)
            self.cand_id = wp.zeros(cap * T, dtype=wp.int32)
            self.cand_rank = wp.zeros(cap * T, dtype=wp.float32)
            self.cand_order = wp.zeros(cap * T, dtype=wp.int32)
            self.iteration_count = wp.zeros(1, dtype=wp.int32)

            # Double-buffered cache: ``next`` is written in pair order, then sorted
            # by key into ``prev``. The sort needs 2x capacity scratch.
            self.next = _alloc_cache(cap, K)
            self.prev = _alloc_cache(cap, K)
            self.prev_pair_count = wp.zeros(1, dtype=wp.int32)
            self.sort_keys = wp.zeros(2 * cap, dtype=wp.int64)
            self.sort_values = wp.zeros(2 * cap, dtype=wp.int32)
            self.frame = wp.zeros(1, dtype=wp.int32)

        self._kernels = _create_kernels(
            cache_size=K,
            random_samples=config.random_samples,
            max_iterations=config.max_iterations,
            writer_func=writer_func,
            speculative=speculative,
        )

    def reset(self) -> None:
        """Drop all cached points."""
        self.prev_pair_count.zero_()

    def split_pairs(
        self,
        shape_pairs: wp.array[wp.vec2i],
        shape_pair_count: wp.array[wp.int32],
        shape_sdf_index: wp.array[wp.int32],
        texture_sdf_data: wp.array[TextureSDFData],
        device: Devicelike | None = None,
    ) -> tuple[wp.array[wp.vec2i], wp.array[wp.int32]]:
        """Take the SDF-path pairs whose shapes both have a texture SDF.

        Args:
            shape_pairs: Pairs routed to the SDF path.
            shape_pair_count: Number of routed pairs, one element.
            shape_sdf_index: Per-shape index into ``texture_sdf_data`` (-1 for none).
            texture_sdf_data: Compact texture SDF table.
            device: Launch device.

        Returns:
            The remaining pairs and their count, for the edge-sampling kernel.
        """
        self.counts.zero_()
        wp.launch(
            _split_pairs_kernel,
            dim=min(shape_pairs.shape[0], self.max_pairs),
            inputs=[shape_pairs, shape_pair_count, shape_sdf_index, texture_sdf_data],
            outputs=[self.pairs, self.rest_pairs, self.counts],
            device=device,
            record_tape=False,
        )
        return self.rest_pairs, self.counts[1:2]

    def launch(
        self,
        *,
        shape_types: wp.array[wp.int32],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_sdf_index: wp.array[wp.int32],
        texture_sdf_data: wp.array[TextureSDFData],
        shape_gap: wp.array[wp.float32],
        shape_base_gap: wp.array[wp.float32] | None,
        shape_collision_aabb_lower: wp.array[wp.vec3],
        shape_collision_aabb_upper: wp.array[wp.vec3],
        max_speculative_extension: float,
        writer_data: Any,
        device: Devicelike | None = None,
    ) -> None:
        """Generate contacts for the pairs collected by :meth:`split_pairs`.

        Args:
            shape_types: Geometry type per shape.
            shape_data: Per-shape scale (xyz) and margin (w).
            shape_transform: World transform per shape.
            shape_sdf_index: Per-shape index into ``texture_sdf_data``.
            texture_sdf_data: Compact texture SDF table.
            shape_gap: Per-shape contact gap [m].
            shape_base_gap: Authored per-shape gap for speculative contacts [m].
            shape_collision_aabb_lower: Local AABB lower bound per shape [m].
            shape_collision_aabb_upper: Local AABB upper bound per shape [m].
            max_speculative_extension: Speculative search extension [m].
            writer_data: Contact writer data.
            device: Launch device.
        """
        cfg = self.config
        k = self._kernels
        cap = self.max_pairs
        count = self.counts[0:1]
        base_gap = shape_base_gap if shape_base_gap is not None else shape_gap
        extension = float(max_speculative_extension) if self.speculative else 0.0

        wp.launch(
            k.prepare,
            dim=cap,
            inputs=[
                self.pairs,
                count,
                shape_types,
                shape_data,
                shape_transform,
                shape_gap,
                shape_collision_aabb_lower,
                shape_collision_aabb_upper,
                self.shape_body,
                extension,
                self._key_shape_bits,
                int(cfg.skip_convex_samples),
                self.prev,
                self.prev_pair_count,
            ],
            outputs=[
                self.pair_key,
                self.pair_prev_slot,
                self.pair_cached,
                self.pair_lower,
                self.pair_upper,
                self.pair_length,
                self.anchors,
                self.lane_seed,
            ],
            device=device,
            record_tape=False,
        )
        if cfg.curvature_sampling or cfg.surface_sampling:
            wp.launch(
                k.select_seeds,
                dim=cap * _SEED_THREADS,
                inputs=[
                    self.pairs,
                    count,
                    shape_data,
                    shape_transform,
                    shape_sdf_index,
                    texture_sdf_data,
                    self.seed_points,
                    self.seed_range,
                    int(cfg.curvature_sampling),
                    int(cfg.surface_sampling),
                    int(cfg.seed_selection == "depth"),
                    float(cfg.depth_selection_temperature),
                    float(cfg.exploration_radius),
                    float(cfg.softmax_smoothing),
                    self.frame,
                    self.pair_key,
                    self.pair_cached,
                    self.pair_lower,
                    self.pair_upper,
                    self.pair_length,
                    self.anchors,
                ],
                outputs=[self.lane_seed],
                device=device,
                record_tape=False,
            )
        wp.launch(
            k.optimize,
            dim=cap * self.lanes,
            inputs=[
                self.pairs,
                count,
                shape_data,
                shape_transform,
                shape_sdf_index,
                texture_sdf_data,
                shape_gap,
                self.seed_points,
                float(cfg.softmax_smoothing),
                float(cfg.repulsion_weight),
                float(cfg.tolerance),
                float(cfg.search_radius),
                float(cfg.hysteresis),
                float(cfg.reoptimize_distance),
                _NORMAL_MODES[cfg.normal_mode],
                extension,
                self.frame,
                self.pair_key,
                self.pair_prev_slot,
                self.pair_cached,
                self.pair_lower,
                self.pair_upper,
                self.pair_length,
                self.anchors,
                self.lane_seed,
                self.prev,
            ],
            outputs=[
                self.cand_x,
                self.cand_pa,
                self.cand_pb,
                self.cand_n,
                self.cand_sep,
                self.cand_soft,
                self.cand_flags,
                self.cand_id,
                self.iteration_count,
            ],
            device=device,
            record_tape=False,
        )
        wp.launch(
            k.finalize,
            dim=cap,
            inputs=[
                self.pairs,
                count,
                shape_data,
                shape_transform,
                shape_gap,
                base_gap,
                float(cfg.repulsion_weight),
                float(cfg.merge_distance),
                float(cfg.replacement_threshold),
                int(cfg.selection == "spread"),
                self.pair_key,
                self.pair_length,
                self.cand_x,
                self.cand_pa,
                self.cand_pb,
                self.cand_n,
                self.cand_sep,
                self.cand_soft,
                self.cand_flags,
                self.cand_id,
                self.cand_rank,
                self.cand_order,
                self.pair_prev_slot,
                self.prev,
                self.next,
                self.sort_keys,
                self.sort_values,
                writer_data,
            ],
            device=device,
            record_tape=False,
        )
        wp.utils.radix_sort_pairs(self.sort_keys, self.sort_values, cap, end_bit=self._key_end_bit)
        wp.launch(
            k.commit,
            dim=cap,
            inputs=[count, self.sort_values, self.next, self.prev, self.prev_pair_count, self.frame],
            device=device,
            record_tape=False,
        )


# ---------------------------------------------------------------------------
# Cache storage
# ---------------------------------------------------------------------------


@wp.struct
class _PairCache:
    """Cached points, ``cache_size`` entries per pair."""

    keys: wp.array[wp.int64]
    count: wp.array[wp.int32]
    pa: wp.array[wp.vec3]  # point in shape A's local frame
    pb: wp.array[wp.vec3]  # point in shape B's local frame
    x: wp.array[wp.vec3]  # world point when cached
    normal: wp.array[wp.vec3]  # world normal A -> B when cached
    ids: wp.array[wp.int32]
    flags: wp.array[wp.int32]


def _alloc_cache(cap: int, K: int) -> _PairCache:
    c = _PairCache()
    c.keys = wp.zeros(cap, dtype=wp.int64)
    c.count = wp.zeros(cap, dtype=wp.int32)
    c.pa = wp.zeros(cap * K, dtype=wp.vec3)
    c.pb = wp.zeros(cap * K, dtype=wp.vec3)
    c.x = wp.zeros(cap * K, dtype=wp.vec3)
    c.normal = wp.zeros(cap * K, dtype=wp.vec3)
    c.ids = wp.zeros(cap * K, dtype=wp.int32)
    c.flags = wp.zeros(cap * K, dtype=wp.int32)
    return c


# ---------------------------------------------------------------------------
# Device functions
# ---------------------------------------------------------------------------


@wp.func
def _sdf_scale(shape_data: wp.array[wp.vec4], shape: int, sdf: TextureSDFData) -> wp.vec3:
    """Scale applied at query time to an SDF's local coordinates."""
    if sdf.scale_baked:
        return wp.vec3(1.0, 1.0, 1.0)
    d = shape_data[shape]
    return wp.vec3(d[0], d[1], d[2])


@wp.func
def _min_abs(v: wp.vec3) -> float:
    return wp.min(wp.min(wp.abs(v[0]), wp.abs(v[1])), wp.abs(v[2]))


@wp.func
def _safe_inverse(v: wp.vec3) -> wp.vec3:
    eps = 1.0e-10
    return wp.vec3(
        1.0 / wp.where(wp.abs(v[0]) > eps, v[0], eps),
        1.0 / wp.where(wp.abs(v[1]) > eps, v[1], eps),
        1.0 / wp.where(wp.abs(v[2]) > eps, v[2], eps),
    )


@wp.func
def _phi(sdf: TextureSDFData, X_inv: wp.transform, inv_scale: wp.vec3, min_scale: float, x: wp.vec3) -> float:
    """World-space signed distance from one hardware-filtered texture fetch."""
    p = wp.cw_mul(wp.transform_point(X_inv, x), inv_scale)
    return texture_sample_sdf_hw(sdf, p) * min_scale


@wp.func
def _phi_grad(
    sdf: TextureSDFData, X: wp.transform, X_inv: wp.transform, inv_scale: wp.vec3, min_scale: float, x: wp.vec3
) -> tuple[float, wp.vec3]:
    """World-space signed distance and unit gradient from exact trilinear sampling."""
    p = wp.cw_mul(wp.transform_point(X_inv, x), inv_scale)
    d, g = texture_sample_sdf_grad(sdf, p)
    gw = wp.transform_vector(X, wp.cw_mul(g, inv_scale))
    n = wp.length(gw)
    if n > 1.0e-12:
        gw = gw / n
    else:
        gw = wp.vec3(0.0, 0.0, 1.0)
    return d * min_scale, gw


@wp.func
def _laplacian(
    sdf: TextureSDFData, X_inv: wp.transform, inv_scale: wp.vec3, min_scale: float, x: wp.vec3, phi: float
) -> float:
    """World-space Laplacian of an SDF (mean curvature near the surface), one-voxel stencil."""
    v = sdf.voxel_size
    h = (v[0] + v[1] + v[2]) / 3.0 * min_scale
    s = float(0.0)
    for k in range(3):
        e = wp.vec3(0.0)
        e[k] = h
        s += _phi(sdf, X_inv, inv_scale, min_scale, x + e) + _phi(sdf, X_inv, inv_scale, min_scale, x - e)
    return (s - 6.0 * phi) / (h * h)


@wp.func
def _objective(
    sdf_a: TextureSDFData,
    sdf_b: TextureSDFData,
    XA_inv: wp.transform,
    XB_inv: wp.transform,
    inv_a: wp.vec3,
    inv_b: wp.vec3,
    min_a: float,
    min_b: float,
    eps: float,
    alpha: float,
    anchors: wp.array[wp.vec3],
    anchor_start: int,
    anchor_count: int,
    x: wp.vec3,
) -> float:
    """Soft maximum of both SDFs (Eq. 2) minus the repulsion from anchors (Eq. 3)."""
    phi_a = _phi(sdf_a, XA_inv, inv_a, min_a, x)
    phi_b = _phi(sdf_b, XB_inv, inv_b, min_b, x)
    d = phi_a - phi_b
    g = 0.5 * (phi_a + phi_b + wp.sqrt(d * d + eps))
    for j in range(anchor_count):
        g -= alpha * wp.length(x - anchors[anchor_start + j])
    return g


@wp.func
def _world_aabb(X: wp.transform, lower: wp.vec3, upper: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
    c = wp.transform_point(X, 0.5 * (lower + upper))
    h = 0.5 * (upper - lower)
    R = wp.quat_to_matrix(wp.transform_get_rotation(X))
    e = wp.vec3(
        wp.abs(R[0, 0]) * h[0] + wp.abs(R[0, 1]) * h[1] + wp.abs(R[0, 2]) * h[2],
        wp.abs(R[1, 0]) * h[0] + wp.abs(R[1, 1]) * h[1] + wp.abs(R[1, 2]) * h[2],
        wp.abs(R[2, 0]) * h[0] + wp.abs(R[2, 1]) * h[1] + wp.abs(R[2, 2]) * h[2],
    )
    return c - e, c + e


@wp.func
def _max_extent(lower: wp.vec3, upper: wp.vec3) -> float:
    d = upper - lower
    return wp.max(wp.max(d[0], d[1]), d[2])


@wp.func
def _clamp3(p: wp.vec3, lo: wp.vec3, hi: wp.vec3) -> wp.vec3:
    return wp.vec3(wp.clamp(p[0], lo[0], hi[0]), wp.clamp(p[1], lo[1], hi[1]), wp.clamp(p[2], lo[2], hi[2]))


@wp.func
def _inside(p: wp.vec3, lo: wp.vec3, hi: wp.vec3) -> bool:
    return p[0] >= lo[0] and p[1] >= lo[1] and p[2] >= lo[2] and p[0] <= hi[0] and p[1] <= hi[1] and p[2] <= hi[2]


@wp.func
def _cswap(va: wp.vec3, fa: float, vb: wp.vec3, fb: float) -> tuple[wp.vec3, float, wp.vec3, float]:
    if fb < fa:
        return vb, fb, va, fa
    return va, fa, vb, fb


@wp.func
def _pair_key(a: int, b: int, bits: int) -> wp.int64:
    return (wp.int64(a) << wp.int64(bits)) | wp.int64(b)


@wp.func
def _rng_seed(key: wp.int64, frame: int, lane: int) -> int:
    return (
        wp.int32(key & wp.int64(0x7FFFFFFF))
        ^ (wp.int32(key >> wp.int64(31)) * 73856093)
        ^ (frame * 19349663)
        ^ (lane * 83492791)
    )


@wp.func
def _is_convex(shape_type: int) -> bool:
    return shape_type == GeoType.BOX or shape_type == GeoType.CONVEX_MESH


@wp.kernel(enable_backward=False)
def _split_pairs_kernel(
    shape_pairs: wp.array[wp.vec2i],
    shape_pair_count: wp.array[wp.int32],
    shape_sdf_index: wp.array[wp.int32],
    texture_sdf_data: wp.array[TextureSDFData],
    cs_pairs: wp.array[wp.vec2i],
    rest_pairs: wp.array[wp.vec2i],
    counts: wp.array[wp.int32],
):
    i = wp.tid()
    if i >= wp.min(shape_pair_count[0], shape_pairs.shape[0]):
        return
    pair = shape_pairs[i]
    eligible = (pair[0] & SHAPE_PAIR_HFIELD_BIT) == 0
    if eligible:
        sa = shape_sdf_index[pair[0]]
        sb = shape_sdf_index[pair[1]]
        n = texture_sdf_data.shape[0]
        eligible = sa >= 0 and sb >= 0 and sa < n and sb < n
        if eligible:
            eligible = texture_sdf_data[sa].coarse_texture.width > 0 and texture_sdf_data[sb].coarse_texture.width > 0
    if eligible:
        j = wp.atomic_add(counts, 0, 1)
        if j < cs_pairs.shape[0]:
            cs_pairs[j] = pair
    else:
        j = wp.atomic_add(counts, 1, 1)
        if j < rest_pairs.shape[0]:
            rest_pairs[j] = pair


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


class _Kernels:
    def __init__(self, prepare, select_seeds, optimize, finalize, commit):
        self.prepare = prepare
        self.select_seeds = select_seeds
        self.optimize = optimize
        self.finalize = finalize
        self.commit = commit


_kernel_cache: dict[tuple, _Kernels] = {}


def _create_kernels(
    *, cache_size: int, random_samples: int, max_iterations: int, writer_func: Any, speculative: bool
) -> _Kernels:
    key = (cache_size, random_samples, max_iterations, writer_func, speculative)
    cached = _kernel_cache.get(key)
    if cached is not None:
        return cached

    K = cache_size
    T = cache_size + random_samples
    MAX_IT = max_iterations
    S = _SEED_THREADS
    # A stable module name per configuration keeps identical pipelines on one
    # compiled kernel, which keeps results bit-identical across instances.
    module = f"sdf_cached_sampling_{K}_{random_samples}_{MAX_IT}_{int(speculative)}_{writer_func.__name__}"

    @wp.kernel(enable_backward=False, module=module)
    def prepare(
        pairs: wp.array[wp.vec2i],
        pair_count: wp.array[wp.int32],
        shape_types: wp.array[wp.int32],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_gap: wp.array[wp.float32],
        aabb_lower: wp.array[wp.vec3],
        aabb_upper: wp.array[wp.vec3],
        shape_body: wp.array[wp.int32],
        speculative_extension: float,
        key_bits: int,
        skip_convex_samples: int,
        prev: _PairCache,
        prev_pair_count: wp.array[wp.int32],
        pair_key: wp.array[wp.int64],
        pair_prev_slot: wp.array[wp.int32],
        pair_cached: wp.array[wp.int32],
        pair_lower: wp.array[wp.vec3],
        pair_upper: wp.array[wp.vec3],
        pair_length: wp.array[wp.float32],
        anchors: wp.array[wp.vec3],
        lane_seed: wp.array[wp.int64],
    ):
        """Search domain, characteristic size, cache lookup and cache advection."""
        p = wp.tid()
        if p >= wp.min(pair_count[0], pairs.shape[0]):
            return
        for lane in range(T):
            lane_seed[p * T + lane] = _NO_SEED
        a = pairs[p][0]
        b = pairs[p][1]
        XA = shape_transform[a]
        XB = shape_transform[b]
        la, ua = _world_aabb(XA, aabb_lower[a], aabb_upper[a])
        lb, ub = _world_aabb(XB, aabb_lower[b], aabb_upper[b])
        # Grow by half the detection threshold each, so V contains every point
        # within the contact skin of both shapes (Sec. 3.1).
        r = 0.5 * (shape_gap[a] + shape_gap[b] + shape_data[a][3] + shape_data[b][3] + speculative_extension)
        grow = wp.vec3(r, r, r)
        pair_lower[p] = wp.max(la, lb) - grow
        pair_upper[p] = wp.min(ua, ub) + grow
        pair_length[p] = wp.max(
            wp.min(_max_extent(aabb_lower[a], aabb_upper[a]), _max_extent(aabb_lower[b], aabb_upper[b])), 1.0e-6
        )

        key = _pair_key(a, b, key_bits)
        pair_key[p] = key

        # Binary search the previous call's sorted pair keys.
        lo = int(0)
        hi = prev_pair_count[0]
        while lo < hi:
            mid = (lo + hi) // 2
            if prev.keys[mid] < key:
                lo = mid + 1
            else:
                hi = mid
        slot = int(-1)
        if lo < prev_pair_count[0]:
            if prev.keys[lo] == key:
                slot = lo
        pair_prev_slot[p] = slot
        n_cached = int(0)
        if slot >= 0:
            n_cached = prev.count[slot]

        # Cached points follow the shapes' rigid motion; a static shape does not
        # move them, only the other shape does (Eq. 6).
        static_a = bool(False)
        static_b = bool(False)
        if a < shape_body.shape[0]:
            static_a = shape_body[a] < 0
        if b < shape_body.shape[0]:
            static_b = shape_body[b] < 0
        all_active = bool(True)
        for k in range(n_cached):
            e = slot * K + k
            xa = wp.transform_point(XA, prev.pa[e])
            xb = wp.transform_point(XB, prev.pb[e])
            x = 0.5 * (xa + xb)
            if static_a and not static_b:
                x = xb
            elif static_b and not static_a:
                x = xa
            anchors[p * K + k] = x
            if (prev.flags[e] & _FLAG_ACTIVE) == 0:
                all_active = False

        # Lanes [n_cached, T) take stochastic samples, except for a convex pair
        # whose full cache is in contact (one convex overlap region).
        if (
            skip_convex_samples != 0
            and n_cached == K
            and all_active
            and _is_convex(shape_types[a])
            and _is_convex(shape_types[b])
        ):
            n_cached = -K
        pair_cached[p] = n_cached

    @wp.kernel(enable_backward=False, module=module)
    def select_seeds(
        pairs: wp.array[wp.vec2i],
        pair_count: wp.array[wp.int32],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_sdf_index: wp.array[wp.int32],
        texture_sdf_data: wp.array[TextureSDFData],
        seed_points: wp.array[wp.vec3],
        seed_range: wp.array[wp.vec4i],
        curvature_sampling: int,
        surface_sampling: int,
        depth_selection: int,
        depth_temperature_rel: float,
        exploration_radius_rel: float,
        softmax_smoothing: float,
        frame: wp.array[wp.int32],
        pair_key: wp.array[wp.int64],
        pair_cached: wp.array[wp.int32],
        pair_lower: wp.array[wp.vec3],
        pair_upper: wp.array[wp.vec3],
        pair_length: wp.array[wp.float32],
        anchors: wp.array[wp.vec3],
        lane_seed: wp.array[wp.int64],
    ):
        """Pick one seed point inside V per stochastic lane (Sec. 3.5.2).

        Every seed gets a score per lane, the lowest wins. ``"uniform"`` scores are
        random, so the winner is uniform among the seeds inside V. ``"depth"`` scores
        are the objective plus Gumbel noise (a randomized argmin, so lanes differ),
        skipping seeds near cached points. The packed (score, seed) atomic minimum
        makes the result independent of thread order.
        """
        tid = wp.tid()
        p = tid // S
        t = tid - p * S
        if p >= wp.min(pair_count[0], pairs.shape[0]):
            return
        n_cached = pair_cached[p]
        if n_cached < 0 or n_cached >= T:
            return
        a = pairs[p][0]
        b = pairs[p][1]
        lower = pair_lower[p]
        upper = pair_upper[p]
        if lower[0] > upper[0] or lower[1] > upper[1] or lower[2] > upper[2]:
            return
        sdf_a = texture_sdf_data[shape_sdf_index[a]]
        sdf_b = texture_sdf_data[shape_sdf_index[b]]
        XA = shape_transform[a]
        XB = shape_transform[b]
        XA_inv = wp.transform_inverse(XA)
        XB_inv = wp.transform_inverse(XB)
        scale_a = _sdf_scale(shape_data, a, sdf_a)
        scale_b = _sdf_scale(shape_data, b, sdf_b)
        inv_a = _safe_inverse(scale_a)
        inv_b = _safe_inverse(scale_b)
        min_a = _min_abs(scale_a)
        min_b = _min_abs(scale_b)
        L = pair_length[p]
        eps = (softmax_smoothing * L) * (softmax_smoothing * L)
        temperature = depth_temperature_rel * L
        explore = exploration_radius_rel * L
        resolution = _SCORE_RESOLUTION * L
        key = pair_key[p]
        rng_base = _rng_seed(key, frame[0], 0)

        ra = seed_range[shape_sdf_index[a]]
        rb = seed_range[shape_sdf_index[b]]
        # Enumerate [A curvature, A surface, B curvature, B surface] as one range.
        na0 = wp.where(curvature_sampling != 0, ra[1], 0)
        na1 = wp.where(surface_sampling != 0, ra[3], 0)
        nb0 = wp.where(curvature_sampling != 0, rb[1], 0)
        nb1 = wp.where(surface_sampling != 0, rb[3], 0)
        total = na0 + na1 + nb0 + nb1
        for j in range(t, total, S):
            side = int(0)
            idx = int(0)
            if j < na0:
                idx = ra[0] + j
            elif j < na0 + na1:
                idx = ra[2] + j - na0
            elif j < na0 + na1 + nb0:
                side = 1
                idx = rb[0] + j - na0 - na1
            else:
                side = 1
                idx = rb[2] + j - na0 - na1 - nb0
            if side == 0:
                q = wp.transform_point(XA, wp.cw_mul(seed_points[idx], scale_a))
            else:
                q = wp.transform_point(XB, wp.cw_mul(seed_points[idx], scale_b))
            if not _inside(q, lower, upper):
                continue
            g = float(0.0)
            if depth_selection != 0:
                covered = bool(False)
                for k in range(n_cached):
                    if wp.length_sq(q - anchors[p * K + k]) < explore * explore:
                        covered = True
                if covered:
                    continue
                g = _objective(sdf_a, sdf_b, XA_inv, XB_inv, inv_a, inv_b, min_a, min_b, eps, 0.0, anchors, 0, 0, q)
            rng = wp.rand_init(rng_base, j)
            for lane in range(n_cached, T):
                u = wp.clamp(wp.randf(rng), 1.0e-7, 1.0 - 1.0e-7)
                score = float(0.0)
                if depth_selection != 0:
                    score = g - temperature * wp.log(-wp.log(u))
                else:
                    score = u * L
                # 31-bit quantized score above (side, seed index).
                quantized = wp.int64(wp.clamp(score / resolution, -1.0e9, 1.0e9)) + wp.int64(1 << 30)
                packed = (quantized << wp.int64(32)) | (wp.int64(side) << wp.int64(31)) | wp.int64(idx)
                wp.atomic_min(lane_seed, p * T + lane, packed)

    @wp.kernel(enable_backward=False, module=module)
    def optimize(
        pairs: wp.array[wp.vec2i],
        pair_count: wp.array[wp.int32],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_sdf_index: wp.array[wp.int32],
        texture_sdf_data: wp.array[TextureSDFData],
        shape_gap: wp.array[wp.float32],
        seed_points: wp.array[wp.vec3],
        softmax_smoothing: float,
        repulsion_weight: float,
        tolerance_rel: float,
        search_radius_rel: float,
        hysteresis_rel: float,
        reoptimize_rel: float,
        normal_mode: int,
        speculative_extension: float,
        frame: wp.array[wp.int32],
        pair_key: wp.array[wp.int64],
        pair_prev_slot: wp.array[wp.int32],
        pair_cached: wp.array[wp.int32],
        pair_lower: wp.array[wp.vec3],
        pair_upper: wp.array[wp.vec3],
        pair_length: wp.array[wp.float32],
        anchors: wp.array[wp.vec3],
        lane_seed: wp.array[wp.int64],
        prev: _PairCache,
        cand_x: wp.array[wp.vec3],
        cand_pa: wp.array[wp.vec3],
        cand_pb: wp.array[wp.vec3],
        cand_n: wp.array[wp.vec3],
        cand_sep: wp.array[wp.float32],
        cand_soft: wp.array[wp.float32],
        cand_flags: wp.array[wp.int32],
        cand_id: wp.array[wp.int32],
        iteration_count: wp.array[wp.int32],
    ):
        """One Nelder-Mead minimization of Eq. 3 per candidate lane."""
        tid = wp.tid()
        p = tid // T
        lane = tid - p * T
        if p >= wp.min(pair_count[0], pairs.shape[0]):
            return
        c = p * T + lane
        cand_flags[c] = 0

        lower = pair_lower[p]
        upper = pair_upper[p]
        if lower[0] > upper[0] or lower[1] > upper[1] or lower[2] > upper[2]:
            return
        n_cached = pair_cached[p]
        skip_samples = n_cached < 0
        if skip_samples:
            n_cached = -n_cached
        cached = lane < n_cached
        if not cached and skip_samples:
            return

        a = pairs[p][0]
        b = pairs[p][1]
        sdf_a = texture_sdf_data[shape_sdf_index[a]]
        sdf_b = texture_sdf_data[shape_sdf_index[b]]
        XA = shape_transform[a]
        XB = shape_transform[b]
        XA_inv = wp.transform_inverse(XA)
        XB_inv = wp.transform_inverse(XB)
        scale_a = _sdf_scale(shape_data, a, sdf_a)
        scale_b = _sdf_scale(shape_data, b, sdf_b)
        inv_a = _safe_inverse(scale_a)
        inv_b = _safe_inverse(scale_b)
        min_a = _min_abs(scale_a)
        min_b = _min_abs(scale_b)

        L = pair_length[p]
        tol = tolerance_rel * L
        eps = (softmax_smoothing * L) * (softmax_smoothing * L)
        alpha = repulsion_weight

        # Seed: a carried cached point, a selected seed point, or uniform in V.
        prev_x = wp.vec3(0.0)
        prev_n = wp.vec3(0.0)
        prev_id = int(-1)
        n_anchor = n_cached
        frozen = bool(False)
        if cached:
            e = pair_prev_slot[p] * K + lane
            seed = anchors[p * K + lane]
            prev_x = prev.x[e]
            prev_n = prev.normal[e]
            prev_id = prev.ids[e]
            # A cached point is repelled by the cached points before it.
            n_anchor = lane
            # Without relative motion of the shapes at the point since it was last
            # optimized, its optimum is unchanged: skip the optimizer.
            drift = wp.length(wp.transform_point(XA, prev.pa[e]) - wp.transform_point(XB, prev.pb[e]))
            frozen = drift < reoptimize_rel * L
        else:
            packed = lane_seed[c]
            if packed != _NO_SEED:
                idx = int(packed & wp.int64(0x7FFFFFFF))
                if ((packed >> wp.int64(31)) & wp.int64(1)) == wp.int64(0):
                    seed = wp.transform_point(XA, wp.cw_mul(seed_points[idx], scale_a))
                else:
                    seed = wp.transform_point(XB, wp.cw_mul(seed_points[idx], scale_b))
            else:
                rng = wp.rand_init(_rng_seed(pair_key[p], frame[0], 1), lane)
                seed = lower + wp.cw_mul(wp.vec3(wp.randf(rng), wp.randf(rng), wp.randf(rng)), upper - lower)

        # Search a box around the seed (inside V) so each lane resolves the deepest
        # point of its own contact region instead of all lanes falling into the
        # globally deepest basin.
        x0 = _clamp3(seed, lower, upper)
        if search_radius_rel > 0.0:
            r = search_radius_rel * L
            if cached:
                r += wp.length(seed - prev_x)
            lower = wp.max(lower, x0 - wp.vec3(r))
            upper = wp.min(upper, x0 + wp.vec3(r))

        # Initial simplex along the axes into the box, 10% of its size (Sec. 3.7).
        # A cached point only follows the shapes, so its simplex is limited to the
        # advection distance to keep it from wandering on flat contacts.
        h = wp.max(_SIMPLEX_FRACTION * (upper - lower), wp.vec3(0.5 * tol))
        if cached:
            h = wp.min(h, wp.vec3(wp.max(wp.length(seed - prev_x), 5.0 * tol)))
        v0 = x0
        v1 = _clamp3(x0 + wp.vec3(wp.where(x0[0] + h[0] <= upper[0], h[0], -h[0]), 0.0, 0.0), lower, upper)
        v2 = _clamp3(x0 + wp.vec3(0.0, wp.where(x0[1] + h[1] <= upper[1], h[1], -h[1]), 0.0), lower, upper)
        v3 = _clamp3(x0 + wp.vec3(0.0, 0.0, wp.where(x0[2] + h[2] <= upper[2], h[2], -h[2])), lower, upper)
        anchor_start = p * K
        f0 = _objective(
            sdf_a, sdf_b, XA_inv, XB_inv, inv_a, inv_b, min_a, min_b, eps, alpha, anchors, anchor_start, n_anchor, v0
        )
        f1 = _objective(
            sdf_a, sdf_b, XA_inv, XB_inv, inv_a, inv_b, min_a, min_b, eps, alpha, anchors, anchor_start, n_anchor, v1
        )
        f2 = _objective(
            sdf_a, sdf_b, XA_inv, XB_inv, inv_a, inv_b, min_a, min_b, eps, alpha, anchors, anchor_start, n_anchor, v2
        )
        f3 = _objective(
            sdf_a, sdf_b, XA_inv, XB_inv, inv_a, inv_b, min_a, min_b, eps, alpha, anchors, anchor_start, n_anchor, v3
        )
        f_seed = f0

        it = int(0)
        if frozen:
            it = MAX_IT
        while it < MAX_IT:
            # Order the vertices best to worst (5-comparator network).
            v0, f0, v1, f1 = _cswap(v0, f0, v1, f1)
            v2, f2, v3, f3 = _cswap(v2, f2, v3, f3)
            v0, f0, v2, f2 = _cswap(v0, f0, v2, f2)
            v1, f1, v3, f3 = _cswap(v1, f1, v3, f3)
            v1, f1, v2, f2 = _cswap(v1, f1, v2, f2)
            extent = wp.max(wp.max(wp.length(v1 - v0), wp.length(v2 - v0)), wp.length(v3 - v0))
            if extent < tol or f3 - f0 < 0.05 * tol:
                break
            centroid = (v0 + v1 + v2) / 3.0
            reflected = _clamp3(centroid + (centroid - v3), lower, upper)
            fr = _objective(
                sdf_a,
                sdf_b,
                XA_inv,
                XB_inv,
                inv_a,
                inv_b,
                min_a,
                min_b,
                eps,
                alpha,
                anchors,
                anchor_start,
                n_anchor,
                reflected,
            )
            if fr < f0:
                expanded = _clamp3(centroid + 2.0 * (centroid - v3), lower, upper)
                fe = _objective(
                    sdf_a,
                    sdf_b,
                    XA_inv,
                    XB_inv,
                    inv_a,
                    inv_b,
                    min_a,
                    min_b,
                    eps,
                    alpha,
                    anchors,
                    anchor_start,
                    n_anchor,
                    expanded,
                )
                if fe < fr:
                    v3 = expanded
                    f3 = fe
                else:
                    v3 = reflected
                    f3 = fr
            elif fr < f2:
                v3 = reflected
                f3 = fr
            else:
                # Contract towards the better of the reflected and worst points,
                # shrink towards the best vertex if that fails.
                target = wp.where(fr < f3, reflected, v3)
                contracted = centroid + 0.5 * (target - centroid)
                fc = _objective(
                    sdf_a,
                    sdf_b,
                    XA_inv,
                    XB_inv,
                    inv_a,
                    inv_b,
                    min_a,
                    min_b,
                    eps,
                    alpha,
                    anchors,
                    anchor_start,
                    n_anchor,
                    contracted,
                )
                if fc < wp.min(fr, f3):
                    v3 = contracted
                    f3 = fc
                else:
                    v1 = v0 + 0.5 * (v1 - v0)
                    v2 = v0 + 0.5 * (v2 - v0)
                    v3 = v0 + 0.5 * (v3 - v0)
                    f1 = _objective(
                        sdf_a,
                        sdf_b,
                        XA_inv,
                        XB_inv,
                        inv_a,
                        inv_b,
                        min_a,
                        min_b,
                        eps,
                        alpha,
                        anchors,
                        anchor_start,
                        n_anchor,
                        v1,
                    )
                    f2 = _objective(
                        sdf_a,
                        sdf_b,
                        XA_inv,
                        XB_inv,
                        inv_a,
                        inv_b,
                        min_a,
                        min_b,
                        eps,
                        alpha,
                        anchors,
                        anchor_start,
                        n_anchor,
                        v2,
                    )
                    f3 = _objective(
                        sdf_a,
                        sdf_b,
                        XA_inv,
                        XB_inv,
                        inv_a,
                        inv_b,
                        min_a,
                        min_b,
                        eps,
                        alpha,
                        anchors,
                        anchor_start,
                        n_anchor,
                        v3,
                    )
            it += 1

        x = v0
        fx = f0
        if f1 < fx:
            x = v1
            fx = f1
        if f2 < fx:
            x = v2
            fx = f2
        if f3 < fx:
            x = v3
            fx = f3
        # Hysteresis: a cached point stays fixed to the shapes unless moving it
        # improves the objective by more than the tolerance. On flat or resting
        # contacts the optimizer would otherwise shift it every call, which moves
        # the contact and disturbs the solver's friction anchors and warm start.
        if cached and f_seed - fx < hysteresis_rel * L:
            x = x0
        if frozen:
            x = x0
        else:
            wp.atomic_add(iteration_count, 0, it)

        # Contact data at x* (Sec. 3.4, Eq. 5).
        phi_a, g_a = _phi_grad(sdf_a, XA, XA_inv, inv_a, min_a, x)
        phi_b, g_b = _phi_grad(sdf_b, XB, XB_inv, inv_b, min_b, x)
        n_a = g_a
        n_b = -g_b
        normal = wp.where(phi_a < phi_b, n_a, n_b)
        if cached:
            # Keep the normal of a cached point coherent: x* balances phi_a and
            # phi_b, so either field can be the deeper one from step to step.
            normal = wp.where(wp.dot(n_a, prev_n) > wp.dot(n_b, prev_n), n_a, n_b)
        if normal_mode != 0 and wp.dot(n_a, n_b) > 0.0:
            # At an exact minimizer the gradients are antiparallel (n_a == n_b).
            # Away from it, e.g. at the rim of a shallow overlap where the surfaces
            # cross at an angle, the curved surface's normal tilts. Weighting each
            # normal by inverse squared curvature picks the flatter surface's normal
            # (the face of a feature-on-face contact) and gives the bisector for
            # equal curvature. Opposing normals (a thin feature inside the other
            # shape) keep the coherent choice.
            w_a = float(1.0)
            w_b = float(1.0)
            if normal_mode == 2:
                k0 = 0.01 / L
                k_a = k0 + wp.abs(_laplacian(sdf_a, XA_inv, inv_a, min_a, x, phi_a))
                k_b = k0 + wp.abs(_laplacian(sdf_b, XB_inv, inv_b, min_b, x, phi_b))
                w_a = 1.0 / (k_a * k_a)
                w_b = 1.0 / (k_b * k_b)
            blend = w_a * n_a + w_b * n_b
            blend_len = wp.length(blend)
            if blend_len > 1.0e-6:
                normal = blend / blend_len
        # Witness points: one Newton step onto each surface.
        pa = x - g_a * phi_a
        pb = x - g_b * phi_b
        if phi_a > 0.0 and phi_b > 0.0:
            # Outside both shapes the gap between the witness points gives the normal,
            # unless that reverses a cached point's normal.
            gap = pb - pa
            gap_len = wp.length(gap)
            if gap_len > 1.0e-9:
                if not (cached and wp.dot(gap, prev_n) < 0.0):
                    normal = gap / gap_len
        sep = wp.dot(pb - pa, normal)

        threshold = shape_gap[a] + shape_gap[b] + shape_data[a][3] + shape_data[b][3] + speculative_extension
        flags = _FLAG_VALID
        if sep < threshold:
            flags = flags | _FLAG_ACTIVE
        if cached:
            flags = flags | _FLAG_CACHED
        if frozen:
            flags = flags | _FLAG_FROZEN
        cand_x[c] = x
        cand_pa[c] = pa
        cand_pb[c] = pb
        cand_n[c] = normal
        cand_sep[c] = sep
        cand_soft[c] = 0.5 * (phi_a + phi_b + wp.sqrt((phi_a - phi_b) * (phi_a - phi_b) + eps))
        cand_flags[c] = flags
        cand_id[c] = prev_id

    @wp.kernel(enable_backward=False, module=module)
    def finalize(
        pairs: wp.array[wp.vec2i],
        pair_count: wp.array[wp.int32],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_gap: wp.array[wp.float32],
        shape_base_gap: wp.array[wp.float32],
        repulsion_weight: float,
        merge_rel: float,
        replacement_rel: float,
        spread_selection: int,
        pair_key: wp.array[wp.int64],
        pair_length: wp.array[wp.float32],
        cand_x: wp.array[wp.vec3],
        cand_pa: wp.array[wp.vec3],
        cand_pb: wp.array[wp.vec3],
        cand_n: wp.array[wp.vec3],
        cand_sep: wp.array[wp.float32],
        cand_soft: wp.array[wp.float32],
        cand_flags: wp.array[wp.int32],
        cand_id: wp.array[wp.int32],
        cand_rank: wp.array[wp.float32],
        cand_order: wp.array[wp.int32],
        pair_prev_slot: wp.array[wp.int32],
        prev: _PairCache,
        next: _PairCache,
        sort_keys: wp.array[wp.int64],
        sort_values: wp.array[wp.int32],
        writer_data: Any,
    ):
        """De-duplicate, select, update the cache and emit contacts per pair."""
        p = wp.tid()
        n_pairs = wp.min(pair_count[0], pairs.shape[0])
        sort_values[p] = p
        if p >= n_pairs:
            sort_keys[p] = wp.int64(0x7FFFFFFFFFFFFFFF)
            return
        sort_keys[p] = pair_key[p]
        next.keys[p] = pair_key[p]
        base = p * T
        L = pair_length[p]
        merge = merge_rel * L
        bonus = replacement_rel * L

        # Remove points near an earlier point; cached lanes come first, so they
        # survive over new samples (Alg. 1, lines 20-27).
        n = int(0)
        for i in range(T):
            ci = base + i
            if (cand_flags[ci] & _FLAG_VALID) == 0:
                continue
            duplicate = bool(False)
            for j in range(n):
                cj = base + cand_order[base + j]
                if wp.length_sq(cand_x[ci] - cand_x[cj]) < merge * merge:
                    duplicate = True
            if not duplicate:
                cand_order[base + n] = i
                n += 1

        if spread_selection != 0:
            # Greedy farthest-point selection in lexicographic (tier, value) order:
            # tier 2 is the deepest point in contact (first slot only), tier 1 the
            # points in contact by distance to those already selected, tier 0 the
            # rest by depth. Cached points get the replacement threshold as a bonus
            # so the selection persists.
            for slot in range(n):
                best_i = int(slot)
                best_t = int(-1)
                best_v = float(0.0)
                for i in range(slot, n):
                    ci = base + cand_order[base + i]
                    tier = int(0)
                    v = -cand_soft[ci]
                    if (cand_flags[ci] & _FLAG_ACTIVE) != 0:
                        if slot == 0:
                            tier = 2
                        else:
                            tier = 1
                            dmin = float(1.0e30)
                            for j in range(slot):
                                cj = base + cand_order[base + j]
                                dmin = wp.min(dmin, wp.length(cand_x[ci] - cand_x[cj]))
                            v = dmin
                    if (cand_flags[ci] & _FLAG_CACHED) != 0:
                        v += bonus
                    if tier > best_t or (tier == best_t and v > best_v):
                        best_t = tier
                        best_v = v
                        best_i = i
                tmp = cand_order[base + slot]
                cand_order[base + slot] = cand_order[base + best_i]
                cand_order[base + best_i] = tmp
        else:
            # Rank by g against the final manifold (Alg. 1, line 28); new samples
            # pay the replacement threshold. Stable insertion sort by rank.
            for i in range(n):
                ci = base + cand_order[base + i]
                r = cand_soft[ci]
                if (cand_flags[ci] & _FLAG_CACHED) == 0:
                    r += bonus
                for j in range(n):
                    r -= repulsion_weight * wp.length(cand_x[ci] - cand_x[base + cand_order[base + j]])
                cand_rank[ci] = r
            for i in range(1, n):
                li = cand_order[base + i]
                ri = cand_rank[base + li]
                j = i - 1
                while j >= 0 and cand_rank[base + cand_order[base + j]] > ri:
                    cand_order[base + j + 1] = cand_order[base + j]
                    j -= 1
                cand_order[base + j + 1] = li

        # Stable contact ids: cached points keep theirs, new points take the
        # smallest free one. Ids are the contacts' sort sub-keys.
        kept = wp.min(n, K)
        used = int(0)
        for i in range(kept):
            ci = base + cand_order[base + i]
            if cand_id[ci] >= 0:
                used = used | (1 << cand_id[ci])
        for i in range(kept):
            ci = base + cand_order[base + i]
            if cand_id[ci] < 0:
                fid = int(0)
                while (used & (1 << fid)) != 0:
                    fid += 1
                used = used | (1 << fid)
                cand_id[ci] = fid

        a = pairs[p][0]
        b = pairs[p][1]
        XA_inv = wp.transform_inverse(shape_transform[a])
        XB_inv = wp.transform_inverse(shape_transform[b])
        next.count[p] = kept
        if wp.static(speculative):
            gap_sum = shape_base_gap[a] + shape_base_gap[b]
        else:
            gap_sum = shape_gap[a] + shape_gap[b]
        for i in range(kept):
            ci = base + cand_order[base + i]
            e = p * K + i
            x = cand_x[ci]
            next.x[e] = x
            if (cand_flags[ci] & _FLAG_FROZEN) != 0:
                # Keep the frames of the last optimization, so the relative drift
                # that triggers re-optimization accumulates.
                src = pair_prev_slot[p] * K + (ci - base)
                next.pa[e] = prev.pa[src]
                next.pb[e] = prev.pb[src]
            else:
                next.pa[e] = wp.transform_point(XA_inv, x)
                next.pb[e] = wp.transform_point(XB_inv, x)
            next.normal[e] = cand_n[ci]
            next.ids[e] = cand_id[ci]
            next.flags[e] = cand_flags[ci]
            # Only kept points become contacts, so every contact persists into the
            # next call; a transient sample at a new spot each call would apply
            # impulses at random places.
            if (cand_flags[ci] & _FLAG_ACTIVE) == 0:
                continue
            cd = ContactData()
            cd.contact_point_center = 0.5 * (cand_pa[ci] + cand_pb[ci])
            cd.contact_normal_a_to_b = cand_n[ci]
            cd.contact_distance = cand_sep[ci]
            cd.radius_eff_a = 0.0
            cd.radius_eff_b = 0.0
            cd.margin_a = shape_data[a][3]
            cd.margin_b = shape_data[b][3]
            cd.shape_a = a
            cd.shape_b = b
            cd.gap_sum = gap_sum
            cd.sort_sub_key = cand_id[ci] << 2
            writer_func(cd, writer_data, -1)

    @wp.kernel(enable_backward=False, module=module)
    def commit(
        pair_count: wp.array[wp.int32],
        order: wp.array[wp.int32],
        next: _PairCache,
        prev: _PairCache,
        prev_pair_count: wp.array[wp.int32],
        frame: wp.array[wp.int32],
    ):
        """Gather the cache into key order for the next call's binary search."""
        i = wp.tid()
        n = wp.min(pair_count[0], next.keys.shape[0])
        if i == 0:
            prev_pair_count[0] = n
            frame[0] = frame[0] + 1
        if i >= n:
            return
        src = order[i]
        prev.keys[i] = next.keys[src]
        cnt = next.count[src]
        prev.count[i] = cnt
        for k in range(cnt):
            s = src * K + k
            d = i * K + k
            prev.pa[d] = next.pa[s]
            prev.pb[d] = next.pb[s]
            prev.x[d] = next.x[s]
            prev.normal[d] = next.normal[s]
            prev.ids[d] = next.ids[s]
            prev.flags[d] = next.flags[s]

    kernels = _Kernels(prepare, select_seeds, optimize, finalize, commit)
    _kernel_cache[key] = kernels
    return kernels


# ---------------------------------------------------------------------------
# Seed points (Sec. 3.5.2, Eq. 7)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _seed_candidates_kernel(
    texture_sdf_data: wp.array[TextureSDFData],
    candidates: int,
    points: wp.array[wp.vec3],
    laplacian: wp.array[wp.float32],
    valid: wp.array[wp.int32],
):
    """Project random points of each SDF onto its surface and evaluate the Laplacian."""
    s, i = wp.tid()
    out = s * candidates + i
    valid[out] = 0
    sdf = texture_sdf_data[s]
    if sdf.coarse_texture.width == 0:
        return
    rng = wp.rand_init(1234567 + s * 7919, i)
    lo = sdf.sdf_box_lower
    hi = sdf.sdf_box_upper
    p = lo + wp.cw_mul(wp.vec3(wp.randf(rng), wp.randf(rng), wp.randf(rng)), hi - lo)
    voxel = sdf.voxel_size
    h = (voxel[0] + voxel[1] + voxel[2]) / 3.0
    tol = 0.05 * h
    for _it in range(32):
        d, g = texture_sample_sdf_grad(sdf, p)
        gg = wp.length_sq(g)
        if wp.abs(d) < tol or gg < 1.0e-12:
            break
        p = _clamp3(p - g * (d / gg), lo, hi)
    d, g = texture_sample_sdf_grad(sdf, p)
    if wp.abs(d) >= tol:
        return
    # Eq. 7: central differences of the gradient, one voxel apart (the trilinear
    # gradient is piecewise constant below that).
    lap = float(0.0)
    for k in range(3):
        e = wp.vec3(0.0)
        e[k] = h
        _d1, g1 = texture_sample_sdf_grad(sdf, p + e)
        _d0, g0 = texture_sample_sdf_grad(sdf, p - e)
        lap += (g1[k] - g0[k]) / (2.0 * h)
    points[out] = p
    laplacian[out] = lap
    valid[out] = 1


def _poisson_thin(points: np.ndarray, spacing: float, limit: int) -> list[np.ndarray]:
    """Greedy Poisson-disk thinning in the given order, after a grid pre-pass."""
    if len(points) == 0:
        return []
    cell = np.floor(points / (spacing / np.sqrt(3.0))).astype(np.int64)
    _, first = np.unique(cell, axis=0, return_index=True)
    chosen: list[np.ndarray] = []
    for x in points[np.sort(first)]:
        if chosen and np.min(np.sum((np.asarray(chosen) - x) ** 2, axis=1)) < spacing * spacing:
            continue
        chosen.append(x)
        if len(chosen) >= limit:
            break
    return chosen


def compute_sdf_seeds(
    texture_sdf_data: wp.array[TextureSDFData],
    config: SDFCachedSampling.Config,
    device: Devicelike | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.vec4i]]:
    """Compute curvature and surface seed points of each texture SDF.

    Random points are projected onto each SDF's zero level set. Curvature seeds
    (Sec. 3.5.2) are the samples with ``|laplacian(phi)| > seed_curvature_threshold / L``
    in order of decreasing curvature, thinned to a Poisson-disk spacing of
    ``seed_spacing * L``, where ``L`` is the largest extent of the samples. Surface
    seeds are all samples thinned to a spacing of ``surface_seed_spacing * L``.

    Args:
        texture_sdf_data: Compact texture SDF table.
        config: Seed parameters.
        device: Device for the outputs.

    Returns:
        Seed points in each SDF's unscaled local frame, and per SDF
        ``(curvature_start, curvature_count, surface_start, surface_count)``.
    """
    n_sdf = texture_sdf_data.shape[0]
    N = int(config.seed_candidates)
    points = wp.zeros(n_sdf * N, dtype=wp.vec3, device=device)
    lap = wp.zeros(n_sdf * N, dtype=wp.float32, device=device)
    valid = wp.zeros(n_sdf * N, dtype=wp.int32, device=device)
    wp.launch(
        _seed_candidates_kernel,
        dim=(n_sdf, N),
        inputs=[texture_sdf_data, N],
        outputs=[points, lap, valid],
        device=device,
    )
    pts = points.numpy().reshape(n_sdf, N, 3)
    lp = lap.numpy().reshape(n_sdf, N)
    ok = valid.numpy().reshape(n_sdf, N).astype(bool)

    seeds: list[np.ndarray] = []
    ranges = np.zeros((n_sdf, 4), dtype=np.int32)
    for s in range(n_sdf):
        ranges[s, 0] = len(seeds)
        ranges[s, 2] = len(seeds)
        if not ok[s].any():
            continue
        p = pts[s][ok[s]]
        q = np.abs(lp[s][ok[s]])
        L = float(np.max(p.max(axis=0) - p.min(axis=0)))
        if L <= 0.0:
            continue
        if config.curvature_sampling:
            keep = q > config.seed_curvature_threshold / L
            order = np.argsort(-q[keep], kind="stable")
            chosen = _poisson_thin(p[keep][order], config.seed_spacing * L, config.max_seeds_per_sdf)
            seeds.extend(chosen)
            ranges[s, 1] = len(chosen)
        ranges[s, 2] = len(seeds)
        if config.surface_sampling:
            chosen = _poisson_thin(p, config.surface_seed_spacing * L, config.max_surface_seeds_per_sdf)
            seeds.extend(chosen)
            ranges[s, 3] = len(chosen)

    seed_arr = np.asarray(seeds, dtype=np.float32).reshape(-1, 3) if seeds else np.zeros((1, 3), dtype=np.float32)
    return wp.array(seed_arr, dtype=wp.vec3, device=device), wp.array(ranges, dtype=wp.vec4i, device=device)
