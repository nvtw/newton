# Cached importance-sampled SDF contacts vs edge sampling (PhoenX)

Branch `phoenx/sdf-cached-sampling`, 2026-10-15. Method: Giles & Andrews, PG 2026
(see `PLAN.md`), implemented in `newton/_src/geometry/sdf_cached_sampling.py`,
selected with `CollisionPipeline(sdf_contact_method="cached_sampling")`.

Harness: `compare.py` (CUDA-graph replay, solo GPU, RTX PRO 6000 Blackwell).
Penetration is measured by an independent edge-sampling pipeline on the simulated
state (max over samples every 30 frames), so the method under test never grades
its own contacts. 600 frames at 60 Hz. Raw data: `final_v2.json` / `final_v2.log`.

| Scene | Method | FPS | collide ms/frame | contacts | penetration mm | stability |
|---|---|---:|---:|---:|---:|---|
| box pyramid (21 mesh cubes) | edge | 442 | 0.135 | 377 | 0.52 | drift 18.9 mm |
| | cached | 460 | 0.199 | 204 | 0.45 | **drift 0.22 mm** |
| box tower (10) | edge | 791 | 0.092 | 80 | 0.45 | drift 14.1 mm |
| | cached | 817 | 0.157 | 40 | 0.85 | drift 10.9 mm |
| gear stack (6 gears) | edge | 268 | 0.105 | 345 | 0.45 | drift 18.0 mm |
| | cached | **827** | 0.148 | 24 | 1.47 | **drift 7.1 mm** |
| cube drops (16, tilted) | edge | 971 | 0.119 | 72 | 0.45 | KE 1.2e-4 J |
| | cached | 943 | 0.190 | 64 | 2.50 | **KE 5e-11 J** |
| bunny drops (16) | edge | 706 | 0.143 | 247 | 0.45 | KE 2.2e-3 J |
| | cached | 931 | 0.209 | 64 | 3.86 | KE 2.6e-2 J |
| bunny pile (27) | edge | 315 | 0.193 | 768 | 5.17 | KE 4.1e-3 J |
| | cached | **661** | 0.261 | 171 | 14.1 | KE 5.4e-3 J |
| 100 gear stacks (600 gears) | edge | 87 | 0.243 | 33509 | 7.57 | drift 473 mm |
| | cached | **323** | 0.402 | 2400 | 3.93 | **drift 6.3 mm** |
| bunny pile (343, still falling) | edge | 78 | 1.105 | 15779 | 44.5 | — |
| | cached | **271** | **0.374** | 3061 | 50.8 | — |
| analog clock | edge | 99 | 0.413 | 315 | **0.05** | **joint 0.64 mm** |
| | cached | 105 | 0.591 | 173 | 2.30 | joint 7.42 mm |
| tourbillon clock | edge | 96 | 1.123 | 656 | 1.36 | **drive 0.109 rad/s** |
| | cached | 106 | 1.257 | 94 | 1.34 | drive 0.018 rad/s (train binds) |
| bike transmission | edge | 85 | 0.531 | 2658 | 1.00 | joint 0.09 mm |
| | cached | 87 | 1.034 | 521 | 0.84 | joint 0.05 mm |
| caterpillar | edge | 78 | 0.684 | 4246 | 49.7 | joint 0.02 mm |
| | cached | 83 | 0.669 | 1811 | 47.3 | joint 0.03 mm |
| nut on bolt (M20) | edge | 407 | 0.095 | 109 | 0.00 | nut threads down |
| | cached | — | 0.059 | 0 | — | **nut falls through** |

## Conclusions

* **Stacking and piles: clear win.** Persistent, spread-out manifolds of 4 points
  per pair keep stacks still (pyramid drift 0.2 vs 19 mm, 600 gears 6 vs 473 mm,
  cubes at rest to 1e-11 J) with 3-15x fewer contacts, so end-to-end FPS rises
  1.0-3.7x even where the narrow phase itself is slower.
* **Narrow-phase cost.** The kernels are latency bound: a few long serial chains
  (Nelder-Mead over branchy sparse-texture lookups) per pair instead of thousands
  of short ones. Collide is 1.3-2x slower on small and mechanism scenes and only
  faster for complex meshes at scale (343 bunnies: 0.37 vs 1.1 ms). Skipping
  points without relative motion ("frozen") is what makes resting scenes cheap.
* **Mechanisms with sliding or tight-clearance contact: worse.** Cam/follower
  contacts in the analog clock penetrate 2.3 mm (vs 0.05 mm) and stretch joints,
  the tourbillon gear train binds, and a nut on a bolt falls through: helical
  thread contact needs contact in many separate regions, and optimized points
  converge onto the opposite thread flank, flipping the normal. Increasing
  `cache_size` to 16 does not fix it. Bike and caterpillar are on par.
* Curved shapes resting on flat ones (bunny drops) settle less reliably than with
  edge sampling (one or two of 16 keep rocking in some runs) and show more
  transient penetration.

Recommendation: keep `"edge"` as the default; `"cached_sampling"` is worth it for
scenes dominated by stacking/piling of many SDF bodies, not for mechanisms.

## Deviations from the paper that were required

The paper's algorithm as written (cache 4, 1 random sample, repulsion, rank
selection, Eq. 5 normal) does not bring curved or tilted bodies to rest under
PhoenX (single bunny keeps rocking at ~2 rad/s, cube drops stay at 1e-2 J). Each
change below was needed (see `Config` docstrings for the switches):

1. Emit only the kept (cached) points. A transient random sample emitted as a
   contact at a new place every call kicks resting bodies (biggest single fix).
2. Greedy farthest-point selection of the kept points among active ones
   (support polygon instead of deepest points).
3. Depth-guided seed selection over curvature + uniform surface seeds that skips
   regions already covered by cached points, with a local search box per lane.
4. Hysteresis for cached points; skip re-optimization while the shapes have not
   moved relative to each other at a settled point, or while it is farther apart
   than the motion since (also the main performance lever).
5. Curvature-weighted normal blend (a feature resting on a face takes the face
   normal; Eq. 5's choice tilts normals at the rim of shallow overlaps).
6. Pose-based cache advection and relative tolerances (paper constants assume
   unit-size models).

## Tools

* `compare.py SCENE... [--cfg key=value]`: A/B harness.
* `drop_energy.py`: kinetic-energy traces of drop scenes (`SCENE=make_cube_drops`).
* `clock_penetration.py`: lists the most penetrating shape pairs of the analog clock.
* `lane_stats.py SCENE`: per-call pair/lane/iteration/frozen statistics.
* `profile_collide.py SCENE METHOD`: nsys target (100 collide calls after warmup).
