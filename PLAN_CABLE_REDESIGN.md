# Cable bend / twist redesign — 3-point bending with angle-proportional springs

## Motivation

The current cable formulation uses **angular soft rows** with the
Darboux-vector / quaternion-log-map orientation error. For a thin rod link
(`I_link ≈ m·L²/12`, mass 5 g, length 22 mm) the angular row's
`eff_inv = dᵀ·(I₁⁻¹ + I₂⁻¹)·d ≈ 1e7`, so:

- the implicit-Euler Nyquist limit is `k_max = 1/(eff_inv·dt²) ≈ 3 N·m/rad`
  even at 100 substeps, capping the user's `bend_stiffness=1e10` to ≈3
  (we had to opt out of the clamp to even get visible stiffness);
- without the clamp, PGS struggles to converge with `eff_mass_s ≈ 1/eff_inv`
  through long chains.

Meanwhile, the **point-anchor** rows used by ball-socket / revolute /
prismatic converge robustly even at very high `hertz` — translational mass
(`1/m`) dominates `eff_inv`, so the Nyquist limit is comfortably high
(`k_max ≈ m/dt² ≈ 1.8e5 N/m` for the same link), and PGS propagates point
locks through chains in a few iterations.

## Existing anchor convention (recap, then we generalise)

- **Revolute joint = 2 anchors** (anchor-1 + anchor-2, both *on* the
  hinge axis). Anchor-1 is a 3-row hard point lock (joint origin);
  anchor-2 is a 2-row tangent lock at distance `L_axial` along the
  axis (catches bend of any direction; axial twist remains free
  because the anchor sits on the axis).
- **Prismatic joint = 3 anchors** (anchor-1 on slide axis, anchor-2 on
  slide axis at distance `L_axial`, anchor-3 *off* the slide axis at
  distance `L_perp`). Anchor-1 + anchor-2 each contribute 2-row
  tangent locks (lateral lock); anchor-3 contributes a 1-row scalar
  lock that catches axial rotation (twist).

So: 2 anchors lock bend, the **3rd anchor locks twist**. Cable wants
both, soft.

## New formulation: 3-point bending with linear springs

Joint between body 1 (parent) and body 2 (child):

```
  body 1                           body 2

      a-1*                              a-1*           anchor-1 (joint origin):
      / |                              / |               HARD 3-row ball-socket
     /  | n_hat                       /  | n_hat
    /   |                            /   |              anchor-2 (along n_hat,
   *----+---a-2*  --- bend --->  ___*----+-a-2          distance L_axial):
   |    |                       /  |                      SOFT 2-row tangent
   |a-3*|                       a-3*                      spring (bend lock)
   +----+                       +---+
                                                       anchor-3 (perpendicular,
                                                        distance L_perp):
                                                          SOFT 1-row scalar
                                                          spring (twist lock)
```

- `anchor-1`: 3-row hard ball-socket lock (existing). Pins joint position.
- `anchor-2`: a body-2 point at `la1 + L_axial·n_hat_rest`. Spring
  pulls it back to body-1's anchor-2 world position. Bends ⇒ tangent
  displacement ⇒ restoring force ⇒ restoring torque about anchor-1.
  Catches both bend axes in one 2-row tangent lock.
- `anchor-3`: a body-2 point at `la1 + L_perp·t1_rest` (perpendicular
  to `n_hat`). Spring pulls it back along the second perpendicular `t2`.
  Twist about `n_hat` ⇒ scalar displacement ⇒ restoring force ⇒
  restoring torque about `n_hat`.

Three angular DoF total (2 bend + 1 twist) covered by two soft anchor
springs at predictable lever arms.

## Why this converges better than Darboux soft rows

Per-row `eff_inv` for a point anchor is

`eff_inv = (1/m1 + 1/m2) + skew(r1)ᵀ·I1⁻¹·skew(r1) + skew(r2)ᵀ·I2⁻¹·skew(r2)`

dominated by `1/m` (translational mass) for typical thin-rope geometry,
**not** the tiny `I⁻¹`. For a 5 g link, `1/m = 200`; at `dt = 1/6000`,
Nyquist `k_max ≈ 1/(200 · 2.78e-8) ≈ 1.8e5 N/m`. At lever arm `L = 1
cm`, that's `k_eff ≈ 18 N·m/rad` — about an order of magnitude over what
the angular formulation gave at the clamp on, and crucially the row
**still converges** with the clamp on.

For `bend_stiffness=1e10` the user wants, clamping kicks in — but the
clamp now bounds at a useful stiffness, not a useless one.

## Angle-proportional spring (not just first-order)

Naively setting `bias = chord_length · idt` gives a spring whose
**force** is linear in chord length but whose **torque** about the
joint is `τ ≈ k_lin · L² · sin(θ)` — fine for small θ, *saturates* at
θ = π/2 (`τ = k_lin·L²`), and *wraps back to zero* at θ = π. Not what
the user wants from a "stiffness".

The fix is to make the spring's bias proportional to θ rather than to
the chord length:

```
chord = ‖d_tan‖     # tangent-projected anchor-2 displacement
sin_half = chord / (2·L)        # exact for the chord-of-θ relation
sin_half = clamp(sin_half, -1, 1)
θ        = 2·asin(sin_half)
bias_mag = (θ · L) · idt        # arc-length per dt instead of chord per dt
bias_vec = (d_tan / chord) · bias_mag      # direction = chord direction
```

What this does to the per-iter PGS update:

```
lam = -eff_mass · (Jv − bias + γ·acc)
```

`Jv` is still the linear tangent velocity (kinematics; no choice).
`bias` is the **target velocity** that would close the angle in one step
along the chord direction. By using `θ·L` instead of `chord`, the
target *magnitude* grows linearly with θ even past 90°, so the
restoring torque the row converges to is `τ ≈ k_lin · L · θ · L =
k_lin · L² · θ` — linear in θ all the way.

(Direction is still the chord direction, which matches the natural
motion the joint takes when restoring; only the magnitude is
re-scaled.)

For twist (anchor-3 scalar): the displacement is along a single
direction `t2`, and `chord_twist = (p3_b2 - p3_b1)·t2`. Same trick:

```
sin_half = chord_twist / (2·L_perp)
θ_twist  = 2·asin(clamp(sin_half, -1, 1))
bias_mag = (θ_twist · L_perp) · idt
```

Cost: one `asin` per cable joint per prepare. ~6 ns on Blackwell — utterly
negligible vs. the rest of the joint solve.

For small θ, `2·asin(x)·L = 2L·x + O(x³) ≈ chord`, so we recover the
naive linear-spring behaviour at small angles. For large θ the bias
keeps growing linearly, matching the user's "torque ∝ θ" intent.

## API choice: hertz/damping_ratio vs stiffness/damping

Recommendation:

- **Bend / twist: use `(stiffness, damping)` absolute** via
  `pd_coefficients_split`. User specifies stiffness in N·m/rad,
  damping in N·m·s/rad (angular convention, matches what the existing
  cable scenes pass). Internally we convert to linear:
  `k_lin = k_ang / L²`, `c_lin = c_ang / L²`. Keep `clamp_nyquist=True`.
  - Why angular API even though internal storage is linear: existing
    `bend_stiffness=1e6` user scenes keep producing the same physical
    behaviour. No churn.
- **Anchor-1 lock: keep hertz/damping_ratio** (current behaviour;
  rigid lock).

## Implementation steps

### 1. Cable joint storage

Reuse the existing revolute + prismatic anchor-2 / anchor-3 storage
slots that the column already has — cable mode doesn't currently use
them, so no widening:

- Anchor-2: `_OFF_R2_B1`, `_OFF_R2_B2`, `_OFF_T1`, `_OFF_T2`,
  `_OFF_S_INV` (reuse 2x2 Schur block layout from revolute).
- Anchor-3: `_OFF_R3_B1`, `_OFF_R3_B2`, `_OFF_S_SCALAR_INV` (reuse
  scalar Schur from prismatic).
- PD coefficient triples per row group:
  - bend: `(γ, eff_mass_s, damp_mass)` (3 dwords) — bias is computed
    fresh each prepare from the angle, not stored.
  - twist: same 3 dwords.
- Body-frame rest pose: anchor-2 at `la1_b{1,2} + L_axial·n_hat`
  captured at init from each body's frame (same pattern as revolute
  uses for `la2_b{1,2}`). Anchor-3 at `la1_b{1,2} + L_perp·t1_rest`.
  No new dwords vs. revolute+prismatic.

Drop these slots (currently used by Darboux):
- `_OFF_ACC_KAPPA` (3-vec angular accumulator)
- `_OFF_CABLE_*_DAMP_MASS_*` (bend1, bend2, twist)
- ut_ai-stored row directions (replaced by the standard t1/t2/n_hat tangent
  basis from revolute)

### 2. Init kernel

Extend `actuated_double_ball_socket_initialize_kernel` cable branch:
- Capture `la1_b{1,2}` (existing).
- Capture `n_hat_rest` = unit(anchor2 - anchor1), in *each* body's local
  frame at init time (new — current cable also does something similar
  for the Darboux reference frame).
- Capture `la2_b{1,2} = la1_b{1,2} + L_axial · n_hat_rest_b{1,2}`.
- Capture an arbitrary perpendicular `t1_rest` to `n_hat_rest`
  (consistent with revolute's basis-from-axis helper) and
  `la3_b{1,2} = la1_b{1,2} + L_perp · t1_rest_b{1,2}`.
- Lever arms are joint-construction time constants. Either store
  `L_axial`, `L_perp` per-cid (2 floats) or fold them into the captured
  body-local anchors (no extra storage; recover `L_axial = ‖la2 −
  la1‖` at prepare).

### 3. Prepare

```
_cable_prepare_at(...):
  # Anchor-1: existing 3-row ball-socket prepare (hard).
  ...

  # Tangent basis from current anchor-1 → anchor-2 axis.
  n_hat = unit(p2_b1 - p1_b1)  # world frame
  t1, t2 = tangent_basis(n_hat)
  store r2_b{1,2}, t1, t2

  # Anchor-2 effective mass (2x2 Schur, same as revolute).
  build A2_inv (2x2)
  L_axial = ‖la2 − la1‖
  k_bend_lin = k_bend_ang / L_axial²
  c_bend_lin = c_bend_ang / L_axial²
  γ_bend, _bias_unused, eff_mass_bend, damp_bend =
      pd_coefficients_split(k_bend_lin, c_bend_lin, 0,
                            eff_inv_bend, dt, True)

  # Bend bias: angle-proportional (see "Angle-proportional spring")
  d_tan = (p2_b2 - p2_b1) - ((p2_b2 - p2_b1)·n_hat)·n_hat
  chord = ‖d_tan‖
  sin_half = clamp(chord / (2·L_axial), -1, 1)
  θ_bend = 2·asin(sin_half)
  if chord > eps:
      bias2_2vec = (d_tan / chord) projected to (t1, t2) · θ_bend·L_axial · idt
  else:
      bias2_2vec = 0

  # Anchor-3 (twist) — analogous, scalar.
  ...
  L_perp = ‖la3 − la1‖
  k_twist_lin = k_twist_ang / L_perp²
  ... pd_coefficients_split(k_twist_lin, c_twist_lin, 0, eff_inv_twist, dt, True)
  chord_twist = (p3_b2 - p3_b1)·t2_world
  sin_half_t = clamp(chord_twist / (2·L_perp), -1, 1)
  θ_twist = 2·asin(sin_half_t)
  bias3 = sign(chord_twist)·θ_twist·L_perp · idt

  store all coefficients + biases + accumulators
```

### 4. Iterate

`_cable_iterate_at`:
- Anchor-1: existing 3-row hard lock impulse.
- Anchor-2 bend (2 soft 1D rows in (t1, t2)):
  ```
  for i in (0, 1):
      jv_i = (J_i · velocity_state)
      lam_i = -eff_mass_bend · (jv_i - bias2_i + γ_bend · acc2_i)
      acc2_i += lam_i
      apply lam_i along t_i with anchor-2 lever arm
  ```
- Anchor-3 twist (1 soft scalar row along t2):
  ```
  jv_t = (J_twist · velocity_state)
  lam_t = -eff_mass_twist · (jv_t - bias3 + γ_twist · acc3)
  acc3 += lam_t
  apply lam_t along t2 with anchor-3 lever arm
  ```
- Damping pass (`use_bias=False`): per row, `lam = -damp_mass·jv`. No
  bias, no `γ·acc` on the relax (the spring half already settled
  `acc` to its equilibrium during the main solve).

### 5. Drop angular-row code

Remove `_cable_capture_orientation_at_init`, the bend/twist Darboux
formulas in `_cable_prepare_at`, and the angular soft-row iterate
block. ~150 lines deleted.

### 6. WorldBuilder API

`add_joint(... mode=JointMode.CABLE, bend_stiffness=K, bend_damping=C,
twist_stiffness=K, twist_damping=C)` — parameter names and units
unchanged (angular). User scenes work as before.

### 7. Tests

- Existing `test_cable_joint.py` analytical tests — should still
  pass with the same parameters; the small-angle behaviour is
  identical to first-order, only large-angle behaviour changes.
  - May need to retune one or two damping ratios to recover the
    exact zeta the analytical decrement test expects (the new
    formulation's effective damping has an `L²` factor matched to the
    new `k_lin`, so the *ratio* is preserved — but integer-rounding
    in the test thresholds may need a delta bump).
- New test: rigid-rod cantilever at high stiffness. 64-link chain,
  bend_stiffness=1e6, gravity on, settled droop angle bounded by
  `m·g·L_chain² / (2 · k_bend_ang)` to within 10%.
- New test: large-angle bend (start at 90° bend) settles with
  oscillation period matching `2π·sqrt(I_link / k_bend_ang)` (no
  saturation that the chord-only spring would have shown).
- `example_bend_stiffness_sweep.py` — visually verify rigid-rod
  behaviour at 1e10 stiffness. Should be ≈ rigid for all 6 strands.

## Open questions

1. **Lever arms**: I propose `L_axial = ‖anchor2 − anchor1‖` (already
   user-supplied), `L_perp = L_axial` (a sane default; same regime).
   Acceptable, or expose `L_perp` separately?
2. **API units**: keep `bend_stiffness` / `twist_stiffness` as
   **angular** (N·m/rad) for back-compat? I assume yes — internal
   conversion `k_lin = k_ang / L²`, no user-visible change.
3. **Anchor-3 perpendicular direction**: pick a canonical
   perpendicular to `n_hat_rest` at init, store in body-local frames,
   recompute world-frame each prepare. Reuse the prismatic
   `_tangent_basis_from_anchor3` helper or add a cable-specific
   `_perpendicular_unit` helper.
4. **Bias direction**: I'm proposing the bias direction to follow the
   chord (current displacement direction). Alternative: use the *first
   eigenvector* of the orientation error, which would be more "true"
   for large angles but adds an eigendecomposition. The chord
   direction matches what the joint will move toward when restoring,
   and PGS is robust to small directional inaccuracy. I'd go with
   chord direction unless you want the eigen path.
5. **Prismatic anchor-3 reuse**: the prismatic joint uses a
   *hard-locked* anchor-3 (1-row scalar lock with bias). Cable's new
   anchor-3 is **soft** (PD spring). The constraint structure (J,
   sign convention) is identical; only the impulse formula differs.
   I'll factor a shared `_anchor3_iterate_soft_scalar` helper if it
   pays off, otherwise inline.

## Order of work

1. Restore `clamp_nyquist=True` everywhere except cable bend/twist. ✅
   committed.
2. New cable prepare/iterate in scratch — port revolute anchor-2
   tangent helpers + prismatic anchor-3 scalar helpers. Add the
   angle-proportional bias.
3. Wire init / prepare / iterate into `JointMode.CABLE`. Drop
   Darboux angular-row code.
4. Run cable analytical tests; tune any threshold drift. Verify the
   new formulation matches small-angle period/damping within solver
   slop.
5. New high-stiffness test (cantilever droop bound).
6. Run `example_bend_stiffness_sweep.py` at 1e10 stiffness; verify
   rigid-rod behaviour.

Estimated diff: +200 / −250 lines (net deletion since the new path
reuses revolute/prismatic primitives).

## Open questions to confirm before I start coding

- Q1 — lever arm `L_perp` policy (default to `L_axial` vs configurable).
- Q2 — keep angular stiffness API (`N·m/rad`) and convert internally,
  or change to linear (`N/m`) and document migration.
- Q3 — angle-proportional bias direction: chord direction (simple,
  what I described) vs eigen-direction of the orientation error (more
  accurate at large angles, costs an eigendecomposition).
- Q4 — should I write a small standalone unit test exercising the
  angle-proportional bias before wiring into cable, e.g. drop the new
  helper into a single-joint scene and verify torque-vs-angle linearity
  at θ ∈ {10°, 45°, 90°, 135°}?
