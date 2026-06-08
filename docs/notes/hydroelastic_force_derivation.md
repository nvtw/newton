# Hydroelastic per-face force: derivation

This note explains why Newton's hydroelastic contact-stiffness formula is

```
c_stiffness = area · p_face / (2 · |depth|)
```

and why the previous form `c_stiffness = area · k_eff(k_a, k_b)` was correct
only for symmetric pairs and silently wrong otherwise.

## 1. The hydroelastic model

Following Elandt et al. (2019), each elastic body `i` carries a scalar
**pressure field** `p_i(x)` defined inside the body. For an SDF-based
linear law: `p_i(x) = -k_i · φ_i(x)`, where `φ_i` is the signed distance
(negative inside, so `p_i` is positive inside).

The **contact patch** `C` is the iso-pressure surface

$$
C = \{x \in \Omega_A \cap \Omega_B \;:\; p_A(x) = p_B(x)\}.
$$

By Newton's third law, both bodies exert the same pressure across `C`.
Define `p_face(x) = p_A(x) = p_B(x)` for `x ∈ C`. The total contact force is

$$
\mathbf{F} = \int_C p_{\text{face}}(x)\, \mathbf{n}(x)\, dA.
$$

In Newton, marching cubes triangulates `C` so the integral becomes a sum
over faces. **Per face**: `F_face = area · p_face · n`.

This is the only force law in play. There is no separate "spring stiffness"
- the spring/material model is fully encoded by `p_i(·)`.

## 2. Newton's solver convention

Newton's solver consumes two per-contact quantities, `c_stiffness` and
`contact_distance`, and applies

$$
\mathbf{F} = c_{\text{stiffness}} \cdot (-\text{contact\_distance}) \cdot \mathbf{n}.
$$

The hydroelastic generator stores

$$
\text{contact\_distance} = 2 \cdot \text{depth}, \qquad
\text{depth} = \varphi_B(x_{\text{face}}) \;<\; 0 \text{ when penetrating}.
$$

Note `depth = φ_B` is **one side's** signed distance (the SDF of body B
sampled at the face centroid), not the total gap.

For a penetrating contact (`depth < 0`):
$F = c_{\text{stiffness}} \cdot 2 \cdot |\text{depth}|$.

## 3. Deriving c_stiffness

We want `F_face = area · p_face` (Section 1). Setting that equal to the
solver's force expression:

$$
c_{\text{stiffness}} \cdot 2 \cdot |\varphi_B| = \text{area} \cdot p_{\text{face}}
\;\;\Longrightarrow\;\;
\boxed{\;c_{\text{stiffness}} = \frac{\text{area} \cdot p_{\text{face}}}{2 \cdot |\varphi_B|}\;}
$$

The factor of 2 absorbs the `contact_distance = 2·depth` convention. The
result holds for **any** pressure law `p` and **any** stiffness pair
`(k_A, k_B)`.

For the linear law `p_i = -k_i · φ_i`, evaluating at the iso-surface gives
`p_face = k_B · |φ_B|`, so

$$
c_{\text{stiffness}}^{\text{linear}} = \frac{\text{area} \cdot k_B}{2}.
$$

## 4. Why the previous formula `c_stiffness = area · k_eff` failed for asymmetric pairs

The springs-in-series intuition: total compression `pen_total = |φ_A| + |φ_B|`,
combined stiffness $k_{\text{eff}} = k_A k_B / (k_A + k_B)$, force per area
$p_{\text{face}} = k_{\text{eff}} \cdot \text{pen\_total}$.

Plugging into the solver expression:

$$
F = (\text{area} \cdot k_{\text{eff}}) \cdot 2 |\varphi_B|
   = 2 \cdot \text{area} \cdot k_{\text{eff}} \cdot |\varphi_B|.
$$

For this to equal `area · p_face = area · k_eff · pen_total`, we need
`pen_total = 2 · |φ_B|`. From the iso-pressure condition $k_A |\varphi_A| = k_B |\varphi_B|$:

$$
|\varphi_A| = \frac{k_B}{k_A} \cdot |\varphi_B|, \qquad
\text{pen\_total} = |\varphi_A| + |\varphi_B| = \left(1 + \frac{k_B}{k_A}\right) |\varphi_B|.
$$

So `pen_total = 2 · |φ_B|` **iff** $k_A = k_B$ (symmetric). For any
asymmetric pair, `area · k_eff` scales force incorrectly by

$$
\frac{F_{\text{old}}}{F_{\text{correct}}}
  = \frac{2 |\varphi_B|}{\text{pen\_total}}
  = \frac{2 k_A}{k_A + k_B}.
$$

Concrete cases:
| `k_A : k_B` | `F_old / F_correct` |
| --- | --- |
| 1 : 1  | 1.000 (correct by accident) |
| 2 : 1  | 1.333 |
| 3 : 1  | 1.500 |
| 10 : 1 | 1.818 |
| 1 : 10 | 0.182 |

The new formula `c_stiffness = area · p_face / (2|φ_B|)` does **not**
reconstruct `pen_total`; it uses the iso-surface pressure value directly,
which is well-defined from `(φ_B, k_B)` alone via `p_face = k_B · |φ_B|`
(or `p_face = k_A · |φ_A|`, equal by the iso-pressure condition).

## 5. Aggregate stiffness for reduced contacts

Contact reduction collapses N raw faces into K representative contacts.
We require their summed solver force to equal the unreduced total:

$$
\sum_{j=1}^{K} c_{\text{stiffness}}^{\text{shared}} \cdot 2 |\varphi_j^{\text{red}}|
\;=\; \sum_{i=1}^{N} \text{area}_i \cdot p_{\text{face},i}
\;=\; |\mathbf{F}_{\text{agg}}|.
$$

Solving for `c_stiffness_shared`:

$$
\boxed{\;c_{\text{stiffness}}^{\text{shared}}
   = \frac{|\mathbf{F}_{\text{agg}}|}{2 \cdot \sum_{j} |\varphi_j^{\text{red}}|}\;}
$$

where $\mathbf{F}_{\text{agg}}$ is accumulated in the generate kernel as
$\sum_i \text{area}_i \cdot p_{\text{face},i} \cdot \mathbf{n}_i$. The
factor of 2 is the same `contact_distance` correction as Section 3.

## 6. Margin contacts (`depth ≥ 0`)

For non-penetrating "margin" contacts the user-supplied `pressure_func` is
only required to be defined and monotone (per `HydroelasticSDF.Config`
docstring); its value is not physically meaningful. Margin contact
stiffness uses the linear-law harmonic mean as a constraint
regularization:

$$
c_{\text{stiffness}}^{\text{margin}} = A_{\text{margin}} \cdot k_{\text{eff}}(k_A, k_B).
$$

This is unchanged from the pre-refactor behavior.

## References

- Elandt, R., Drumwright, E., Sherman, M., Ruina, A. "A pressure field
  model for fast, robust approximation of net contact force and moment
  between nominally rigid objects." *IROS 2019*. The original
  hydroelastic contact theory.
- Drake hydroelastic documentation (Toyota Research Institute / MIT).
  See https://drake.mit.edu/doxygen_cxx/group__hug__contact__model.html
  and the "Hydroelastic Contact Model" technical notes for the discrete
  marching-cubes formulation that Newton mirrors.
