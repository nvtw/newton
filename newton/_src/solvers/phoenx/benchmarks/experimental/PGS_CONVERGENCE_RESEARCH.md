# PhoenX colored-PGS convergence research

This is a living research log for improving the standard small-step PhoenX
solver on ill-conditioned rigid systems. Candidate methods must remain
GPU-efficient, CUDA-graph capturable, compatible with projected constraints,
and physically correct.

## Benchmarks and measurements

The primary probe is `bench_pgs_motor_chain.py`, a 96-link realistic-inertia
revolute chain. The original 250-link case remains a scaling confirmation.
Every timed result replays an end-to-end CUDA graph and reports wall time,
tip sag, segment error, and residual linear/angular speed.

The acceptance matrix will also include:

- multiple independent and branched articulations, to measure useful GPU
  parallelism rather than single-chain serialization;
- dense rigid stacks/Kapla scenes, to cover unilateral contact and friction;
- mass-ratio and closed-loop mechanisms, to expose conditioning and topology;
- momentum and energy checks, to reject numerically fast but unphysical schemes.

## Baseline evidence

The original 250-link, 500-substep, four-sweep example drooped 9.92 m after
100 frames. Initial controlled results on the 96-link benchmark are:

| Experiment | Result | Decision |
|---|---:|---|
| SOR 1.4 vs 1.0 | slightly worse sag at equal cost | not sufficient |
| SOR 1.8 | catastrophic divergence | reject fixed aggressive SOR |
| 50 substeps x 8 sweeps | 5.91 m sag, 3.03 ms/frame | fast but inaccurate |
| 100 substeps x 4 sweeps | 4.52 m sag, 4.50 ms/frame | baseline |
| 200 substeps x 2 sweeps | 1.51 m sag, 7.33 ms/frame | effective, costly |
| mass splitting, one regular color | 4.06 m sag, 36.57 ms/frame and high residual speed | reject for chains |
| one topology level per joint | 4.35 m sag, 61.63 ms/frame | reject naive serialization |

At 100 substeps, increasing PGS sweeps from 1 to 16 only reduced 30-frame sag
from 1.85 m to 1.69 m while increasing time from 3.25 to 9.07 ms/frame. This
is the signature of a slow global mode, not a local row-solve deficiency.

## Candidate taxonomy

### Highest priority

1. **Symmetric-PGS Chebyshev acceleration.** A forward/backward pair has a
   real spectrum suitable for the low-storage Chebyshev recurrence. It needs
   no dot-product reductions and therefore maps well to CUDA. Huamin Wang's
   analysis reports order-of-magnitude gains and specifically notes that a
   forward plus backward GS pair avoids the complex-spectrum failure of a
   single GS sweep.
2. **Safeguarded Anderson acceleration.** Treat one projected symmetric-PGS
   pair as a nonlinear fixed-point map. Recent multibody work reports PSOR
   accelerated to first-order/Krylov-class performance. A small history
   (2--5) is plausible on GPU, but reductions, projection consistency, restart
   rules, and multiplier/body-state consistency must be measured.
3. **Topology-aware coarse correction.** Minimal coloring is a good smoother
   but poor at propagating low-frequency load through long chains. Use PGS for
   local/high-frequency error and solve an aggregated articulation/tree
   correction between sweep groups. Parallel prefix/tree contraction or a
   small block-tridiagonal coarse solve is preferable to one color per depth.
4. **Batched bidirectional wavefronts.** Root-to-leaf and leaf-to-root levels
   can propagate load, but only when many branches/worlds keep lanes occupied.
   The naive single-chain version was 14x slower and is rejected.

### Secondary experiments

- adaptive spectral-radius estimation from residual norm ratios;
- safeguarded Aitken/Barzilai-Borwein scalar relaxation;
- nonstationary Chebyshev-Jacobi/PSOR schedules;
- randomized or periodically rotated color order;
- shock-propagation and gravity/topology priority orderings;
- block grouping across neighboring joints, with exact small block solves;
- two-level additive/multiplicative Schwarz and aggregation AMG;
- PGS-preconditioned projected CG/MINRES for bilateral islands, falling back
  to PGS at active-set changes;
- augmented-Lagrangian/ADMM outer correction with PGS local projections;
- mixed precision only after convergence behavior is solved.

### Already present or unsuitable alone

- warm starting, local joint blocks, symmetric color traversal, fixed SOR,
  small steps, and copy-state mass splitting already exist;
- adding more small steps works but pays repeated prepare/integration cost;
- naive sequential topology coloring destroys GPU utilization;
- mass splitting targets parallel consistency/jitter and did not resolve this
  chain's global mode.

## Key sources

- Wang, *A Chebyshev Semi-Iterative Approach for Accelerating Projective and
  Position-Based Dynamics* (2015):
  https://wanghmin.github.io/publication/wang-2015-csi/Wang-2015-CSI.pdf
- Fusai and Tasora, *Anderson-accelerated fixed-point method for solving
  multibody problems with bilateral constraints and non-smooth frictional
  contacts* (2025): https://doi.org/10.1016/j.cma.2025.118576
- Poulsen, Abel, and Erleben, *Heuristic Convergence Rate Improvements of the
  Projected Gauss-Seidel Method for Frictional Contact Problems* (2010):
  https://iphys.wordpress.com/wp-content/uploads/2012/09/paper.pdf
- Tonge et al., *Mass Splitting for Jitter-Free Parallel Rigid Body
  Simulation* (2012): https://doi.org/10.1145/2185520.2185601
- Lan et al., *JGS2: Near Second-order Converging Jacobi/Gauss-Seidel for GPU
  Elastodynamics* (2025): https://arxiv.org/abs/2506.06494
- Mazhar et al., comparison of PGS, projected Jacobi, APGD, and Newton methods
  for frictional multibody DVI (2017):
  https://doi.org/10.1016/j.cma.2016.12.023
- NVIDIA AmgX, GPU aggregation AMG and preconditioned iterative methods:
  https://research.nvidia.com/publication/2015-10_amgx-library-gpu-accelerated-algebraic-multigrid-and-preconditioned-iterative

## Next implementation gate

Prototype symmetric-PGS Chebyshev with complete state consistency: body
velocities and every accumulated constraint multiplier must represent the same
accelerated iterate. Start with bilateral joint-only islands, estimate the
spectral radius from unaccelerated residual ratios, and add divergence/restart
guards before enabling contacts. Compare at equal wall time, not only equal
sweep count.
