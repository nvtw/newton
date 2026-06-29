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
| cross-substep impulse prediction, beta 0.8 | 0.05% sag gain at one sweep, 6% slower | reject |
| balanced 3-color, 8 sweeps | 4.13 m sag at 4.06 ms vs greedy 4-sweep 4.52 m at 4.39 ms | retain as scheduler candidate |
| one topology level per joint | 4.35 m sag, 61.63 ms/frame | reject naive serialization |

At 100 substeps, increasing PGS sweeps from 1 to 16 only reduced 30-frame sag
from 1.85 m to 1.69 m while increasing time from 3.25 to 9.07 ms/frame. This
is the signature of a slow global mode, not a local row-solve deficiency.

## Candidate taxonomy

### Highest priority

1. **Factor-2 articulation multilevel correction.** The measured 96-joint
   Delassus operator has symmetric block-GS spectral radius 0.99999994 and
   condition number 9.8e7. A
   depth-linear factor-2 coarse space with an exact half-size solve reduces
   the 12-cycle residual to 0.0019, versus 2.78 for SGS. A recursive V-cycle
   with two pre/post smooths reaches 0.56 and supplies an O(N) starting point.
2. **Safeguarded Anderson acceleration.** Depth 2--4 reaches residual 0.61
   after 60 symmetric pairs versus 1.55 for SGS, without scene-specific
   tuning. It is also a candidate coarse-level solver.
3. **Symmetric-PGS Chebyshev acceleration.** A forward/backward pair has a
   real spectrum suitable for the low-storage Chebyshev recurrence. It needs
   no dot-product reductions and therefore maps well to CUDA. Huamin Wang's
   analysis reports order-of-magnitude gains and specifically notes that a
   forward plus backward GS pair avoids the complex-spectrum failure of a
   single GS sweep.
   Direct application is rejected on this chain: the nearly-unit spectral
   radius causes persistent oscillation rather than acceleration. It may remain
   useful after multilevel preconditioning narrows the spectrum.
4. **Balanced cyclic/wavefront coloring.** Three equally sized colors on the
   96-link chain are substantially cheaper than the imbalanced greedy layout,
   allowing twice the sweeps at lower wall time. Generalization must preserve
   independence and work on arbitrary graphs. Fully serial depth coloring is
   still rejected.

### Secondary experiments

- adaptive spectral-radius estimation from residual norm ratios;
- safeguarded Aitken/Barzilai-Borwein scalar relaxation;
- nonstationary Chebyshev-Jacobi/PSOR schedules;
- randomized or periodically rotated color order;
- shock-propagation and gravity/topology priority orderings;
- block grouping across neighboring joints, with exact small block solves;
- two-level additive/multiplicative Schwarz and aggregation AMG;
- flexible GMRES on multilevel-preconditioned bilateral islands, falling
  back to PGS at active-set changes;
- augmented-Lagrangian/ADMM outer correction with PGS local projections;
- mixed precision only after convergence behavior is solved.

### Already present or unsuitable alone

- warm starting, local joint blocks, symmetric color traversal, fixed SOR,
  small steps, and copy-state mass splitting already exist;
- adding more small steps works but pays repeated prepare/integration cost;
- naive sequential topology coloring destroys GPU utilization;
- mass splitting targets parallel consistency/jitter and did not resolve this
  chain's global mode.
- extrapolating converged warm-start impulses across tiny substeps was stable
  but did not target the slow spatial mode;
- the existing exact device articulation LDLT is graph-capturable for tiny
  systems but expands a long chain into hundreds of level launches, making it
  unsuitable as a per-substep correction in its current form;
- CG with no preconditioner, block Jacobi, symmetric PGS, or the practical
  recursive V-cycle does not reduce the residual in a 12-iteration budget.

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
- Miyamoto et al., accelerated modulus-based GS for sparse rigid-body LCPs:
  https://arxiv.org/abs/1910.09873
- Kamino, GPU proximal-ADMM for coupled multibody systems in Newton/Warp:
  https://arxiv.org/abs/2603.16536
- NVIDIA AmgX, GPU aggregation AMG and preconditioned iterative methods:
  https://research.nvidia.com/publication/2015-10_amgx-library-gpu-accelerated-algebraic-multigrid-and-preconditioned-iterative

## Next implementation gate

Prototype a GPU factor-2 articulation hierarchy with depth-linear
prolongation and Galerkin coarse operators. Use colored PGS as the fine smoother
and compare recursive V-cycles, fixed-count coarse CG, and depth-2 Anderson on
the coarse equations. Preserve exact impulse/body-state consistency and test at
equal wall time before extending the correction beyond bilateral joint islands.
