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
   the 12-cycle residual to 0.059, versus 2.78 for SGS. The earlier 0.0019
   measurement included an accidental extra fine forward sweep and is
   superseded. A recursive V-cycle with two pre/post smooths reaches 0.56 and
   supplies an O(N) starting point.
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

## Half-resolution coarse-solve evidence

The factor-2 depth-linear Galerkin operator on the 96-joint chain contains 49
5-by-5 blocks. It is exactly block tridiagonal (145 nonzero blocks, maximum
block distance one) and remains ill-conditioned at 2.1e7. A block Thomas solve
matches the dense solve to 7.8e-12 relative solution error and produces the
same 0.0594 outer residual after 12 one-pre/one-post cycles.

Replacing that exact coarse solve with local work leaves much of the global
mode unresolved:

| Half-resolution solver per outer cycle | Final relative residual |
|---|---:|
| one block-SGS pair | 1.399 |
| two block-SGS pairs | 1.225 |
| four block-SGS pairs | 1.055 |
| eight block-SGS pairs | 0.916 |
| Anderson depth 2, four updates | 0.643 |
| Anderson depth 2, eight updates | 0.421 |
| exact block-tridiagonal solve | 0.0594 |

A parallel cyclic-reduction prototype matches the dense solution to 1.3e-11
in float64, but native float32 is numerically unacceptable: its relative coarse
residual is 1.4e3, and diagonal scaling plus fixed iterative refinement is
erratic. Sequential block Thomas is substantially more stable in float32
(0.0136 coarse residual and 3.3% solution error). The next GPU prototype should
therefore use a coarse block-Thomas megakernel per articulation island, exposing
parallelism across islands/worlds while keeping the dependent short-chain solve
inside one launch. More local coarse smoothing is work-inefficient and does not
remove the measured long-wavelength mode.

## Balanced direct-correction experiments

The existing device block Cholesky uses MECA tie-breaking that peels this path
from one end. The 96-block chain therefore has 96 dependency levels and a
captured factor/solve replay costs 3.09 ms. Repeated odd-node elimination adds
89 fill blocks (184 versus 95 strict-lower blocks) but reduces the elimination
tree to eight levels with widths `[1, 1, 2, 4, 8, 16, 32, 32]`. The same replay
then costs 0.502 ms, a 6.2x speedup, and its float32 residual is substantially
better than the one-ended ordering.

Used inside PhoenX small steps, this is a strong convergence oracle:

| Correction schedule (100 substeps, one PGS sweep) | 30-frame result | Time |
|---|---:|---:|
| no global correction | about 1.85 m sag | about 3.25 ms/frame |
| correction every two substeps, regularization 0.001 | 0.059 m sag, low speed | 30.8 ms/frame |
| correction every ten, regularization 0.01, relaxation 0.1 | 1.79 m sag | 8.74 ms/frame |
| correction every five, regularization 0.01, relaxation 0.1 | 1.74 m sag | 14.4 ms/frame |

Full intermittent corrections are unstable because the native-float operator
is still extremely ill-conditioned. Regularization and impulse relaxation
bound them, but also make sparse corrections too weak to beat extra small
steps at equal wall time. A once-per-frame bias-free correction likewise gives
only a few percent sag improvement once regularized enough to remain physical.

Two megakernel layouts were also measured. A one-thread natural-path block
solve is exact but costs 76.2 ms/frame at stride two; warp-cooperative block
algebra reduces this to 46.8 ms. A warp-fused balanced sparse solve retains
pivot parallelism and reaches 27.7 ms, only 10% faster than the eight-level
launch sequence. General sparse fusion inside the monolithic articulation
module was rejected after module compilation exceeded three minutes.

The direct correction proves that eliminating the long spatial mode solves the
physical failure, but refactorization on every correction is too expensive.
Modified-Newton factor reuse was also tested. Full stale-factor corrections
produce NaNs after even one reuse interval. With regularization 0.01,
relaxation 0.1, solves every two substeps, and refactorization every ten, the
method is stable and reduces sag to 1.61 m, but costs 15.4 ms/frame. Fusing the
reused forward/back substitutions lowers that to 13.7 ms/frame, still far
worse than extra small steps at equal time. Reused direct factors are therefore
not the production answer.

Every correction remains a paired `J^T lambda` body impulse, so the direct
experiments preserve internal linear/angular momentum by construction. The
next implementation candidate should obtain the global-mode benefit without a
full articulation factorization: a matrix-free factor-2 coarse correction with
fixed local work and explicit momentum tests.

## Factor-2 local Galerkin correction breakthrough

The first competitive physical method is a factor-2 path correction built from
local fine 5-by-5 blocks. It assembles the 49-block depth-linear Galerkin
operator, performs fixed red/black coarse block-GS sweeps in one captured
kernel, prolongates the impulse, and applies it through the original joint
Jacobians. The corrected GPU operator matches `P^T A P` to 7.1e-8 relative
error and the restricted RHS exactly. An earlier terminal-node mapping bug was
found by this comparison and removed before physical conclusions were drawn.

Equal-time 60-frame results are:

| Method | Time | Tip sag | RMS position row violation | RMS angular row violation |
|---|---:|---:|---:|---:|
| classic PGS, 200 substeps x 2 sweeps | 7.27 ms | 1.507 m | 0.831 mm | 0.0155 rad |
| coarse correction, 12 color sweeps, stride 2 | 7.47 ms | 1.291 m | 0.254 mm | 0.00535 rad |
| coarse correction, 16 color sweeps, stride 2 | 7.76 ms | 0.757 m | 0.096 mm | 0.00206 rad |

The 16-sweep method is only 6.7% slower than the equal-time baseline but halves
sag, reduces position violation 8.6x, reduces angular violation 7.5x, and has
far lower residual speed. It remains finite for 300 frames at 7.78 ms/frame.
The older center-spacing metric is not a constraint error for bent hinged
capsules; the row-level articulation violations above are authoritative.

A free-chain CUDA graph test measures absolute momentum changes of 2.38e-7
linear (initial scale 3.06) and 9.61e-7 angular (initial scale 11.74), consistent
with float32 roundoff. This follows from applying every coarse correction as a
paired `J^T lambda` impulse. Correction stride three is not robust without
weakening the solve, so stride two is the current stability boundary.

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

Generalize the factor-2 local Galerkin correction from one path to packed
independent path islands/worlds, preserving one fixed graph-captured launch
shape. Measure scaling and add branched-articulation coverage, then determine
whether a related aggregation is safe for projected contact islands.
