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

## Packed multi-world path scaling

The coarse kernels now pack equal-length independent path islands with one
fixed 128-thread CUDA block per path. A two-path CUDA graph test checks the
full block-diagonal Galerkin operator and global momentum, so interpolation
and sparse-edge indexing cannot cross path boundaries.

For 96-link worlds with 16 coarse color sweeps and stride two, 60-frame replay
scales as follows:

| Worlds | Frame time | Mean tip sag | RMS position violation |
|---:|---:|---:|---:|
| 1 | 7.76 ms | 0.757 m | 0.096 mm |
| 2 | 8.86 ms | 0.769 m | 0.119 mm |
| 4 | 10.38 ms | 0.804 m | 0.106 mm |
| 8 | 10.94 ms | 0.791 m | 0.112 mm |

Eight chains therefore provide about 5.7x the one-chain throughput. For
comparison, classic 200-substep x 2-sweep PGS at eight worlds costs 8.29 ms but
has 1.556 m mean sag, 0.665 mm RMS position violation, and far larger residual
speeds. The coarse method spends 32% more time there but roughly halves sag and
reduces position violation 5.9x. Identical-world sag spread is also lower than
the classic baseline (0.056 m versus 0.111 m after 60 frames).

## Branched-tree coarse-space evidence

A 104-joint trunk with three 24-joint arms was extracted as a 312-row physical
ball-socket operator. Its condition number is 1.52e4. Alternating-depth
interpolation that averages a fine branch joint across its parent and children
fails even with an exact coarse solve (12-cycle residual 1.22--1.35 versus
0.327 for SGS). The successful basis is simpler: retain odd-depth and leaf
joints, and assign every omitted even-depth joint wholly to its retained parent
aggregate. That factor-2 parent basis reaches residual 0.00605 with an exact
coarse solve.

The 55-block parent coarse graph has 169 nonzero blocks and needs four colors.
Fixed local coarse work gives:

| Coarse method | 12-cycle residual |
|---|---:|
| 8 color sweeps | 0.252 |
| 16 color sweeps | 0.224 |
| 32 color sweeps | 0.202 |
| depth-2 Anderson, 8 symmetric cycles | 0.158 |

Anderson improves the endpoint but oscillates strongly, while the fixed colored
method is monotone after its initial load-spreading transient. Depth-stride 4
and 8 parent aggregates regress even with exact solves, so recursive coarse
aggregation is not automatically beneficial. The next GPU implementation
should use the factor-2 one-hot parent aggregate, deterministic coarse graph
coloring, and local block sweeps; it must not reuse the failed multi-child
"smooth" interpolation.

## Generic parent-aggregate GPU correction

The one-hot parent aggregate is now implemented for arbitrary sparse bilateral
joint graphs. Topology setup precomputes fine-to-coarse mapping, canonical
coarse edges, CSR adjacency, and deterministic coarse colors. Runtime uses
three fixed graph-captured kernels for local Galerkin assembly, colored block
solve, and prolongation. On a 26-joint test tree, the GPU coarse operator
matches `P^T A P` to 1.2e-8 relative error and the RHS exactly. A free-tree CUDA
graph test verifies linear and angular momentum preservation within float32
tolerance.

A realistic 104-joint revolute scene uses the original capsule inertia: a
32-link trunk with three overlapping 24-link arms attached to the dynamic
branch body. Sixty-frame results are:

| Method | Time | Max sag | RMS position violation | RMS angular violation | RMS speed |
|---|---:|---:|---:|---:|---:|
| classic 100 x 4 PGS | 4.97 ms | 1.275 m | 0.285 mm | 0.00570 rad | 1.79 m/s |
| classic 200 x 2 PGS | 8.14 ms | 1.804 m | 3.292 mm | 0.0609 rad | 59.3 m/s |
| aggregate, stride 4, 16 colors | 6.90 ms | 0.999 m | 0.250 mm | 0.00499 rad | 2.14 m/s |
| aggregate, stride 3, 16 colors | 8.16 ms | 0.726 m | 0.279 mm | 0.00521 rad | 2.11 m/s |
| aggregate, stride 2, 16 colors | 10.31 ms | 0.419 m | 0.101 mm | 0.00235 rad | 1.39 m/s |

At identical time to the 200-substep baseline, stride three cuts sag 2.5x,
position violation 11.8x, and RMS speed 28x. Stride four is 15% faster than that
baseline while improving every constraint metric. It remains finite through
300 frames (6.94 ms/frame, 1.077 m sag, 0.267 mm RMS position violation).

The earlier co-located point-mass tree is retained only as a topology probe;
it was too well-conditioned physically for a coarse correction to help. The
realistic revolute tree is the acceptance benchmark.

## Core integration and general-graph fallback

The path and parent-aggregate solvers now run through the normal PhoenX
constructor instead of benchmark monkeypatches. articulation_coarse_mode
selects auto, path, tree, or graph; auto recognizes path forests and rooted
trees before falling back to deterministic independent-set interpolation. The runtime
remains three fixed CUDA kernels: Galerkin assembly, colored local block solve,
and prolongation. Contacts and other constraint families remain on fine PGS.

Mixed articulation types use the common leading row prefix. For PhoenX's
supported joints this includes the three point-attachment translation rows,
while revolute/fixed angular rows continue through fine PGS. CUDA graph tests
cover mixed revolute/ball-socket paths, rooted trees, cyclic graphs, paired
impulse momentum, and an articulated box chain with active ground contacts.

After promotion, the 96-link path runs at 8.34 ms/frame with 100 substeps, two
symmetric PGS iterations, 16 coarse color sweeps, and correction stride two.
At 60 frames it has 0.673 m tip sag, 0.101 mm RMS position violation, and
0.00239 rad RMS angular violation. The realistic 104-joint tree with stride
four runs at 8.29 ms/frame, with 0.600 m sag and 0.146 mm RMS position
violation. These measurements include the public integration path.

A 96-joint cyclic ball-socket matrix (condition about 8.0e3) tests the generic
fallback. After 12 outer cycles, block SGS leaves relative residual 0.167;
adjacent aggregation with 16 coarse color sweeps leaves 0.115. A topology-only
independent-set interpolation improves this to 0.095. Its three-kernel GPU
implementation matches the explicit Galerkin operator and preserves momentum
under captured replay, so it is now the general-graph fallback.

## Motorized closed-loop acceptance scene

A public-Solver benchmark now builds a 32-link polygonal revolute loop with
realistic box inertia, a world motor, and an excluded closure joint. Auto mode
correctly selects the sparse graph interpolation path. At 120 frames, plain
100-substep PGS with two iterations becomes unstable, while the coarse method
remains finite. The equal-time comparison is more informative:

| Method | Time | RMS position violation | RMS speed |
|---|---:|---:|---:|
| plain PGS, 100 x 16 | 6.33 ms | 0.090 mm | 0.477 m/s |
| graph coarse, 100 x 2, 4 sweeps, stride 2 | 6.35 ms | 0.064 mm | 0.230 m/s |

At 300 frames the times remain 6.29 and 6.31 ms respectively. Plain PGS has
0.072 mm RMS position violation, while graph interpolation has 0.025 mm, a
2.8x reduction. Correction stride four diverges in this driven loop, confirming
that stride two is a stability requirement rather than only a quality setting.

The contact-heavy variant rests all 32 links on a frictional plane and retains
about 410 active contact points. At equal 13.1 ms/frame cost, plain 100 x 4 PGS
has 0.131 mm RMS bilateral position error and 0.475 m/s RMS speed. The 100 x 2
PGS plus four-sweep graph correction has 0.0528 mm error and 0.204 m/s speed.
Thus the bilateral coarse level coexists with warm-started unilateral fine PGS,
while reducing both error and residual motion by more than 2.3x.

## Heterogeneous robot smoke test

The public DR Legs PhoenX example now exposes the coarse controls. This asset
contains 31 bodies, 37 mixed joints, six closed loops, light parallel rods,
stiff motors, mesh/plane contacts, and graph-captured animation updates. Auto
mode resolves to graph interpolation and passes the 60-frame headless example
test at 40 substeps, two PGS iterations, and four coarse sweeps. Single-world
throughput is 259 FPS versus 293 FPS for the existing eight-iteration PGS
configuration. This is a stability and integration result only; the example
does not yet report row-error metrics suitable for an equal-quality claim.

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

Measure longer contact-heavy and multi-world runs, then evaluate physical
closed-loop mechanisms using sparse graph interpolation. Projected unilateral
contacts remain fine-level PGS until a coarse active-set treatment demonstrates
both complementarity safety and an equal-time benefit.
