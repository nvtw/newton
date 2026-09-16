# Incremental D6 / soft-normal accuracy reference

This is an isolated CPU numerical experiment, not a live solver change. It holds
all original tangential impulses fixed and retains every eligible normal row.
The saved biased Colibri fixture contains 12 body velocity coordinates, 6 joint
rows, and 67 normal rows. No claim about complete Coulomb convergence follows.

The prior absolute 18-unknown FP32 solve closed its equations but its returned
velocity differed from its independently reconstructed impulse response by
1.93044e-6. Overwriting velocity with that solution would hide a force error.

`colored_joint_normal_incremental.py` instead solves corrections around the
actual captured velocity, applies the computed impulse increment, and separately
rounds the accumulated multiplier. Refinement evaluates the original equations
using the resulting velocity. It does not reconstruct applied impulses by
subtracting two rounded accumulated multipliers.

The plain FP32 incremental variant still fails the unchanged 1e-8 scatter gate
(1.911e-7). Compensating sums alone reduces that error to 8.65e-8; FP32 product
rounding in cancelling contact/joint torques remains. Compensating both products
and sums reduces it to 1.63e-9 after three refinements. The CPU implementation uses
Dekker splitting; a CUDA implementation can evaluate each product remainder with
an explicit FMA. It must preserve these arithmetic operations when compiled.
There is no FP64 input to the iterative candidate after input conversion; FP64
is used exclusively by the independent audit and existing snapshot assembler.

On the captured biased phase, three refinements give:

- Original normal complementarity residual: 1.16e-9 m/s.
- Original joint residual: 3.97e-9.
- Applied impulse / stored velocity discrepancy: 1.63e-9.
- Net linear momentum ledger discrepancy: 2.85e-10 Ns.
- Maximum per-body spin-angular impulse ledger discrepancy: 9.65e-12 Nms.
- Kinetic energy change / midpoint impulse work discrepancy: 1.72e-12 J.

The angular number is a per-body impulse/scatter audit, not a separate proof of
world angular momentum conservation of the input Jacobians. No hidden velocity
projection, altered drive law, damping, SOR, or changed acceptance threshold is
used. The physical input model and Jacobians are unchanged.

Two unittest regressions pass: a small cancellation fixture and the saved
Colibri fixture with an explicit failing plain-scatter negative control. The
second test skips when that external snapshot is absent; it is not a portable
production regression yet. Both files pass Ruff.

Remaining work before a live candidate: replace the current seed's 14 active-set
solves with an efficient bounded implementation, port FP32 factorization/scatter,
handle hard normals, solve tangential coupling and original point cones, and
validate integration before position updates. No timing improvement or reduced
live creep has been demonstrated by this reference.
