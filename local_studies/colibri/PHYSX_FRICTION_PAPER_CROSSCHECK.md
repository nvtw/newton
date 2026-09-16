# Supplied friction paper versus current GPU TGS source

Read the user's RigidBodyFriction.pdf (May27,2024),18pages, together with current
PhysX source ed6e5ca2474c9c80ad4f4826591b88476779c6ef.

Actionable confirmations:
- Pages6/10–11 describe coupled2D tangent magnitude and equal distribution of
  patch normal load. Current GPU solverBlockTGS.cuh415–446 uses radial clamp;
  contactConstraintBlockPrep.cuh978–983 scales both friction coefficients by.5
  for two anchors. One patch therefore has one total capacity, not two.
- Normal rows precede friction. The hybrid retains that order, but retains
  Newton soft normal law and exact conditioned joint response: it is not fullSDK.
- Patch anchors maximize separation and preserve material references. Current
  source90–203 uses the quarter-bounding-diagonal retention criterion and
  preserves oldanchor0 when growing1→2. No need to invent extra yaw damping:
  two separated friction forces already provide a torsional couple.
- TGS relaxation factor min(.8,2/sqrt(positionIterations)) is source-based.
  It scales both error correction and velocity response; paper itself questions
  that choice. It is not evidence of exact Coulomb convergence at finite budget.

Important source/prose differences and limits:
- Paper says reuse requires never breaking during priorstep. Current GPU
  initializes broken=0 per solve, ORs across anchors and overwrites header.
  Latest-solve status is the actual source behavior; do not add whole-step latching
  merely from prose.
- PGS independent-axis box limits and friction-only-lastiterations are different
  paths; do not import them into the GPU TGS comparison.
- Current GPU TGS hardcodes two-anchor half scaling in the inspected path.
  Paper's suggestion to clear improved-patch flag is not proof this path changes.
- Paper explicitly warns equal anchor loads can impair static arrest when
  mu_dynamic<mu_static. Two-anchor patch friction approximates distributed
  contact friction; totalcapacity and momentum checks do not certify every
  original per-point Coulomb equation.
- Paper's discussion of biased normal load is a limitation, not authorization
  for isolated-row recovery subtraction or fictitious minimum support loads.
- Strong friction is not covered comprehensively; the document cannot certify
  long-term stationarity or physically correct material-reference updates.

Review of local hybrid found stale anchor retention when a patch disappeared.
Kepler fixed finish_refresh to clear unclaimed count/broken; independent CPU
no-contact test passed. Contact-only work is insufficient to certify work
through the compliant joint response; Kepler added actual joint impulse/wrench
capture. Independent frozen endpoint levers can create angular impulse defects,
which must remain exposed in physical acceptance rather than excused by source
fidelity.
