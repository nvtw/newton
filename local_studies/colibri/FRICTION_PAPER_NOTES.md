# Supplied PhysX friction paper: implications for Colibri

Read `/home/twidmer/Downloads/RigidBodyFriction.pdf`, *Rigid Body Friction in
PhysX*, May 27, 2024, all 18 pages. The paper explicitly cautions that device,
solver, articulation and friction-mode combinations follow different paths;
it does not comprehensively describe strong-friction persistence.

## Useful checks for the running experiment

- Pages 9–11 describe two persistent anchors selected far apart, with the
  patch's summed normal impulse divided between them. This motivates the
  current patch experiment: per-point normal loads can redistribute without
  changing total support, changing each individual friction capacity and its
  reference-reset decisions. This is a hypothesis about our remaining creep,
  not an established causal result.
- Pages 6 and 11 distinguish a combined two-dimensional friction limit from
  independent tangent limits. Current GPU TGS uses the combined radial clamp.
  Copying the PGS independent-axis scheme would change the friction region.
- Pages 10–11 explain why granting both anchors the entire patch load can
  double sliding friction. The live hybrid divides its one normal budget by
  anchor count, matching the inspected GPU path.
- Pages 8 and 11 question the geometric correction coefficient also scaling
  velocity response. Current GPU source uses approximately 0.365148 at 30
  iterations. The literal diagnostic records that coefficient explicitly;
  it is not an acceptable hidden workaround for production PhoenX's SOR 1.
- Page 9 discusses a minimum normal load sometimes used for articulation
  friction. We must not manufacture normal force to hide sliding. No such
  force floor has been added to our candidate.

## Where current code is more authoritative

The paper describes history retention in terms of static friction never having
broken during the previous step. The inspected GPU `solveContactBlockTGS`
resets a local broken flag per solve, combines the anchors' flags and overwrites
the header. Therefore the port retains actual current-source semantics instead
of silently changing it to a step-wide latch.

The paper mentions material flags changing patch behavior, but the inspected
GPU TGS preparation explicitly halves friction coefficients for two anchors.
We must verify a flag's actual code path before treating it as an experimental
switch. Merely setting a scene or material option is not proof that it executes.

## Tests prompted or reinforced

1. One shared normal-load budget; no doubled sliding capacity.
2. Combined tangent limit and rotational/sliding work checks.
3. Static-to-sliding history transitions and disappearance/recontact reset.
4. Creep comparison with source coefficient versus a unit velocity multiplier,
   if the literal hybrid is promising, keeping recovery targets controlled.
5. Physical angular impulse and joint/contact work accounting. The literal
   independent-anchor reference is not assumed momentum-conserving.

The paper strengthens the case for this experiment. It does not establish
that patch friction is the remaining cause or that the literal PhysX model
should replace our physical formulation unchanged.
