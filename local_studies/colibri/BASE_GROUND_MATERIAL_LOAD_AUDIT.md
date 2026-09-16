# Base/ground material and support-load audit

CPU-only source/asset audit; no Warp/GPU initialization or production changes.

## Material and geometry

Original source: /home/twidmer/Documents/colibri/Colibri.usd.
Fresh data: /tmp/colibri_base_ground_source_audit.json and
/tmp/colibri_material_audit.json.

* FrameGround/Base binds PhysicsMaterial_Heavy_Friction: authored static and
  dynamic friction both 0.5, restitution 0, density approximately 500 kg/m³.
  All six authored physics materials have equal static/dynamic friction.
* The extraction table deliberately carries dynamic friction only. Phoenx
  Solver._sync_materials copies model.shape_material_mu into BOTH static and
  dynamic material fields. This API limitation does not lower Base static
  friction for this asset.
* The public example uses shared build_scene unchanged, then modifies only
  contact gaps. Ground uses default_shape_cfg.mu=0.5; Base overrides to 0.5.
  Solver default combine mode is average; contact ingestion resolves static
  and dynamic independently and writes both to contact column headers.
  Thus actual Base/ground has mu_s=mu_k=0.5, agreeing with saved native headers.
* The USD CollisionPlane has PhysicsCollisionAPI, collisionEnabled=true,
  no bound physics material, no filtered-pair relations, and no authored
  material-combine override. Its effective external PhysX fallback is NOT
  established from this USD alone; do not claim a source-authored ground 0.5.
* Source ground transformed Y normal is approximately (2.73e-9,1.87e-10,1);
  transformed elevation is 0.002965275 m. Python builds the same infinite
  Z plane at that height (negligible source rotation-rounding difference).
  FrameAsm/Plane is an invisible visual mesh, not the collision ground.
* Base source SDF mesh has 4598 vertices, resolution128/subgrid6.
  Python Base.obj has4598 vertices and matches source vertices after authored
  1.5 scale and centimeter-to-meter conversion within7.53e-9 m.
  Python uses its own SDF construction, not the USD cooked SDF buffer.
  Vertex agreement does not assert identical discretized SDF/contact samples.
* Base is a dynamic free root; no world fixed joint is created by default.
  The builder has no explicit source-ground filter. Non-fixed world joints
  default to collision_filter_parent=false; saved actual Base/ground contacts
  establish that this pair is admitted.

## Coupled support exposes incorrect recovery subtraction

Current contact_projection._friction_normal_lambda uses
max(0, lambda_n + mass_coeff*eff_n*bias_n*SOR), capped by lambda_n.
For penetration bias=-b and SOR1 it subtracts the ISOLATED row recovery
mass_coeff*b/Aii. This is not the recovery impulse of a coupled manifold.

Independent CPU result: /tmp/colibri_two_support_load_separation.json.
A=[[1.01,.99],[.99,1.01]] is strictly SPD, eigenvalues2 and.02.
For gravity velocity G=.001 m/s and recovery b=.02 m/s:

* Hard rows: biased impulses .0105 Ns each; physical support .0005 each;
  true coupled recovery .01 each. Local subtraction .019802 Ns each
  incorrectly makes both friction loads zero.
* Actual default Nyquist softness: mass_coeff=.9417004454,
  impulse_coeff=.0582995546. The converged operator is
  A+diag((impulse_coeff/mass_coeff)*Aii).
  Biased impulses .01018168046 each, physical .00048484193 each,
  recovery .00969683853 each. Local subtraction again makes BOTH loads zero.
  Therefore this defect survives exact normal convergence and current softness.

A meaningful rigid analytical regression can use a 1kg body, Iyy=1kg m²,
two ground contacts x=+/-0.1m,z=-0.1m, and external COM impulses
(+.0001,0,-.001) Ns. Physical sticking impulses are tangential -.00005 Ns
at each point and normal .00045/.00055 Ns, balancing all force/torque.
Each lies strictly inside mu=.5 disks. Net body momentum is stopped by
equal-and-opposite recorded ground reactions; kinetic loss is dissipative.
Separate recovery-only control must give zero physical friction capacity.
The test must audit actual production updates and physical/recovery budgets,
not simply replace the disk radius by biased total normal lambda.

A coupled unbiased normal solve (same contact set/operator) distinguishes
physical support from recovery for this fixed-active linear example.
General unilateral activation changes and finite iteration/softness require
explicit impulse/velocity ownership; subtracting two arbitrary unconverged
per-row estimates is not automatically a valid implementation.
This diagnostic proves the current scalar formula can erase real support.
It does not yet prove which replacement is safe in the full live solver.

## Clarification: two valid channel semantics

Zero recovery-only friction applies ONLY to explicitly separated physical/pseudo velocity. Current bias changes real velocity; total accumulated normal impulse is internally consistent with a stabilized discrete Coulomb law. Normal recovery work can be positive while tangential friction remains dissipative. Existing diagonal subtraction is not an exact coupled decomposition. The unbiased reference is not a mandatory acceptance gate for total-normal friction. As penetration/dt vanishes, bias vanishes; penetration tending to zero alone is insufficient if penetration/dt remains finite.
