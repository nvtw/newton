# Tail-local coupled solver audit

CPU/source inspection only; no new runtime strategy or GPU experiment.

## Actual scene graph

The source builder creates one 35-body Colibri spanning tree with 34 revolute joints. Four remaining closures are in the wings (two ball and two revolute joints). The full equality system has 188 rows including the two implicit drives.

The tail subtree consists of HummerBody -> TailMount, TailMount -> five feathers, and TailMount -> TailPinion: seven revolute joints, 35 structural rows, eight bodies including HummerBody. Its unconstrained root remains dynamic, yielding 48 - 35 = 13 independent instantaneous velocity dimensions for consistent independent rows. TailRack is a separate free body supported only by contact. The fan gear belongs to HummerBody. Tail fan/rack/feather coupling is contact coupling, not an authored joint loop.

HummerBody is also connected to Frame, head, shoulders and wing links. Treating it as fixed to isolate the tail would change physics. A tail solve must include its true inverse mass/inertia and transmit balanced reaction impulses into the rest of the mechanism.

## Existing capabilities and limitations

* DirectEqualitySystem accepts excluded_joint_mask and builds connected components from the retained rows. A retained tail subset can therefore factor 35 rows instead of the entire 188-row mechanism without changing source joints or coordinates.
* BlockJointSystem reuses the same structural wrenches, but the constructor currently chooses it OR DirectEqualitySystem globally. The world has one _direct_equality_system, and one constraints.bilateral block sidecar. A second owner needs explicit preparation/lifecycle and disjoint joint ownership.
* DirectContactResponse accepts active_mechanisms, but the mask selects complete components, not an arbitrary tail subset of the existing connected full mechanism. A new excluded-row topology is necessary.
* Direct contact scheduling supports contacts within one constrained component and contacts from it to an unconstrained external body. Thus TailRack can remain free and receive the opposite physical impulse. Contacts spanning two distinct constrained components are deliberately not owned by this path.
* Existing reduced ownership is articulation-level: it consumes complete model.articulation_start/end ranges, plus applicable loop joints. _get_reduced_model only filters prescribed-motion trees; it does not select the tail. The current Colibri tree cannot be reduced only at the tail through an option.
* block_pgs currently requires maximal coordinates. Constructor-owned color groups and contact chunks also require block_pgs. A hybrid tail-direct/rest-block mode is not exposed or automatically configured.
* Existing joint masks distinguish skipped, ordinary, and prepare-only rows, but changing masks alone does not install the missing constrained contact mobility, projection, or state exchange.

## Most bounded next experiment if coverage is exonerated

Use a small maximal-coordinate tail DirectEqualitySystem selected by excluded_joint_mask, plus ordinary block callbacks for all other joints. Retain the original bodies, masses, poses, contact points, and drive rows. Interleave:
1. ordinary block/contact copy sweeps and physical averaging;
2. tail equality projection on the original body velocities;
3. tail-owned contact sweeps using tail-constrained mobility and equal/opposite impulses on external bodies.

Never solve the same joint/contact row through both paths. Rebuild direct response at the correct configuration, retain positional/physical bias separation, and preserve projected warm starts. All contacts touching a tail component body (including HummerBody) must be classified consistently; selecting only the offending pair would omit reaction coupling. Boundary joints outside the tail can reintroduce tail residual, so repeat the prescribed coupled sweep sequence rather than treating the last projection as a convergence proof.

The existing direct machinery makes this more bounded than splitting model articulation coordinates. Nevertheless, preparing both owners, routing contacts, coloring copy nodes, and property refresh need implementation and tests. Factorization cost may fall markedly with 35 instead of 188 rows, but per-contact responses and all HummerBody contacts can dominate; no speed claim follows from row count alone.

A larger local multi-joint block without constrained contact response is a cheaper preconditioner, but by itself does not resolve the contact/joint competition. A reduced tail with a dynamic six-DOF root is a longer-term option requiring articulation regrouping and mixed integration ownership, not a safe one-line change.

## Required acceptance evidence

Use the frozen failure to verify all required contacts were admitted first. For a mixed-solver prototype, test six-component momentum and contact energy, exact row/contact coverage, independent boundary-body reaction, finite motor bounds, changing contacts, and original high-mass-ratio direct behavior. Then repeat the failing trajectory with fresh geometry and joint residual checks. No altered offsets, dropped rows, modified masses, damping, or weakened drive targets.
