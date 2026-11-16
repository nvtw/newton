# Berlin bottling plant

This example reads the access-controlled GTC Berlin bottling plant assets. The
assets are not redistributed in this repository. Keep their directory structure
and source license/attribution files intact.

The verified simulation subset contains **2,509 files, 2.33 GB**. It includes
`README.md`, `bottling_plant.usda`, `assets/`, `materials/`, `plant/lines_1x8/`,
`scenarios/line_sim/`, `scenarios/logistics/`, and `sim/`. Source working
directories and authoring applications are unnecessary. Access requires the
asset owner's NVIDIA Storage authorization; the supplied `fetch_bottling_plant.py`
uses NVIDIA SSO and supports `--dry-run`, resumable downloads, and size checks.

Put the resulting `bottling_plant/` directory under `~/newton_local_assets/`,
set `NEWTON_LOCAL_ASSETS` to its parent, or pass `--asset-root`:

```bash
uv run -m newton.examples phoenx_bottling_plant
uv run -m newton.examples phoenx_bottling_plant --asset-root /path/to/bottling_plant
uv run -m newton.examples phoenx_bottling_plant --viewer null --test --num-frames 1320
uv run -m newton.examples phoenx_bottling_plant --viewer null --test --num-frames 7800
uv run -m newton.examples phoenx_bottling_plant --viewer null --timing-frames 600
```

The example-specific OpenUSD adapter composes the running line and forklift
fleet in an anonymous session layer. It reads source meshes, primitive shapes,
materials, mass/COM/inertia, joint frames, limits, force-limited drives, and
surface velocities. It shares geometry between repeated objects and merges
point-instancer prototypes by material, retaining their placements. The complete
factory remains visible; `--no-visuals` removes rendering geometry for physics
measurements. `--no-cutaway` restores the source exterior variant.

Product bodies stored below the floor are inactive supply pools and are omitted.
The 2,458 bottles already on the line and all 56 open cases remain dynamic. Cases
have their source hinged flaps. All 60 existing dispenser pallets use their
authored dynamic variants and compound colliders. The separator fingers retain
their exact source triangles and the station's contact material. Coarse magazine
proxies are retired only after their physical stock is installed. The wrapper
ports start empty, matching the source live scenario: its layout-only loads and
their proxies are removed together, without removing runtime products.

The inlet, case-line, transport, and palletizer
PLCs and motion libraries are read from `sim/`; Kit and Isaac Sim are not needed.
Vacuum grasps require ray hits on the actual product surfaces within the source
reach (6 mm for bottles, 12 mm for cases), then create physical constraints at
the acquired relative pose. Products are carried and released dynamically.

Computed collider moments are integrated near their geometric centers to avoid
precision loss in thin off-origin meshes. Explicit source tensors are retained;
`--correct-inertia` opts into Newton's correction of source approximations that
violate the inertia triangle inequality. Angular drive gains are converted from
USD degrees to radians as specified by the
[OpenUSD drive schema](https://openusd.org/release/api/class_usd_physics_drive_a_p_i.html).

The runtime adapter does not execute arbitrary embedded Kit scripts. Forklifts
start at their authored home positions. Dispatch uses the source navigation and
pickup/place sequences with measured physical inventory: completed pallets go
to an available wrapper infeed, and empty pallets can replace them only when
physical stock exists at the corresponding supply port. Swept-route reservations
coordinate trucks. Pickup and placement require measured load feedback; the
source's pallet pose corrections and virtual stock pools are not executed.
Palletizers wait for forklift withdrawal before accepting a replacement.
These controls have isolated regression coverage; sustained autonomous physical
routes through the full factory have not yet been validated. Wrapper outfeed,
pallet dispensing, and outbound logistics remain incomplete.
The first empty-pallet extraction has been tested with all 15 source pallets
and physical separator supports. Stock remains supported, but the default
four-iteration setting permits excessive upper-stack drift. Higher iterations
stabilize this isolated stack; they are not yet a validated full-factory setting.
The initial product stock and case supply are finite; continuous
bottle-pool replenishment and the source profile's operator jam recovery are not
implemented. Inlet faults remain visible and fail validation.

Cases retain their physical contents. After all four flaps close and slow down,
fixed tape bonds form a rigid sealed assembly; the source instead swaps in a
sealed-case proxy. Both use rigid-body approximations of cardboard and tape.
Liquids, pneumatic air, flexible packaging, and
manufacturing processes inside decorative machines are not fluid/deformable
simulations. Source convex-decomposition/SDF authoring uses the available triangle
surfaces unless explicit convex hulls are supplied. The renderer uses scalar
roughness/metallic fallbacks where texture channels are unsupported.

The default solver uses exact maximal-coordinate solves for the authored
scalar-joint mechanisms. General D6 joints and runtime grasp/tape constraints
remain colored blocks. Each 60 Hz frame runs four collision-and-solver steps
at 240 Hz, with four solver iterations per step. Control runs once per frame.
`--substeps` sets this collision-and-solver step count. Refreshing contacts only
at 60 Hz allowed bottles to penetrate thin case floors during lifting, even
with four internal solver substeps. Sealed cases must retain all twelve bottles;
the 130-second test additionally requires a delivery from every palletizer.

Moving conveyors retain triangle contacts on their cooked collision hulls.
Using convex contacts on the same geometry caused loaded cases to catch at
descending-ramp transitions. A small physical replay reproduced the failure;
restoring triangle contacts restored case transport.

Before physical dispenser stock was enabled, the 130-second test with the
defaults above retained twelve bottles in each of
48 sealed cases. The four palletizers placed 3, 2, 2, and 1 cases, respectively,
with no PLC faults. Maximum joint-anchor error was approximately 0.3 mm.
This verifies that earlier finite production configuration, not the updated
stock or unlimited operation. Autonomous
logistics and continuous replenishment remain incomplete as described above.
No 60 rendered FPS result has been validated for sustained end-to-end production.

The wrapper arm and film carriage are reconstructed from disconnected components
of the source material meshes. A revolute drive physically parks the arm clear
of the infeed; a prismatic drive supports the carriage. The source supplies no
wrapper mass or drive properties. The reconstruction estimates metal parts as
3 mm shells (steel 7850 kg/m³, aluminium 2700 kg/m³), the film roll as solid
polyethylene at 920 kg/m³, and rubber at 1100 kg/m³. Torque is limited to
1000 Nm and lift force to 3000 N. These are explicit modeling assumptions,
not measured machine specifications. Film tension, deposition, cutting and
label application remain unimplemented; the reconstructed joints alone do
not establish a working wrapping process.

The wrapper drives explicitly accept both position and velocity targets.
Continuous rotation requires continuous angular targets; wrapping targets into
one revolution causes the position drive to reverse at the discontinuity.
A two-revolution tracking regression checks finite-torque rotation and lift
tracking, separately from wrapping-process validation.

A clearance audit found that the source overhead crossbeams intersect the
rotary arm and hub. The authored static collision proxies omit these overhead
surfaces, so successful homing against those proxies does not establish full
mechanical clearance. Adding the actual static triangle surfaces stalls the
uncorrected assembly. A diagnostic translation of the moving assembly down by
127 mm provides 10 mm overhead clearance and permits rotation; this placement
correction has not been applied to the example. Full wrapper clearance and
loaded-pallet wrapping remain unvalidated.
