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
have their source hinged flaps. The inlet, case-line, transport, and palletizer
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
start at their authored home positions; autonomous pallet-flow dispatch is not
implemented. The initial product stock and case supply are finite; continuous
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
remain colored blocks. A full-factory comparison measured 25.8 ms/physics frame
with this choice versus 31.8 ms with local blocks for the auxiliary mechanisms;
both passed the 22-second packing checks.

On an RTX PRO 6000 Blackwell, the complete factory measured **35.9 rendered FPS
at 1920×1080 output**, using default OptiX settings (DLSS quality
`performance`), over 900 frames following 120 warmup frames. Earlier measurements
were 8.7 FPS with per-mesh renderer updates and 28.8 FPS after batching those
updates, before switching the auxiliary mechanisms to exact solves. All use four
substeps, two solver iterations, the full geometry, and the same physical grip
reach. Setup and image capture are outside the timing window. Performance varies
with hardware and line state; this measures early packing, not unlimited steady
production. Lower substep/iteration settings failed actuator and packing checks.

A 130-second natural-flow validation sealed 27 cases and reached all four
palletizers, which placed 2, 1, 1, and 1 cases. Poses and velocities remained
finite, source joint-anchor error was approximately 0.2 mm, and palletizer,
merge, and transport-station controllers remained fault-free. Two inlet
holdbacks faulted late in the run, so strict long-run validation still fails:
continuous production needs the source profile's operator recovery and supply
replenishment described above. The documented 22-second test passes.
