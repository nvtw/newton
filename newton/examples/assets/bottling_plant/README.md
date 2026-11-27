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

The nine warehouse rack rows have fixed beam and upright colliders. All 1,282
pallets stored on them are movable and have the source A08 fork pockets.
The 1,063 wrapped pallets use the source 308.068 kg rigid unit load. Each of
the 5,882 filled cartons on the 98 open and 102 partial pallets is a separate
rigid box at its source placement. The other 19 pallets are empty. Rack stock
settles under gravity and uses normal island sleeping and waking. The 376
floor-stock instances are also independently movable: 188 empty pallets, 104
wrapped finished-goods loads and 84 packaging-supply loads. The latter fit the
warehouse cargo geometry to boxes and use the corresponding packaging assets'
payload masses and friction, with separate mass and collision geometry for the
wood pallet. The two finished-goods pallets in front of the fleet forklifts
start free on the floor. W4's eight return stacks and its damaged stack contain
106 independent A08 pallets and one loose board. The W1 inspection cartons
rest on their own movable pallet. Parked forklifts and warehouse fixtures
retain their source colliders. Reach-truck storage and retrieval use timers
in the source; the Newton fleet has no automated high-rack mission.

Product bodies stored below the floor are inactive supply pools and are omitted.
The 2,458 bottles already on the line and all 56 open cases remain dynamic. Cases
have their source hinged flaps. Behind the four palletizing robots, all 302
reserve pallets in the two floor stores are separate 61.196 kg bodies. Their
32 stacks rest on the floor, with fork pockets formed by the source boards.
All 60 dispenser pallets use the same authored dynamic variants and compound
colliders. The separator fingers retain
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

The default 400 cartons need 4,800 bottles. The line starts with 2,458 and
reserves the remaining 2,342 for the upstream IFM. It admits up to seven bottles
per row at the source full-line rate of 425 bottles/min, subject to the PE-A1
permit and physical clearance at the inlet. Each bottle retains all eight
source collision hulls, mass, contact materials and visual geometry. Bottles
already in cartons stay there. `--extra-bottles N` caps admissions at N or the
remaining carton demand, whichever is smaller. `--extra-bottles 0` runs the
original stock without allocating reserve geometry. Operator jam recovery
remains manual, and inlet faults fail validation.

With `--filling-room-physics`, the pasteuriser outfeed follows the source.
Its 0.24 m chute opens onto one straight conveyor through the dryer to the
accumulation table. The source authors this conveyor as 0.415 m mat tops from
x 30.6 to 37.3 m. Its green rails are parallel, 0.454 m apart, in one band
105-135 mm above the deck. The example reads the conveyor's ends and rails from
the source. The source has no single filer and no narrowing before the
accumulation table, which feeds the IFM as a mass. It places a single file in
the dryer only as stock. The source authors no belt speeds. The conveyor runs
on at the chute's speed, a modelling choice, so the stream stays loose and
widens without a speed step. Its deck extends under the rails, like the
example's other tables. Stops use the plant's standing-queue eyes: a queue at
the conveyor's entrance stops the pasteuriser bed, and a queue at the
accumulation table's entrance stops the conveyor.

`--pasteuriser-scale N` is a benchmark and visualization option, not source
data. It enables filling-room physics and widens the pasteuriser bed N times,
leaving the default factory unchanged. The source bed is 2.3 m wide and 24.6 m
long from funnel to funnel, about 60 m² holding 16,900 bottles. The building
wall behind the infeed and the dryer, accumulation table and IFM downstream fix
its length, so only its width grows. The chutes keep their 0.24 m width,
matching the single file and the outfeed rails. The funnels keep their source
angle, lengthening with the bed's width; a steeper outfeed funnel let the
packed bed arch across the chute and stop the line. The infeed and outfeed
tables and the tunnel housing are widened across the line by the width added to
the bed, so the housing keeps its source clearance; the hot-water pipes from
the heat-exchanger station end at its widened wall. Like a built tunnel
pasteuriser, the housing hides most of the bed; `--no-pasteuriser-housing`
leaves it out to show the bed. The packed bed fills the wider deck from the
start. As in the source, the belt speed carries `PASTEURISER_CAPACITY` times
the rated 425 bottles/min as a packed bed of the bed's width, so a wider bed
runs slower and holds the product longer:

| N | Bed width | Bed area | Bed bottles | Belt speed | Residence |
|---|-----------|----------|-------------|------------|-----------|
| 1 | 2.3 m     | 60 m²    | 16,871      | 0.70 m/min | 36 min    |
| 2 | 4.6 m     | 114 m²   | 32,801      | 0.35 m/min | 64 min    |
| 3 | 6.9 m     | 163 m²   | 46,683      | 0.23 m/min | 85 min    |
| 4 | 9.2 m     | 206 m²   | 59,283      | 0.17 m/min | 100 min   |

The residence time is the packed bed's transit between the funnels. The widened
table visuals keep the source funnel shape, so their funnels are steeper than
the physical ones, which run inside them.

Built tunnel pasteurisers run 2 to 7 m wide and 11 to 30 m long
([Sidel](https://www.directindustry.com/prod/sidel/product-8751-1527545.html);
KHS Innopas: 2.6 to 7.2 m processing widths,
[datasheet](https://www.khs.com/fileadmin/user_upload/Datenblaetter-Produkte/Einzelmaschinen/Downloads__PDFs_/Innopas_PI-IISC_tunnel_pasteurizer.pdf)).
A KHS Innopas P II SG 30-200, rated 24,000 to 25,000 bottles/h like this
line, has a 113.5 m² treatment area and treats 0.7 l bottles for 48.6 min
([listing](https://www.exapro.com/khs-innopas-p-ii-sg-30-200-p260827141/));
total treatment cycles of about 40 to 80 min are common. N = 2 thus matches
a built machine of this rating; its 6.3 m housing clears the heat-exchanger
station to the south but touches the filler's control cabinet to the north.
N = 3 reaches the widest built beds but holds the product longer than usual,
and its housing overlaps the heat-exchanger station. N = 4 exceeds built
widths, and its housing reaches the building columns. Only visuals overlap:
the machine colliders there end below the deck, and the bed's guards stay
clear of all equipment.

Headless measurements on one RTX PRO 6000 Blackwell, 120 frames after 30
simulated seconds, with the OptiX viewer at 1920 by 1080 (one sample, three
bounces):

| N | Line bottles | Contacts | Physics      | With OptiX   |
|---|--------------|----------|--------------|--------------|
| 1 | 21,020       | 288,000  | 57.0 FPS     | 44.7 FPS     |
| 2 | 36,950       | 520,000  | 42.8 FPS     | 35.7 FPS     |
| 4 | 63,432       | 918,000  | 26.0 FPS     | 23.0 FPS     |

Over these 34 seconds every scale drained 244 to 291 bottles from the bed
onto the outfeed conveyor (280 at N = 1), the infeed kept feeding the bed, no
wet-end bottle tipped or dropped below the deck, and the per-second product
containment checks passed.

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
remain colored blocks. Each 60 Hz frame runs two collision-and-solver steps
at 120 Hz, each with 12 temporal contact sweeps (contact steps of 1/1440 s) and
an 80 Hz contact spring; heavy bottles in a light carton only come to rest with
such short contact steps. Contact columns that press on a light body shared by
many contact rows, such as a packed carton, take half of each sweep update;
without this, fewer than 24 sweeps let packed cartons jitter. Control runs once per frame.
`--substeps` sets this collision-and-solver step count. Refreshing contacts only
at 60 Hz allowed bottles to penetrate thin case floors during lifting, even
with four internal solver substeps. Sealed cases must retain all twelve bottles;
the 130-second test additionally requires a delivery from every palletizer.

Moving conveyors retain triangle contacts on their cooked collision hulls.
Using convex contacts on the same geometry caused loaded cases to catch at
descending-ramp transitions. A small physical replay reproduced the failure;
restoring triangle contacts restored case transport.

A 130-second run with 50 cartons per packer, full visual geometry and the
default feeder admitted 623 new bottles. All eight packers placed 88 bottles
each, and all 48 sealed cases retained twelve bottles. The four palletizers
placed 4, 3, 3 and 2 cases respectively. The largest joint-anchor error was
1.9 mm. Product containment was checked every simulated second.
No 60 rendered FPS result has been validated for sustained end-to-end production.

The factory defaults to regular colored sweeps. Use `--mass-splitting` to
test the CUDA temporal PGS overflow path. It uses twelve regular colors by
default, with the cap set by `--max-colored-partitions`. Timing output
includes the actual partition count and overflow size.

In a matched pair of 160-second headless CUDA runs, both configurations
sealed 64 cartons and placed the same 26 cartons in the same slots. With
overflow enabled, 12 cartons tilted more than 2 degrees. With regular
colored sweeps, all 26 stayed within 2 degrees. The worst carton excursion
relative to the pallet from one to five seconds after release dropped from
9.29 mm to 0.14 mm with regular sweeps. The pre-stacked two-carton rest test
passes with both configurations.

An initial twelve-color experiment failed the captured real-carton stability
test because overflow copies changed the effective normal compliance. After
preserving physical compliance across copies, the same test measured 0.15 mm
relative pose excursion with twelve colors, versus 0.19 mm without splitting,
with no bottle-to-floor contacts through the carton. Force and momentum checks
also pass. This does not establish stability of multiple stacked filled cartons.

A matched short full-factory comparison with the compliance correction measured
30.66 ms/frame without splitting and 27.37 ms/frame with twelve regular colors
plus overflow. With the legacy OptiX viewer at 1920 by 1080, one sample and three
bounces, the corresponding measurements were 31.22 and 29.26 ms/frame
(32.03 and 34.18 FPS). Each measurement used 120 timed frames following 600
physics warmup frames; rendering added 30 warmup frames. These are single paired
measurements in the early production window, not sustained production results.
Both controller reports matched at 12.5 simulated seconds.

A later support-footprint regression exposed an independent solver-side contact
cap, which could retain just one approaching support point. Removing that cap
preserves the collision pipeline's manifold and supersedes the performance
comparison above: a matched early factory pair increased from 30.84 to
35.89 ms/physics frame while solving 67,779 rather than 51,182 contact points.
The temporal joint-motion correction has no measurable additional early-frame
cost. Corrected rendered throughput remains unmeasured. Canonical collision
body-pair reduction with latest matching passed bounded stack, force, and creep
controls, but increased factory physics time from 36.32 to 37.45 ms/frame and
remains disabled. Filled cartons using runtime tape joints still fail the strict
rest test, despite the passing equivalent rigid-compound control.

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
