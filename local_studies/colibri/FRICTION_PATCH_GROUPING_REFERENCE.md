# Explicit-connectivity history grouping: bounded CPU reference

**History patches and coupled solve groups are different.** Disconnected
contacts can safely belong to one coupled solve group when every point's
constraint, Coulomb cone and physical application point is retained. The
connectivity restriction here concerns sharing a material history field, not
permission to solve those equations together. No force or solver change.

## Deterministic graph rule

`friction_patch_grouping_reference.py` accepts explicit undirected connectivity
edges, compatibility keys(world, ordered bodies, material identity), FP32 unit
normals and stable ordering IDs. It picks the lexicographically smallest
unassigned key/ID, then traverses only supplied edges between compatible
unassigned points within that seed normal's cone. Repeat until all points are
assigned. Groups are connected; their representative stays fixed, so a normal
chain cannot gradually turn the group's frame. Members/groups are sorted by
stable key/ID. IDs order points only: they do not prove persistent material
identity between frames.

No spatial radius, contact truncation, whole-body merging, or invented edges.
No edges means one group per point. Invalid indices, repeated IDs within a key,
nonunit/nonfinite normals are rejected. This is CPU semantic code, not a new
core API or production device algorithm.

Five tests PASS:20 input/edge reorder controls, disjoint same-key components,
normal chain, world/body/material/opposite-normal separation,257-contact chain,
257 isolated points, empty and invalid input. All point membership is retained.
`/tmp/colibri_patch_grouping_reference.log`.

## Actual collision inputs available

`contact_ingest.py` supplies pair runs and contact match remaps, not contact-region
adjacency. Shape/body pair membership alone cannot certify shared history.

For Colibri mesh/ground, `narrow_phase.py:1919-1952` and2047-2083 emit candidates
from mesh vertices and set `ContactData.sort_sub_key = vertex_idx`, including
its reduction path. Body-frame witnesses and shape IDs reach Contacts. Exact
feature subkeys are encoded in private deterministic sort buffers
(`sim/collide.py:317-325,3050-3094`); they are not the generic
`rigid_contact_point_id` allocation. Any caller must transport their exact
permutation through reduction/sticky matching rather than decode an unrelated
array or infer IDs from floating-point nearest vertices.

Source mesh triangles are available in `model.shape_source` and mesh handles;
`model.mesh_edge_indices`/`shape_edge_range` provide edges. Source builder
`example_kamino_colibri.py:2273-2289` preserves source faces while transforming
vertices. For mesh/SDF contacts, edge-derived subkeys identify source edges
(`sdf_contact.py:1523`), but do not identify the other shape's support topology.
Primitive contacts require their own supplied feature/connectivity contract.

## Saved Colibri ground-contact evidence

In corrected final native state, shape46 FrameGround/Base contributes52contacts;
shape34 Frame/Cylinder contributes15. Base.obj contains4598vertices,4630faces,
24 raw index-connected components, but only2299 unique FP32 positions. Exact
position identification yields one component: the raw index disconnections are
consistent with duplicate-position seams. Diagnostic nearest-vertex matching
of saved witnesses spans7 raw components with maximum distance1.12e-7m;
this is not an exact feature provenance replacement.

Artifact `/tmp/colibri_ground_mesh_connectivity_inputs.json`.

A future material-history caller needs connectivity of the **current contacting
region**, not the whole welded mesh. Paths through elevated/noncontacting
bridges may connect two physically separate contact islands. For a known plane,
source faces clipped against the actual existing contact-generation envelope
could supply contact-region topology, with exact feature IDs carried through
reduction. That is an explicit further implementation/validation task; no
such connectivity is fabricated in this reference. General shape pairs need
both-side material support correspondence. Work stops here pending a real
caller; no device adapter was added.

## Conservative history storage limitation

The additive core history container's complete-membership rebirth is an initial
safety gate. Contact churn can reset the whole shared reference, so this is not
a proven cure or desired final persistence model. A later model should preserve
matched existing material anchors while giving new/rolling points appropriate
fresh births, with a physical feasibility/invalidation contract. It must not
retroactively apply an old constraint to a new material point.
